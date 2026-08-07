"""The Lyceum Cloud class.

Registered into `sky.utils.registry.CLOUD_REGISTRY` at import time. Kept
lightweight and stateless per SkyPilot's design rule for Cloud objects.
"""
from __future__ import annotations

import json
import posixpath
import re
import typing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sky import clouds
from sky import sky_logging
from sky.utils import registry
from sky.utils import resources_utils

from skypilot_lyceum import api as lyceum_api
# Imported as a MODULE, never `from ... import <fn>`: every catalog call below
# resolves the attribute at call time, which is what makes the single source of
# truth swappable (and observable to a `monkeypatch.setattr` on the module).
from skypilot_lyceum import catalog

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky.utils import status_lib

logger = sky_logging.init_logger(__name__)

_CREDENTIAL_FILES = ['api_key']

#: Derived from `api.API_KEY_PATH` rather than retyped, so the mount, the
#: reader and the `sky check` message can never name different directories.
_CREDENTIAL_DIR = posixpath.dirname(lyceum_api.API_KEY_PATH)

#: The SSH user baked into the Lyceum image (C1). The vendor docs say root;
#: root and ubuntu are both refused by the real image with
#: `Permission denied (publickey)`.
_SSH_USER = 'lyceum'

#: What an operator has to create when there is no credential. Named in one
#: place so `sky check`'s reason and the mount agree.
_CREDENTIAL_HELP = (
    f'Lyceum API key missing or rejected. Set $LYCEUM_API_KEY, or write the '
    f'key to {lyceum_api.API_KEY_PATH}. On a deployed SkyPilot API server, seed '
    f'that file from the deployment secret at container start.')

#: Anything long enough and opaque enough to be a credential. Used to scrub
#: exception text before it reaches `sky check` output, which is routinely
#: pasted into chat and captured in the API server's logs. The API layer
#: legitimately echoes request context, so this class is the boundary that has
#: to strip it.
_SECRET_RE = re.compile(r'[A-Za-z0-9_\-]{20,}')


def _redact(text: str) -> str:
    """Replace anything key-shaped in `text` with a placeholder."""
    return _SECRET_RE.sub('<redacted>', text)


@registry.CLOUD_REGISTRY.register
class Lyceum(clouds.Cloud):
    """Lyceum Cloud -- EU GPU cloud, VM API with SSH access."""

    _REPR = 'Lyceum'
    _MAX_CLUSTER_NAME_LEN_LIMIT = 120

    #: Same envelope as Shadeform. STOP is genuinely absent from the Lyceum API
    #: (terminate only); MULTI_NODE is enterprise-contract-only and not exposed
    #: through the public API; SPOT_INSTANCE is deferred to phase 6 -- spot
    #: provisioning is verified working, but preemption detection is not built.
    # yapf: disable
    _CLOUD_UNSUPPORTED_FEATURES: Dict['clouds.CloudImplementationFeatures', str] = {
        clouds.CloudImplementationFeatures.STOP:
            'The Lyceum API can only terminate a VM (DELETE /vms/{id}); it '
            'exposes no stop endpoint, so a stopped cluster could never be '
            'resumed. Use autodown instead.',
        clouds.CloudImplementationFeatures.AUTOSTOP:
            'Lyceum can only terminate a VM, so an idle timer can only ever '
            'delete it -- use autodown (`--down`, or `down: true`). Without '
            'this entry `sky launch -i N` (no --down) validates and then asks '
            'the node to STOP, which fails on every retry while the VM bills.',
        clouds.CloudImplementationFeatures.MULTI_NODE:
            'Multi-node clusters are an enterprise-contract feature on Lyceum '
            'and are not exposed through the public VM API.',
        clouds.CloudImplementationFeatures.SPOT_INSTANCE:
            'Spot instances are deferred to phase 6 on Lyceum: provisioning is '
            'verified working, but a reclaimed VM carries no spot-specific '
            'field, so preemption detection is not built yet.',
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            'Lyceum provisions a fixed disk with every hardware profile; the '
            'create API has no disk-tier field.',
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            'Lyceum exposes no network-tier selection on its VM API.',
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            'Lyceum attaches exactly one network interface per VM and offers '
            'no way to request more.',
        clouds.CloudImplementationFeatures.DOCKER_IMAGE:
            'Lyceum boots one fixed image; the create API takes no container '
            'or image specification.',
        clouds.CloudImplementationFeatures.IMAGE_ID:
            'Lyceum has no image catalog -- every VM boots the same vendor '
            'image, so an image ID could not be honoured.',
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            'Lyceum offers no object store, and no bucket-mount hook on the '
            'VM API.',
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            'Jobs and serve controllers must not run on Lyceum: it is '
            'single-node only, and there is no cloud-side TTL to stop a leaked '
            'controller billing (C5).',
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            'A high-availability controller needs stop/restart, which the '
            'Lyceum API does not provide.',
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            'Lyceum has no snapshot or volume API to clone a disk from.',
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            'Lyceum does not expose directly attached instance storage as a '
            'selectable resource.',
    }
    # yapf: enable

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.LAUNCH_ONLY

    @classmethod
    def _unsupported_features_for_resources(
            cls,
            resources: 'resources_lib.Resources',
            region: Optional[str] = None,
    ) -> Dict['clouds.CloudImplementationFeatures', str]:
        # `region` matches the base class (sky/clouds/cloud.py:766); omitting it
        # TypeErrors for any caller that passes it by keyword.
        del resources, region  # Lyceum's envelope is uniform.
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def max_cluster_name_length(cls) -> Optional[int]:
        """PUBLIC name, deliberately.

        `sky/backends/backend_utils.py:761` calls `cloud.max_cluster_name_length()`
        and the base class (`sky/clouds/cloud.py:163`) defines only that public
        name -- there is no `_max_cluster_name_length` anywhere in the base class.
        Shadeform defines the private one (`sky/clouds/shadeform.py:88`), so its
        120-char limit is DEAD CODE upstream: `Shadeform.max_cluster_name_length()`
        returns None today. Copying that idiom would silently disable the limit
        here too -- which matters more for Lyceum than for anyone, because
        `display_name` (bounded by this limit) is the entire identity scheme.
        """
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    @classmethod
    def regions_with_offering(
            cls,
            instance_type: str,
            accelerators: Optional[Dict[str, int]],
            use_spot: bool,
            region: Optional[str],
            zone: Optional[str],
            resources: Optional['resources_lib.Resources'] = None,
    ) -> List['clouds.Region']:
        # Trailing `resources` matches the base class (cloud.py:200).
        del instance_type, accelerators, use_spot, resources
        # Lyceum is a single EU footprint with no region or zone concept, so
        # the answer does not depend on the catalog: there is exactly one
        # synthetic region, and it exists only to satisfy
        # `sky/catalog/common.py`. Advertising more would make SkyPilot retry a
        # capacity failure (a fast, free 500 -- C7) once per fictional region
        # instead of failing over to another cloud.
        if zone is not None:
            return []
        regions = [clouds.Region(catalog.DEFAULT_REGION)]
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def zones_provision_loop(cls, *, region: str, num_nodes: int,
                             instance_type: str,
                             accelerators: Optional[Dict[str, int]] = None,
                             use_spot: bool = False
                             ) -> Iterator[Optional[List['clouds.Zone']]]:
        del num_nodes  # Lyceum is single-node; see MULTI_NODE above.
        regions = cls.regions_with_offering(instance_type,
                                            accelerators,
                                            use_spot,
                                            region,
                                            zone=None)
        for r in regions:
            # `None` is what "this cloud has no zones" looks like to the
            # provisioner; one yield means one provision attempt.
            assert r.zones is None, r
            yield r.zones

    @classmethod
    def get_vcpus_mem_from_instance_type(
            cls, instance_type: str) -> Tuple[Optional[float], Optional[float]]:
        return catalog.get_vcpus_mem_from_instance_type(instance_type)

    @classmethod
    def get_accelerators_from_instance_type(
            cls, instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
        return catalog.get_accelerators_from_instance_type(instance_type)

    @classmethod
    def get_default_instance_type(cls, cpus: Optional[str] = None,
                                  memory: Optional[str] = None,
                                  disk_tier: Optional[str] = None,
                                  local_disk: Optional[str] = None,
                                  region: Optional[str] = None,
                                  zone: Optional[str] = None,
                                  use_spot: bool = False,
                                  max_hourly_cost: Optional[float] = None
                                  ) -> Optional[str]:
        return catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            disk_tier=disk_tier,
            local_disk=local_disk,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost)

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def instance_type_exists(self, instance_type: str) -> bool:
        return catalog.instance_type_exists(instance_type)

    def instance_type_to_hourly_cost(self, instance_type: str, use_spot: bool,
                                     region: Optional[str] = None,
                                     zone: Optional[str] = None) -> float:
        # `region`/`zone` are forwarded even though there is one region today:
        # the base class and sky/optimizer.py both pass them by keyword, and
        # dropping them is how a future multi-region catalog would start
        # quoting the wrong region's price.
        return catalog.get_hourly_cost(instance_type,
                                       use_spot=use_spot,
                                       region=region,
                                       zone=zone)

    def accelerators_to_hourly_cost(self, accelerators: Dict[str, int],
                                    use_spot: bool,
                                    region: Optional[str] = None,
                                    zone: Optional[str] = None) -> float:
        """Accelerator cost is already in the instance price."""
        del accelerators, use_spot, region, zone
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        del num_gigabytes  # Lyceum publishes no egress meter.
        return 0.0

    def __repr__(self) -> str:
        return self._REPR

    def make_deploy_resources_variables(
            self, resources: 'resources_lib.Resources', cluster_name: Any,
            region: 'clouds.Region', zones: Optional[List['clouds.Zone']],
            num_nodes: int, dryrun: bool = False,
            volume_mounts: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Variables fed to `templates/lyceum-ray.yml.j2`.

        Must include the hardware profile, gpu count, spot flag and the SSH
        user (`lyceum`, not root -- C1).

        Returns Dict[str, Any], NOT Dict[str, Optional[str]]: the template needs
        `gpu_count` as an int and `use_spot` as a bool. Stringifying to satisfy a
        narrower annotation would make `use_spot` the string 'False', which is
        truthy in Jinja -- silently inverting the spot flag.
        """
        del cluster_name, zones, num_nodes, dryrun, volume_mounts

        instance_type = resources.instance_type
        if instance_type is None:
            # Same shape as Shadeform: the caller may hand us an unresolved
            # request, in which case the catalog picks the instance type.
            feasible = self._get_feasible_launchable_resources(
                resources.copy(accelerators=None))
            instance_type = feasible.resources_list[0].instance_type
        assert instance_type is not None, resources

        # The profile/count split is the catalog's business: 'h100.8x' has to
        # become ('h100', 8) in exactly one place, and it goes verbatim into the
        # create payload's `instance_specs`.
        hardware_profile, gpu_count = catalog.parse_instance_type(instance_type)

        accelerators = resources.accelerators
        custom_resources = None
        if accelerators is not None:
            custom_resources = json.dumps(accelerators, separators=(',', ':'))

        return {
            'instance_type': instance_type,
            'hardware_profile': hardware_profile,
            'gpu_count': gpu_count,
            'use_spot': bool(resources.use_spot),
            'ssh_user': _SSH_USER,
            'region': region.name,
            'custom_resources': custom_resources,
        }

    def get_credential_file_mounts(self) -> Dict[str, str]:
        # Deliberately a pure constant: this runs on every launch for every
        # enabled cloud and feeds the file-mounts hash SkyPilot uses to dedupe
        # controller uploads and to decide whether setup can be skipped. Making
        # it depend on a stat() flips that hash the moment the key is rotated.
        # `sky/check.py:566` already drops entries whose local path is absent.
        return {
            f'{_CREDENTIAL_DIR}/{f}': f'{_CREDENTIAL_DIR}/{f}'
            for f in _CREDENTIAL_FILES
        }

    # NOTE: deliberately NO `check_credentials` override. The base class
    # declares `check_credentials(cls, cloud_capability)` and `sky/check.py:206`
    # calls it positionally for every registered cloud. A zero-arg override
    # TypeErrors there; `check.py:209`'s broad `except Exception` contains the
    # damage to Lyceum alone, but the result is that Lyceum can never pass
    # `sky check` and reports a raw traceback as its reason. The base class
    # already delegates to `_check_compute_credentials`, so implement only that.

    @classmethod
    def _check_compute_credentials(cls) -> Tuple[bool, Optional[str]]:
        """Verify the API key by calling GET /user/status.

        Must never raise, and must never echo the API key into the returned
        message.
        """
        try:
            lyceum_api.LyceumClient().get_user_status()
        except lyceum_api.LyceumAuthError:
            # Covers both "no key anywhere" and "the vendor rejected this key":
            # the exception text is dropped rather than redacted, because it is
            # the one message an operator can act on and it must not depend on
            # what the API echoed back.
            return False, _CREDENTIAL_HELP
        except lyceum_api.LyceumError as e:
            return False, (f'Lyceum credential check failed: '
                           f'{_redact(str(e)) or type(e).__name__}')
        except Exception as e:  # pylint: disable=broad-except
            # `sky check` turns anything escaping here into a formatted
            # traceback as Lyceum's reason, which buries the one line the
            # operator needed. Degrade to a message instead.
            return False, (f'Lyceum credential check failed unexpectedly '
                           f'({type(e).__name__}): {_redact(str(e))}')
        # A non-None message is printed as a warning even when the boolean is
        # True, so a healthy key must say nothing at all.
        return True, None

    @classmethod
    def get_user_identities(cls) -> Optional[List[List[str]]]:
        # Lyceum has one account per API key and no identity endpoint beyond
        # /user/status; there is nothing to disambiguate.
        return None

    def _get_feasible_launchable_resources(
            self, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        if resources.instance_type is not None:
            # Already launchable; the optimizer just wants it echoed back.
            assert resources.is_launchable(), resources
            return resources_utils.FeasibleResources([resources],
                                                     [resources.instance_type],
                                                     None)

        def _make(instance_types: List[str]) -> List['resources_lib.Resources']:
            return [
                resources.copy(
                    cloud=Lyceum(),
                    instance_type=instance_type,
                    accelerators=resources.accelerators,
                    cpus=None,
                    memory=None,
                ) for instance_type in instance_types
            ]

        accelerators = resources.accelerators
        if accelerators is not None:
            # One accelerator request per Resources in practice; take the first.
            acc_name, acc_count = list(accelerators.items())[0]
            instance_types, fuzzy = catalog.get_instance_type_for_accelerator(
                acc_name,
                acc_count,
                cpus=resources.cpus,
                memory=resources.memory,
                use_spot=resources.use_spot,
                region=resources.region,
                zone=resources.zone,
                max_hourly_cost=resources.max_hourly_cost)
            if not instance_types:
                # An accelerator Lyceum does not have must come back empty, not
                # raise: with `infra: "*"` the optimizer asks every enabled
                # cloud about every accelerator, and an exception here takes
                # down the plan for the other clouds too.
                return resources_utils.FeasibleResources(
                    [], list(fuzzy or []),
                    f'Lyceum does not offer {acc_name}:{acc_count}.')
            return resources_utils.FeasibleResources(_make(instance_types),
                                                     list(fuzzy or []), None)

        default_instance_type = self.get_default_instance_type(
            cpus=resources.cpus,
            memory=resources.memory,
            disk_tier=resources.disk_tier,
            region=resources.region,
            zone=resources.zone,
            use_spot=resources.use_spot,
            max_hourly_cost=resources.max_hourly_cost)
        if default_instance_type is None:
            return resources_utils.FeasibleResources([], [], None)
        return resources_utils.FeasibleResources(_make([default_instance_type]),
                                                 [], None)

    @classmethod
    def query_status(cls, name: str, tag_filters: Dict[str, str],
                     region: Optional[str], zone: Optional[str],
                     **kwargs) -> List['status_lib.ClusterStatus']:
        # With STATUS_VERSION is SKYPILOT the real query lives in
        # `provision/instance.py::query_instances`. This method is still on the
        # Cloud interface and is called during validation, so it must return a
        # list rather than inherit the base-class NotImplementedError.
        del name, tag_filters, region, zone, kwargs
        return []
