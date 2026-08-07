"""Lyceum provisioner: the `sky.provision` contract.

Signatures MUST match `sky.provision.<fn>` minus the leading `provider_name`.
`sky.provision._route_to_cloud_impl` does `inspect.signature(func).bind(...)`
at dispatch, so a mismatch is a TypeError in the middle of a launch rather
than an import error.

`query_instances` takes `retry_if_missing` here for that reason, even though
some in-tree providers omit it. `backend_utils._query_cluster_status_via_cloud_api`
forwards that argument BY KEYWORD, so a provider whose `query_instances` lacks
the parameter raises TypeError on every status refresh for its clusters --
which is worth knowing about because the same failure exists in-tree today
(SkyPilot 0.13.0's Shadeform provider omits it) and the only fix from outside
SkyPilot is yet another monkeypatch. Matching the dispatcher exactly means this
package never needs one; `tests/test_signature_conformance.py` enforces it for
all nine dispatched functions.

C-numbers below refer to the Lyceum API quirks tabulated in README.md.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from sky import exceptions as sky_exceptions
from sky import sky_logging
from sky.provision import common
# Imported at runtime, not under TYPE_CHECKING: `query_instances` returns real
# ClusterStatus members and `wait_instances` compares against one.
from sky.utils import status_lib

# Same -- the exception types are caught, not merely annotated.
from skypilot_lyceum import api, intent

logger = sky_logging.init_logger(__name__)

#: How long to wait for a created VM to become usable. Real measurements:
#: 221 s on-demand, 130 s spot; vendor docs claim 1-3 min. Ten minutes is the
#: floor, not the typical case.
PROVISION_TIMEOUT_S = 900

#: Seconds between polls in `wait_instances`. 900 / 5 = 180 requests worst
#: case, which is nothing against an API that answers a list in ~200 ms, and
#: keeps the "ready but null IP" window (C10) short.
POLL_INTERVAL_S = 5

#: Provider name as registered with CLOUD_REGISTRY and register_provisioner.
PROVIDER_NAME = 'lyceum'

#: SSH user. The vendor docs say `root`; root and ubuntu are both refused and
#: the real user is `lyceum` (C1).
SSH_USER = 'lyceum'

#: Statuses that are not in `api.TERMINAL_STATUSES` today but still mean "this
#: VM is on its way out". Treated as terminal locally so a rename or an added
#: state on the vendor side cannot make us adopt a dying VM.
_GONE_STATUSES = frozenset({'terminating', 'deleting', 'deleted', 'stopped',
                            'stopping'})

#: `<profile>.<count>x`, e.g. `l40s.1x` -- see `catalog.instance_type_name`.
_INSTANCE_TYPE_RE = re.compile(r'^([A-Za-z0-9_-]+)\.(\d+)x$')


def _client() -> 'api.LyceumClient':
    """The single seam every function here goes through to reach the API.

    Declared explicitly so tests have ONE thing to patch. Without it the seam is
    whatever the test file guesses, which means the tests end up defining the
    module's structure rather than checking it.

    Deliberately unmemoised: the API server is long-lived and a cached client
    would outlive a credential rotation.
    """
    return api.LyceumClient()


# ---------------------------------------------------------------------------
# Identity
#
# Lyceum has no tags and no server-side filtering, so `display_name ==
# cluster_name_on_cloud` is the ONLY handle. Every predicate below is applied
# here even though `LyceumClient.find_vms_by_display_name` promises the same
# filtering: this module is where terminating someone else's GPU or adopting a
# corpse actually costs money (C4), the checks are two comparisons, and a
# client that ever loosens its contract must not be able to turn that into a
# destroyed job.
# ---------------------------------------------------------------------------


def _is_terminal(vm: 'api.VM') -> bool:
    """True if `vm` is dead, dying, or gone -- never a live cluster member."""
    status = (vm.status or '').lower()
    if status in api.TERMINAL_STATUSES or status in _GONE_STATUSES:
        return True
    return bool(vm.is_terminal)


def _is_usable(vm: 'api.VM') -> bool:
    """True if `vm` is ready AND reachable (C10).

    `status == 'ready'` alone is not enough: an h200 reported ready at 104 s
    with `ip_address` still null, and returning then hands SkyPilot a null host.
    """
    return bool(vm.is_usable) and vm.ip is not None and not _is_terminal(vm)


def _belongs_to(vm: 'api.VM', cluster_name_on_cloud: str) -> bool:
    """Exact display_name equality -- never a prefix or substring match.

    SkyPilot builds `cluster_name_on_cloud` as `<name>-<hash>`, so `sky-cluster-
    abc` and `sky-cluster-abcd` are two ordinary, unrelated clusters. A prefix
    match adopts or terminates the wrong one.
    """
    return vm.display_name == cluster_name_on_cloud


def _cluster_vms(client: 'api.LyceumClient',
                 cluster_name_on_cloud: str,
                 *,
                 include_terminal: bool = False) -> List['api.VM']:
    """VMs of this cluster, newest first.

    `include_terminal` switches the read path deliberately:
    `find_vms_by_display_name` excludes terminal VMs *by contract*, so it can
    never answer "what does this cluster look like including the dead ones"
    (C4). That question goes through `list_vms(include_terminated=True)`.
    """
    if include_terminal:
        vms = client.list_vms(include_terminated=True)
    else:
        vms = client.find_vms_by_display_name(cluster_name_on_cloud)
    matched = [vm for vm in vms if _belongs_to(vm, cluster_name_on_cloud)]
    if not include_terminal:
        matched = [vm for vm in matched if not _is_terminal(vm)]
    # Names are reused across cluster generations; the newest wins.
    matched.sort(key=lambda vm: vm.created_at or '', reverse=True)
    return matched


def _provisioner_error(message: str) -> common.ProvisionerError:
    """`ProvisionerError` with the `errors` attribute its consumers expect.

    `cloud_vm_ray_backend._gcp_handler` reads `err.errors`; the class only
    annotates it, so an instance without it raises AttributeError one layer
    away from anything Lyceum-specific.
    """
    error = common.ProvisionerError(message)
    error.errors = []
    return error


def _parse_instance_type(instance_type: str) -> Tuple[str, int]:
    """`'l40s.1x'` -> `('l40s', 1)`. Mirrors `catalog.instance_type_name`."""
    match = _INSTANCE_TYPE_RE.match(str(instance_type or ''))
    if match is None:
        raise _provisioner_error(
            f'Unparseable Lyceum InstanceType {instance_type!r}; expected '
            '<hardware_profile>.<gpu_count>x, e.g. l40s.1x.')
    return match.group(1).lower(), int(match.group(2))


def _node_spec(config: common.ProvisionConfig) -> Tuple[str, int, bool, str]:
    """(hardware_profile, gpu_count, use_spot, public_key) from the config.

    `HardwareProfile`/`GpuCount` are what `templates/lyceum-ray.yml.j2` writes,
    but `InstanceType` is the field every in-tree provider is guaranteed to
    have, so it is the fallback rather than the other way round.
    """
    node_config = config.node_config or {}
    profile = node_config.get('HardwareProfile')
    gpu_count = node_config.get('GpuCount')
    if not profile or not gpu_count:
        parsed_profile, parsed_count = _parse_instance_type(
            node_config.get('InstanceType'))
        profile = profile or parsed_profile
        gpu_count = gpu_count or parsed_count

    provider_config = config.provider_config or {}
    use_spot = bool(node_config.get('UseSpot',
                                    provider_config.get('use_spot', False)))

    auth_config = config.authentication_config or {}
    public_key = (node_config.get('PublicKey') or
                  auth_config.get('ssh_public_key'))
    if not public_key:
        raise _provisioner_error(
            'No SSH public key in the provision config; Lyceum takes the key '
            'inline on create and has no key registry to fall back on.')
    return str(profile), int(gpu_count), use_spot, str(public_key).strip()


def _record(region: str, cluster_name_on_cloud: str, head_instance_id: str,
            created_instance_ids: List[str]) -> common.ProvisionRecord:
    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        region=region,
        # Lyceum has no zones.
        zone=None,
        cluster_name=cluster_name_on_cloud,
        head_instance_id=head_instance_id,
        # Lyceum has no stop, so nothing is ever resumed.
        resumed_instance_ids=[],
        created_instance_ids=created_instance_ids,
    )


def _stop_the_meter(client: 'api.LyceumClient', cluster_name_on_cloud: str,
                    created_instance_id: Optional[str]) -> None:
    """Terminate anything this create may have left running. Never raises.

    Called only from the failure path of `run_instances`. Two handles, because
    a create that times out after committing server-side leaves us with the
    display_name and nothing else.
    """
    vm_ids: List[str] = []
    if created_instance_id:
        vm_ids.append(created_instance_id)
    try:
        for vm in _cluster_vms(client, cluster_name_on_cloud):
            if vm.vm_id not in vm_ids:
                vm_ids.append(vm.vm_id)
    except Exception as exc:  # pylint: disable=broad-except
        # The API is already unhealthy -- that is why we are here.
        logger.warning(f'Could not list VMs of {cluster_name_on_cloud!r} while '
                       f'cleaning up a failed create: {exc}')
    for vm_id in vm_ids:
        try:
            logger.warning(f'Terminating {vm_id} left behind by a failed '
                           f'create for {cluster_name_on_cloud!r}.')
            client.terminate_vm(vm_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f'FAILED to terminate {vm_id} after a failed create '
                         f'for {cluster_name_on_cloud!r}: {exc}. This VM is '
                         'billing with nothing referencing it; the orphan '
                         'reaper is now the only thing that will stop it.')


# ---------------------------------------------------------------------------
# The nine dispatched functions
# ---------------------------------------------------------------------------


def run_instances(region: str, cluster_name: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    """Create (or adopt) the VM backing `cluster_name_on_cloud`.

    Idempotent: an existing live VM with this display_name is reused rather
    than duplicated, because SkyPilot retries this call on failover.

    Adoption gates on "not terminal", NOT on `VM.is_usable`. Provisioning takes
    130-221 s, and a failover retry landing inside that window would see a
    still-provisioning VM as "nothing here", create a second GPU, and leave the
    first one billing with nothing referencing it (C5: no cloud-side TTL).
    `is_usable` is the gate for `wait_instances` (C10); conflating the two is
    what costs the money.

    Raises `sky.exceptions.ResourcesUnavailableError` when the API reports
    capacity exhaustion (C7), so the optimizer fails over to another cloud
    instead of retrying an exhausted SKU. Every other failure keeps its own
    type -- in particular a genuine 5xx stays retryable and must NOT be
    reported as unavailable capacity.
    """
    del cluster_name  # display_name is cluster_name_on_cloud.
    client = _client()

    if config.count > 1:
        # Declared unsupported in the feature envelope; failing here beats
        # silently provisioning a one-node cluster the caller then waits
        # forever for.
        raise _provisioner_error(
            f'Lyceum does not support multi-node clusters; {config.count} '
            'nodes were requested.')

    existing = _cluster_vms(client, cluster_name_on_cloud)
    if existing:
        head = existing[0]
        logger.info(f'Adopting existing Lyceum VM {head.vm_id} '
                    f'(status={head.status}) for {cluster_name_on_cloud!r}.')
        return _record(region, cluster_name_on_cloud, head.vm_id, [])

    profile, gpu_count, use_spot, public_key = _node_spec(config)

    # The receipt, written BEFORE the create. If this process dies between here
    # and Lyceum acknowledging the VM, the orphan reaper can still collect it --
    # which is the entire leak this ledger exists to close. Recording afterwards
    # would be missing for exactly the VM worth finding. Best-effort inside:
    # a failed ledger write must not fail a launch.
    intent.record(cluster_name_on_cloud)

    created_instance_id: Optional[str] = None
    try:
        vm = client.create_vm(public_key=public_key,
                              hardware_profile=profile,
                              gpu_count=gpu_count,
                              display_name=cluster_name_on_cloud,
                              use_spot=use_spot)
        created_instance_id = vm.vm_id
        return _record(region, cluster_name_on_cloud, vm.vm_id, [vm.vm_id])
    except api.LyceumCapacityError as exc:
        # C7: a 500 whose detail says "could not be provisioned". The request
        # was rejected before anything was created (measured: 2.7-9.1 s, no VM,
        # no charge), so there is nothing to clean up -- and sweeping the
        # display_name here could only ever hit a VM someone else just made.
        raise sky_exceptions.ResourcesUnavailableError(
            f'Lyceum has no capacity for {profile} x{gpu_count} '
            f'({"spot" if use_spot else "on-demand"}): {exc}') from exc
    except BaseException:
        # Anything else -- transport timeout, reset, 5xx -- leaves the outcome
        # of the POST unknown. The create may have committed server-side with
        # the response lost on the way back, in which case display_name is the
        # only handle that exists. Clean up before propagating: SkyPilot's own
        # teardown is a backstop, but `provisioner.py:172-188` skips it for
        # four exception types and even the happy path gives up after three
        # retries. The exception is re-raised unchanged, so a retryable server
        # error stays retryable.
        _stop_the_meter(client, cluster_name_on_cloud, created_instance_id)
        raise


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional['status_lib.ClusterStatus']) -> None:
    """Block until the cluster's VM reaches `state`.

    Must gate on `VM.is_usable` -- ready status AND a non-null IP. A VM can
    report `ready` while `ip_address` is still null (C10); returning then hands
    SkyPilot a null host and fails the launch.

    On timeout or a terminal status, raise `sky.provision.common.ProvisionerError`.
    That type is chosen deliberately: `sky/provision/provisioner.py:105` catches
    `RuntimeError` from this call to drive its retry, and ProvisionerError is a
    RuntimeError subclass.
    """
    del region
    if state is not None and state != status_lib.ClusterStatus.UP:
        # UP is the only state Lyceum can reach: there is no stop/start, so a
        # VM is either coming up, up, or gone.
        return

    client = _client()
    deadline = time.time() + PROVISION_TIMEOUT_S
    last_transient: Optional[Exception] = None
    while True:
        # include_terminal so a VM that died mid-provision is *seen* rather
        # than looking like "not listed yet" and burning the whole budget.
        #
        # A retryable server error here must NOT abort the provision. Learned in
        # production: a launch died on `LyceumServerError: GET /vms/list failed
        # with HTTP 500` while the VM was healthy and had already been assigned
        # an IP. Separating LyceumServerError from LyceumCapacityError (C7) is
        # only worth anything if the retryable one is actually retried, and the
        # Lyceum API is demonstrably flaky -- /vms/availability and /vms/list
        # both returned 500 within ten minutes during phase-3 smoke testing.
        #
        # Auth errors are deliberately NOT caught: a bad credential cannot fix
        # itself, and spinning on it holds a GPU on the meter for the full
        # timeout with no cloud-side TTL to stop it (C5).
        try:
            vms = _cluster_vms(client, cluster_name_on_cloud,
                               include_terminal=True)
            last_transient = None
        except api.LyceumServerError as e:
            last_transient = e
            now = time.time()
            if now >= deadline:
                raise _provisioner_error(
                    f'Timed out after {PROVISION_TIMEOUT_S}s waiting for Lyceum '
                    f'cluster {cluster_name_on_cloud!r}: the API kept failing '
                    f'({e}).') from e
            logger.debug('Lyceum read failed while polling %s (%s); retrying.',
                         cluster_name_on_cloud, e)
            time.sleep(min(POLL_INTERVAL_S, deadline - now))
            continue
        live = [vm for vm in vms if not _is_terminal(vm)]

        if live and all(_is_usable(vm) for vm in live):
            return
        if vms and not live:
            statuses = ', '.join(f'{vm.vm_id}={vm.status}' for vm in vms)
            raise _provisioner_error(
                f'Lyceum cluster {cluster_name_on_cloud!r} reached a terminal '
                f'status while waiting for it to come up ({statuses}).')

        now = time.time()
        if now >= deadline:
            observed = ', '.join(
                f'{vm.vm_id}={vm.status} ip={vm.ip}' for vm in live) or 'none'
            raise _provisioner_error(
                f'Timed out after {PROVISION_TIMEOUT_S}s waiting for Lyceum '
                f'cluster {cluster_name_on_cloud!r} to become usable '
                f'(ready with a non-null IP). Observed: {observed}.')
        time.sleep(min(POLL_INTERVAL_S, deadline - now))


def stop_instances(cluster_name_on_cloud: str,
                   provider_config: Optional[Dict[str, Any]] = None,
                   worker_only: bool = False) -> None:
    """Unsupported: Lyceum has no stop/start, only terminate.

    Must raise, not silently no-op -- a silent no-op would leave a VM billing
    while SkyPilot believed it was stopped. NotImplementedError specifically:
    `provisioner.py` catches that exact type around teardown_cluster and turns
    it into an actionable StopFailoverError. Terminating instead would be worse
    still: silent data loss on a `sky stop`.
    """
    del provider_config, worker_only
    raise NotImplementedError(
        f'Lyceum does not support stopping instances: the API offers only '
        f'terminate, so {cluster_name_on_cloud!r} cannot be stopped and later '
        'restarted. Use `sky down` to terminate it instead.')


def terminate_instances(cluster_name_on_cloud: str,
                        provider_config: Optional[Dict[str, Any]] = None,
                        worker_only: bool = False) -> None:
    """Terminate every live VM for this cluster. Idempotent.

    An explicit DELETE is the only thing that ever stops a Lyceum VM billing
    (C5: no cloud-side TTL), so this must terminate everything it can reach and
    only give up on the failures that are genuinely retryable.
    """
    del provider_config
    if worker_only:
        # Single-node only: there are no workers to remove, and terminating
        # the head here would destroy the cluster.
        return

    client = _client()
    first_error: Optional[BaseException] = None
    for vm in _cluster_vms(client, cluster_name_on_cloud):
        try:
            client.terminate_vm(vm.vm_id)
        except api.LyceumNotFoundError:
            # Already gone: `sky down` races the reaper and autodown by design.
            logger.debug(f'VM {vm.vm_id} of {cluster_name_on_cloud!r} was '
                         'already gone.')
        except Exception as exc:  # pylint: disable=broad-except
            # Keep going: a VM left behind because an earlier one failed is a
            # GPU on the meter. Report the first failure once the rest are down.
            logger.error(f'Failed to terminate {vm.vm_id} of '
                         f'{cluster_name_on_cloud!r}: {exc}')
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def get_cluster_info(region: str, cluster_name_on_cloud: str,
                     provider_config: Optional[Dict[str, Any]] = None
                     ) -> common.ClusterInfo:
    """Return the cluster's connection metadata.

    `head_instance_id` must be a key of `instances`; `ssh_user` must be
    `SSH_USER` (C1); each `InstanceInfo.ssh_port` comes from parsing
    `ip_address` (C2, done once in `api.parse_ip_address`).

    A cluster with no live VM is an empty ClusterInfo, not an error: `sky
    status` calls this on torn-down clusters.
    """
    del region
    client = _client()
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id: Optional[str] = None
    for vm in _cluster_vms(client, cluster_name_on_cloud):
        if head_instance_id is None:
            head_instance_id = vm.vm_id
        instances.setdefault(vm.vm_id, []).append(
            common.InstanceInfo(
                instance_id=vm.vm_id,
                internal_ip=vm.ip or '',
                external_ip=vm.ip,
                tags={},
                ssh_port=vm.ssh_port,
                node_name=vm.raw.get('name') if vm.raw else None,
            ))
    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name=PROVIDER_NAME,
        provider_config=provider_config,
        ssh_user=SSH_USER,
    )


def query_instances(cluster_name: str, cluster_name_on_cloud: str,
                    provider_config: Optional[Dict[str, Any]] = None,
                    non_terminated_only: bool = True,
                    retry_if_missing: bool = False
                    ) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                                         Optional[str]]]:
    """Map vm_id -> (SkyPilot status, reason).

    A None status means terminated/terminating. Terminated VMs are excluded
    when `non_terminated_only`, which matters more here than on other clouds
    because they linger in `/vms/list` indefinitely (C4).

    A terminal VM maps to None and never to STOPPED: SkyPilot treats STOPPED as
    restartable and Lyceum has no start.
    """
    del cluster_name, provider_config
    client = _client()

    # `non_terminated_only=False` means "report the dead ones too", which
    # `find_vms_by_display_name` cannot do by contract -- hence the list path.
    attempts = 3 if retry_if_missing else 1
    result: Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                            Optional[str]]] = {}
    for attempt in range(attempts):
        result = {}
        for vm in _cluster_vms(client,
                               cluster_name_on_cloud,
                               include_terminal=True):
            if _is_terminal(vm):
                if non_terminated_only:
                    continue
                result[vm.vm_id] = (None, f'VM is {vm.status}')
            elif vm.status in api.READY_STATUSES:
                result[vm.vm_id] = (status_lib.ClusterStatus.UP, None)
            else:
                # pending / provisioning / anything new the vendor adds.
                result[vm.vm_id] = (status_lib.ClusterStatus.INIT, None)
        if result or attempt == attempts - 1:
            break
        # `retry_if_missing`: the caller has a cluster it believes exists and
        # the listing does not show it yet.
        time.sleep(POLL_INTERVAL_S)
    return result


def open_ports(cluster_name_on_cloud: str, ports: List[str],
               provider_config: Optional[Dict[str, Any]] = None) -> None:
    """No-op: Lyceum exposes no firewall API.

    `OPEN_PORTS_VERSION = LAUNCH_ONLY` means the backend calls this on every
    launch, so raising (as the in-tree Shadeform provider does) would fail an
    otherwise-good launch.
    """
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(cluster_name_on_cloud: str, ports: List[str],
                  provider_config: Optional[Dict[str, Any]] = None) -> None:
    """No-op counterpart to `open_ports`."""
    del cluster_name_on_cloud, ports, provider_config
