"""Tests for `skypilot_lyceum.cloud.Lyceum` -- the SkyPilot Cloud class.

Adapted from skypilot 0.13.0's own unit tests:
  * `tests/unit_tests/test_hyperbolic.py::test_hyperbolic_unsupported_features`
    -- the one genuinely generic cloud test upstream has. Copied and then
    tightened: upstream only checks that *declared* reasons are strings, which
    a completely empty envelope passes. Ours additionally pins the exact set
    (see `Lyceum._CLOUD_UNSUPPORTED_FEATURES`) and cross-checks it against
    Shadeform so that a `CloudImplementationFeatures` member added by a future
    SkyPilot bump surfaces as a failure instead of silently defaulting to
    "supported".
  * `tests/unit_tests/test_verda.py` -- cloud-basics shape (repr, name-length
    limit, credential file mounts, check-credentials-missing).
  * `tests/unit_tests/test_resources.py::test_aws_make_deploy_variables` -- the
    assert-the-ENTIRE-dict pattern. That is the highest-value test here: it is
    the only thing pinning the contract between this class and
    `skypilot_lyceum/templates/lyceum-ray.yml.j2`.

LOCAL FIXTURES: `lyceum`, `lyceum_catalog_module` and `offline_api` are defined
in this file rather than in `tests/conftest.py`. If they turn out to be useful
to other test modules they should be lifted verbatim.
"""
from __future__ import annotations

import inspect
import pathlib
import sys
from typing import Any, Dict, List
from unittest import mock

import pytest

from sky import clouds
from sky.clouds import shadeform as shadeform_cloud
from sky.resources import Resources
from sky.utils import registry
from sky.utils import resources_utils

from skypilot_lyceum import api as lyceum_api
from skypilot_lyceum import catalog as lyceum_catalog
from skypilot_lyceum import cloud as lyceum_cloud_mod
from skypilot_lyceum.cloud import Lyceum

_TEMPLATE_PATH = (pathlib.Path(lyceum_cloud_mod.__file__).parent / 'templates' /
                  'lyceum-ray.yml.j2')


def _template_variables() -> set:
    """Variables the shipped Jinja template actually references.

    Derived by parsing the template rather than hardcoded, so editing the
    template automatically re-scopes every assertion built on this. Parsing
    (not substring matching) also means the explanatory `{# ... #}` comment
    block at the top of the template cannot be mistaken for a reference.
    """
    jinja2 = pytest.importorskip('jinja2')
    from jinja2 import meta as jinja_meta  # pylint: disable=import-outside-toplevel
    return set(
        jinja_meta.find_undeclared_variables(jinja2.Environment().parse(
            _TEMPLATE_PATH.read_text())))

#: Variables the Jinja template gets from SkyPilot's backend
#: (`backends/backend_utils.py::write_cluster_config`), NOT from
#: `make_deploy_resources_variables`. Everything else the template references is
#: this class's responsibility.
_BACKEND_SUPPLIED_TEMPLATE_VARS = frozenset({
    # Verified at sky/backends/backend_utils.py:1218 -- the backend injects the
    # pip command that targets SkyPilot's remote python env. If the template
    # ever needs it, it comes from the backend and must not be supplied here.
    'sky_pip_cmd',
    'cluster_name_on_cloud',
    'num_nodes',
    'credentials',
    'ssh_private_key',
    'sky_ray_yaml_remote_path',
    'sky_ray_yaml_local_path',
    'sky_remote_path',
    'sky_local_path',
    'sky_wheel_hash',
    'initial_setup_commands',
    'conda_installation_commands',
    'uv_installation_commands',
    'ray_skypilot_installation_commands',
    'copy_skypilot_templates_commands',
    'ssh_max_sessions_config',
})

#: Supplied by `provision.template_override` via `TemplateSpec.variables`, not
#: by `make_deploy_resources_variables`. They describe the SERVER's filesystem
#: (where the node-autodown wheel lives), which is not a function of the
#: Resources being provisioned -- so they must not leak into that method.
_PROVISIONER_SUPPLIED_TEMPLATE_VARS = frozenset({
    'lyceum_file_mounts',
    'lyceum_node_setup_command',
})


# --------------------------------------------------------------------------
# Local fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def lyceum():
    """A `Lyceum()` instance, with the class guaranteed registered.

    Importing `skypilot_lyceum.cloud` is what runs the
    `@registry.CLOUD_REGISTRY.register` decorator; we do not call
    `skypilot_lyceum.enable()` because that also applies the anchored patches,
    which are a separate unit under test (`tests/test_registration.py`).
    """
    assert 'lyceum' in registry.CLOUD_REGISTRY, (
        'importing skypilot_lyceum.cloud must register the cloud')
    return Lyceum()


@pytest.fixture(autouse=True)
def lyceum_catalog_module(monkeypatch):
    """Make `sky.catalog`'s hardcoded importlib dispatch resolve to our module.

    `sky/catalog/__init__.py::_map_clouds_catalog` does
    `importlib.import_module(f'sky.catalog.{cloud}_catalog')` with no registry,
    so without this a delegating `Lyceum` method blows up on ModuleNotFoundError
    instead of on the thing under test. Production does the same injection in
    `skypilot_lyceum.patches.patch_catalog_module`; we do it directly so these
    tests do not depend on that patch being implemented yet.

    Injecting the *same module object* also means a `monkeypatch.setattr` on
    `skypilot_lyceum.catalog.<fn>` is observed no matter which of the two
    plausible delegation routes `cloud.py` takes.
    """
    monkeypatch.setitem(sys.modules, 'sky.catalog.lyceum_catalog',
                        lyceum_catalog)
    yield lyceum_catalog


@pytest.fixture
def offline_api(monkeypatch):
    """Force every catalog build down the baked-CSV fallback path.

    `skypilot_lyceum.api` is the sole owner of HTTP, and `LyceumClient._request`
    is its single egress point, so cutting it is enough to simulate "the Lyceum
    API is unreachable" without knowing anything about the catalog's internals.
    Tests that assert real catalog values use this so they exercise
    `skypilot_lyceum/data/vms.csv` deterministically rather than the network.
    """

    def _offline(*args, **kwargs):
        raise lyceum_api.LyceumServerError('offline: unit test')

    monkeypatch.setattr(lyceum_api.LyceumClient, '_request', _offline)


def _call_check_credentials(cls=Lyceum):
    """Invoke `check_credentials` the way `sky/check.py:206` does, if possible.

    Tolerates the current zero-arg stub signature so that the behavioural tests
    below fail on behaviour, not on a TypeError. The signature itself is pinned
    separately by `test_check_credentials_signature_matches_skypilot_caller`.
    """
    if inspect.signature(cls.check_credentials).parameters:
        return cls.check_credentials(clouds.CloudCapability.COMPUTE)
    return cls.check_credentials()


def _deploy_vars(cloud: Lyceum, resources: Resources,
                 region_name: str = 'lyceum') -> Dict[str, Any]:
    return cloud.make_deploy_resources_variables(
        resources,
        resources_utils.ClusterName(display_name='sky-lyceum-test',
                                    name_on_cloud='sky-lyceum-test-abcd'),
        clouds.Region(region_name),
        None,
        num_nodes=1,
        dryrun=True,
    )


# --------------------------------------------------------------------------
# Cloud basics
# --------------------------------------------------------------------------
def test_lyceum_repr_is_lyceum(lyceum):
    """Prevents the display name drifting from the docs and `sky status` output.

    `_REPR` is what every user-facing message interpolates
    (`cloud.py` uses it in ~8 error strings), and `Cloud.__repr__` must return
    it verbatim rather than the class name or 'Cloud'.
    """
    assert Lyceum._REPR == 'Lyceum'
    assert repr(lyceum) == 'Lyceum'


def test_lyceum_registered_under_the_name_lyceum(lyceum):
    """Prevents `infra: lyceum` resolving to nothing, or to a second instance.

    The registry key is derived from `cls.__name__.lower()`
    (`sky/utils/registry.py::_Registry.register`), and it is the string used in
    a task's `infra:` field, in `patches.patch_all_clouds`, and in every
    `clouds='lyceum'` catalog dispatch. Renaming the class silently breaks all
    three.
    """
    assert Lyceum.canonical_name() == 'lyceum'
    # Several base-class methods we do NOT override -- validate_region_zone,
    # is_image_tag_valid, get_image_size -- dispatch to the catalog with
    # `clouds=cls._REPR.lower()`, so _REPR and the registry key must agree or
    # those silently look for a `sky.catalog.<something-else>_catalog` module.
    assert Lyceum._REPR.lower() == Lyceum.canonical_name()
    assert isinstance(registry.CLOUD_REGISTRY['lyceum'], Lyceum)
    assert isinstance(registry.CLOUD_REGISTRY.from_str('lyceum'), Lyceum)
    # Co-equal with the in-tree clouds, not a replacement for any of them.
    assert 'shadeform' in registry.CLOUD_REGISTRY


def test_lyceum_max_cluster_name_length_is_the_public_hook(lyceum):
    """The limit must live on the PUBLIC name, or it is dead code.

    SkyPilot consults exactly one name: `sky/backends/backend_utils.py:761` does
    `make_cluster_name_on_cloud(cluster_name, max_length=cloud.
    max_cluster_name_length())`, and the base class (`sky/clouds/cloud.py:163`)
    defines only that public method, returning None. There is no
    `_max_cluster_name_length` attribute on the base class at all -- verified
    below, because that is the whole trap.

    Shadeform (`sky/clouds/shadeform.py:88`) defines the PRIVATE
    `_max_cluster_name_length`, which nothing upstream calls, so
    `Shadeform.max_cluster_name_length()` returns None and its 120-char limit is
    dead code. Nine other in-tree clouds copy the same mistake. Copying the
    idiom here would silently mean "unlimited".

    That matters more for Lyceum than for anyone else, because
    `display_name` the ENTIRE identity scheme (Lyceum has no tags and no
    server-side filtering, so `display_name == cluster_name_on_cloud` is how a
    VM is found again). The name that reaches `display_name` is exactly the one
    this limit bounds; if it is unbounded, cluster lookup resolves the wrong VM
    or none at all.
    """
    assert Lyceum._MAX_CLUSTER_NAME_LEN_LIMIT == 120
    assert Lyceum.max_cluster_name_length() == 120

    # The base class has no private spelling -- so a `_max_cluster_name_length`
    # override overrides nothing and is never called.
    assert not hasattr(clouds.Cloud, '_max_cluster_name_length'), (
        'skypilot grew a private _max_cluster_name_length hook; re-check which '
        'name backend_utils actually calls before trusting this test')
    assert clouds.Cloud.max_cluster_name_length() is None, (
        'the base class default is "unlimited"; not overriding the public name '
        'is what this test exists to catch')
    # The concrete demonstration that the private name is inert upstream.
    assert shadeform_cloud.Shadeform.max_cluster_name_length() is None, (
        "Shadeform's private-only override no longer reads as dead code; if "
        'upstream fixed it, this cautionary comparison can go')


def test_max_cluster_name_length_actually_truncates_via_skypilot(lyceum):
    """The value must flow through SkyPilot's own name builder, not just exist.

    Reproduces `backend_utils.py:761` verbatim -- the single call site -- rather
    than asserting the return value in isolation. `make_cluster_name_on_cloud`
    is the function that produces `cluster_name_on_cloud`, which goes straight
    into `display_name`, so this is the end of the chain that matters.

    The Shadeform contrast is the point: with the same 200-char input, the
    public hook truncates to 120 while Shadeform's private-only override lets
    213 characters through untouched.
    """
    from sky.utils import common_utils  # pylint: disable=import-outside-toplevel

    long_name = 'sky-' + 'x' * 200

    ours = common_utils.make_cluster_name_on_cloud(
        long_name, max_length=Lyceum.max_cluster_name_length())
    assert len(ours) <= 120, (
        f'make_cluster_name_on_cloud produced {len(ours)} chars for a Lyceum '
        'cluster; the limit is not reaching SkyPilot')

    untruncated = common_utils.make_cluster_name_on_cloud(
        long_name,
        max_length=shadeform_cloud.Shadeform.max_cluster_name_length())
    assert len(untruncated) > 120, (
        'the Shadeform control no longer demonstrates the unbounded case')


def test_lyceum_uses_the_skypilot_provisioner_and_status_paths(lyceum):
    """Prevents provisioning silently routing down the deprecated Ray path.

    The base class defaults to `ProvisionerVersion.RAY_AUTOSCALER` /
    `StatusVersion.CLOUD_CLI`, both deprecated. Inheriting those would make
    SkyPilot look for a Ray node provider and a cloud CLI that do not exist for
    Lyceum -- a failure that shows up only at launch time, not at import.
    """
    assert Lyceum.PROVISIONER_VERSION is clouds.ProvisionerVersion.SKYPILOT
    assert Lyceum.STATUS_VERSION is clouds.StatusVersion.SKYPILOT
    assert Lyceum.PROVISIONER_VERSION is not clouds.Cloud.PROVISIONER_VERSION
    assert Lyceum.STATUS_VERSION is not clouds.Cloud.STATUS_VERSION
    # Ports can only be opened at launch: Lyceum exposes no firewall API.
    assert Lyceum.OPEN_PORTS_VERSION is clouds.OpenPortsVersion.LAUNCH_ONLY


# --------------------------------------------------------------------------
# Unsupported-feature envelope
# --------------------------------------------------------------------------
#: The exact declared envelope. Kept as names so a rename upstream fails
#: loudly at collection rather than quietly shrinking the set.
_EXPECTED_UNSUPPORTED = frozenset({
    'MULTI_NODE',
    'STOP',
    # Lyceum can only DELETE, so an idle timer can only ever mean autodown.
    # Without this, `sky launch -i N` (no --down) validates and then asks the
    # node to stop forever while the VM bills -- STOP does not cover it, because
    # that gate guards `sky autostop`, not the launch flag.
    'AUTOSTOP',
    'SPOT_INSTANCE',
    'CUSTOM_DISK_TIER',
    'DOCKER_IMAGE',
    'IMAGE_ID',
    'STORAGE_MOUNTING',
    'HOST_CONTROLLERS',
    'CLONE_DISK_FROM_CLUSTER',
    'CUSTOM_NETWORK_TIER',
    'CUSTOM_MULTI_NETWORK',
    'HIGH_AVAILABILITY_CONTROLLERS',
    'LOCAL_DISK',
})


# NOTE: upstream's `test_hyperbolic_unsupported_features` shape -- loop the enum,
# `isinstance(reason, str)` if declared, `assert feature not in dict` otherwise --
# is deliberately NOT reproduced here. Its else branch is a tautology and the
# whole test passes against an empty `_CLOUD_UNSUPPORTED_FEATURES = {}`. The
# test below subsumes it: it checks the same string-ness (via `.strip()`, which
# also rejects the empty string upstream's isinstance accepts) and refuses an
# empty envelope.
def test_lyceum_unsupported_feature_reasons_are_non_empty(lyceum):
    """Prevents an empty-string reason, which upstream's isinstance check passes.

    An empty reason produces "not supported by Lyceum: " with nothing after the
    colon and gives the operator nothing to act on. SkyPilot interpolates the
    value straight into the user-facing "The following features are not
    supported by Lyceum:" message (`sky/clouds/cloud.py:762`), so a non-string
    renders as garbage exactly when a user is already confused about why their
    launch was rejected -- hence the reason must be a string, and a useful one.
    """
    assert lyceum._CLOUD_UNSUPPORTED_FEATURES, (
        'the envelope must not be empty -- an empty dict claims Lyceum '
        'supports multi-node, stop, docker images and storage mounting')
    for feature, reason in lyceum._CLOUD_UNSUPPORTED_FEATURES.items():
        assert isinstance(feature, clouds.CloudImplementationFeatures)
        assert isinstance(reason, str), (
            f'{feature} declared with a {type(reason).__name__}, not a str; '
            'SkyPilot interpolates this straight into the rejection message')
        assert reason.strip(), f'{feature} declared with an empty reason'


def test_lyceum_declares_exactly_the_designed_envelope(lyceum):
    """Prevents the S7 envelope drifting silently in either direction.

    Under-declaring lets SkyPilot attempt something the API cannot do (e.g. a
    2-node cluster, or `sky stop`, which Lyceum has no endpoint for);
    over-declaring makes the optimizer skip Lyceum for jobs it could actually
    run.
    """
    declared = {f.name for f in lyceum._CLOUD_UNSUPPORTED_FEATURES}
    assert declared == set(_EXPECTED_UNSUPPORTED)


#: Features Lyceum refuses that Shadeform allows. Every entry needs a reason
#: rooted in the Lyceum API, not in convenience.
_LYCEUM_ONLY_UNSUPPORTED = frozenset({
    # Shadeform can stop an instance, so an idle timer there can mean "pause".
    # Lyceum has DELETE and nothing else, so the same flag would ask the node to
    # do something impossible -- silently, on a retry loop, while it bills.
    'AUTOSTOP',
})


def test_lyceum_envelope_is_at_least_as_strict_as_shadeform(lyceum):
    """Prevents Lyceum CLAIMING a capability Shadeform has and it does not.

    The two clouds are co-equal failover targets, so a plan feasible on one and
    rejected on the other only surfaces mid-failover. The asymmetry that hurts is
    Lyceum declaring FEWER restrictions than Shadeform: the optimizer would then
    route work to Lyceum that its API cannot perform. Declaring MORE is safe --
    the optimizer simply prefers Shadeform for those jobs.

    So this is a subset assertion, plus an explicit inventory of the difference:
    a new divergence has to be justified in `_LYCEUM_ONLY_UNSUPPORTED` rather
    than appearing by accident.
    """
    ours = {f.name for f in lyceum._CLOUD_UNSUPPORTED_FEATURES}
    theirs = {
        f.name for f in shadeform_cloud.Shadeform._CLOUD_UNSUPPORTED_FEATURES
    }
    assert theirs <= ours, (
        f'Lyceum claims {sorted(theirs - ours)} which Shadeform refuses -- '
        'the optimizer would route work to an API that cannot do it')
    assert ours - theirs == _LYCEUM_ONLY_UNSUPPORTED


def test_shadeform_still_triages_every_upstream_feature(lyceum):
    """Upstream-drift canary. Read the name literally: this is about Shadeform.

    A new `CloudImplementationFeatures` member defaults to "supported" for every
    cloud that does not declare it, so a SkyPilot version bump can hand us a
    capability nobody triaged. This fails on the bump -- at build time, in CI --
    instead of at 3am when SkyPilot tries to use it against an API that has no
    such thing.

    Honest about what it actually proves: the assertion is
    `all_features - ours - theirs == <the four known-undeclared>`, and Lyceum's
    envelope is a subset of Shadeform's (pinned by
    `test_lyceum_envelope_matches_shadeform`), so `ours` never removes a name
    `theirs` does not already remove. Today, with Lyceum's envelope still empty,
    it is a statement about upstream and Shadeform ONLY -- it cannot fail
    because of anything Lyceum declares or omits. That is still worth having:
    Shadeform is our reference triage, and the bump is what we want to catch.
    Lyceum's own coverage is `test_lyceum_declares_exactly_the_designed_envelope`.

    The three members deliberately left undeclared by both:
      OPEN_PORTS     -- handled via OPEN_PORTS_VERSION, not this dict.
      AUTODOWN/AUTO_TERMINATE -- SkyPilot-side skylet behaviour, not a cloud API
      capability. AUTODOWN is what we WANT to work, and node-side autodown is
      what makes it work (see `skypilot_lyceum/node_autodown.py`).

    AUTOSTOP used to be on that list, on the reasoning that an idle timer is
    skylet behaviour rather than a cloud capability. That was wrong in a way
    that cost money: the timer is skylet-side, but what it ASKS the cloud to do
    at the end is not, and Lyceum cannot stop anything. Lyceum now declares it.
    """
    ours = {f.name for f in lyceum._CLOUD_UNSUPPORTED_FEATURES}
    theirs = {
        f.name for f in shadeform_cloud.Shadeform._CLOUD_UNSUPPORTED_FEATURES
    }
    unclassified = {f.name for f in clouds.CloudImplementationFeatures
                   } - ours - theirs
    expected = {'OPEN_PORTS', 'AUTO_TERMINATE', 'AUTODOWN'}
    assert unclassified == expected, (
        'skypilot has a CloudImplementationFeatures member that neither Lyceum '
        f'nor Shadeform has triaged: {sorted(unclassified - expected)}')


def test_stop_is_unsupported_because_the_api_only_terminates(lyceum):
    """Prevents STOP being re-enabled by someone who assumes it is a stub.

    C5: Lyceum exposes `DELETE /vms/{id}` and nothing else. There is
    no stop, and no cloud-side TTL either -- which is why the reason string must
    say so rather than say "not implemented yet".
    """
    reason = lyceum._CLOUD_UNSUPPORTED_FEATURES[
        clouds.CloudImplementationFeatures.STOP]
    assert 'terminat' in reason.lower(), (
        'the STOP reason must record that the Lyceum API has terminate only, '
        f'not that we did not get round to it; got: {reason!r}')


def test_spot_instance_is_flagged_as_a_phase_6_deferral(lyceum):
    """PHASE 6 MARKER -- flip this test and the envelope entry together.

    Spot *provisioning* is verified working (measured: h100 spot ready in
    130 s); what is missing is preemption detection, because a reclaimed VM
    carries no spot-specific field and is observable only as a status
    transition. Declaring SPOT_INSTANCE unsupported for the wrong
    reason -- "spot not available" -- would lose that distinction and cost 2.5x
    on h100 indefinitely.
    """
    reason = lyceum._CLOUD_UNSUPPORTED_FEATURES[
        clouds.CloudImplementationFeatures.SPOT_INSTANCE]
    lowered = reason.lower()
    assert 'phase 6' in lowered, (
        'the SPOT_INSTANCE reason must carry the literal string "phase 6" so '
        f'this deferral is greppable; got: {reason!r}')
    assert 'preempt' in lowered, (
        'the SPOT_INSTANCE reason must name preemption detection as the '
        f'blocker, not provisioning; got: {reason!r}')


def test_unsupported_features_for_resources_returns_the_envelope(lyceum):
    """Prevents `check_features_are_supported` seeing an empty dict.

    `_CLOUD_UNSUPPORTED_FEATURES` is inert on its own: SkyPilot only ever reads
    the envelope through `_unsupported_features_for_resources`. A stub returning
    `{}` makes every declaration above a no-op.
    """
    resources = Resources(cloud=Lyceum(), accelerators={'H100': 1})
    assert (Lyceum._unsupported_features_for_resources(resources) ==
            Lyceum._CLOUD_UNSUPPORTED_FEATURES)


# --------------------------------------------------------------------------
# make_deploy_resources_variables -- the template contract
# --------------------------------------------------------------------------
def test_make_deploy_resources_variables_h100_1x_on_demand(lyceum):
    """Pins the ENTIRE dict handed to `templates/lyceum-ray.yml.j2`.

    Adapted from upstream `test_aws_make_deploy_variables`. Asserting the whole
    dict (rather than a few keys) is what catches a key being renamed, dropped,
    or added without the template being updated -- the template renders missing
    variables as empty rather than failing, so a dropped `hardware_profile`
    would produce a cluster YAML with a blank profile and a create call the
    Lyceum API rejects with a 400.

    C1: `ssh_user` must be `lyceum`. The vendor docs say root; root and ubuntu
    are both refused by the real image (`Permission denied (publickey)`).
    """
    resources = Resources(cloud=Lyceum(),
                          instance_type='h100.1x',
                          accelerators={'H100': 1})
    assert _deploy_vars(lyceum, resources) == {
        'instance_type': 'h100.1x',
        'hardware_profile': 'h100',
        'gpu_count': 1,
        'use_spot': False,
        'ssh_user': 'lyceum',
        'region': 'lyceum',
        'custom_resources': '{"H100":1}',
    }


def test_make_deploy_resources_variables_h100_8x(lyceum):
    """Prevents the 8-GPU row collapsing to gpu_count 1.

    `instance_type` ('h100.8x') and `gpu_count` (8) are separate template
    variables feeding separate things: the former is SkyPilot's catalog key, the
    latter goes verbatim into the create payload's `instance_specs.gpu_count`.
    Deriving one and forgetting the other provisions a 1-GPU node for an 8-GPU
    job, which fails much later, on the training script.
    """
    resources = Resources(cloud=Lyceum(),
                          instance_type='h100.8x',
                          accelerators={'H100': 8})
    assert _deploy_vars(lyceum, resources) == {
        'instance_type': 'h100.8x',
        'hardware_profile': 'h100',
        'gpu_count': 8,
        'use_spot': False,
        'ssh_user': 'lyceum',
        'region': 'lyceum',
        'custom_resources': '{"H100":8}',
    }


def test_make_deploy_resources_variables_carries_the_spot_flag(lyceum):
    """Prevents a spot request silently provisioning at the on-demand price.

    Spot is a separate capacity axis with its own price row and its own
    availability list (C9). If `use_spot` does not reach the template it reaches
    neither `instance_type: "spot"` in the create payload nor the node_config,
    and the optimizer's 2.5x saving turns into a full-price node.
    """
    resources = Resources(cloud=Lyceum(),
                          instance_type='h100.1x',
                          accelerators={'H100': 1},
                          use_spot=True)
    variables = _deploy_vars(lyceum, resources)
    assert variables['use_spot'] is True
    assert variables['hardware_profile'] == 'h100'
    assert variables['gpu_count'] == 1


def test_make_deploy_resources_variables_supplies_every_template_variable(
        lyceum):
    """Prevents the template referencing a variable this method never provides.

    The variable set is derived by parsing the shipped template with
    `jinja2.meta.find_undeclared_variables`, not hardcoded, so editing the
    template automatically re-scopes this assertion. Jinja renders an unknown
    variable as the empty string, so the failure mode being prevented is a
    cluster YAML that is structurally valid and semantically wrong.
    """
    referenced = _template_variables()
    required = (referenced - _BACKEND_SUPPLIED_TEMPLATE_VARS
                - _PROVISIONER_SUPPLIED_TEMPLATE_VARS)

    resources = Resources(cloud=Lyceum(),
                          instance_type='h100.8x',
                          accelerators={'H100': 8})
    provided = set(_deploy_vars(lyceum, resources))

    assert required, 'template parsed to zero cloud-supplied variables'
    assert required <= provided, (
        f'template references {sorted(required - provided)} but '
        'make_deploy_resources_variables does not supply them')
    assert provided <= referenced, (
        f'make_deploy_resources_variables supplies {sorted(provided - referenced)} '
        'which the template never reads -- dead contract surface')


def test_lyceum_never_emits_an_ssh_key_id(lyceum):
    """Prevents a careless copy-paste of shadeform-ray.yml.j2.

    Shadeform pre-registers SSH keys and templates an `ssh_key_id`. Lyceum has
    no key registry at all -- the public key goes inline on every
    `POST /vms/create` -- so an `ssh_key_id` variable would be an unfillable
    reference that renders blank and yields an `auth:` block SkyPilot cannot use.
    Also pins that `ssh_user` is templated rather than hardcoded, which is how
    C1 stays enforced in one place.
    """
    referenced = _template_variables()
    assert 'ssh_key_id' not in referenced, (
        'the template must not reference ssh_key_id -- Lyceum has no SSH-key '
        'registry, so nothing could ever fill it in')
    assert 'ssh_user' in referenced, (
        'ssh_user must come from make_deploy_resources_variables, not be '
        'hardcoded in the template')
    # NOTE: deliberately no literal `'ssh_user: {{ssh_user}}' in read_text()`
    # check. That pins Jinja whitespace -- `{{ ssh_user }}` renders identically
    # but would fail it. The two parsed-variable assertions above are the real
    # content.

    resources = Resources(cloud=Lyceum(),
                          instance_type='h100.1x',
                          accelerators={'H100': 1})
    assert 'ssh_key_id' not in _deploy_vars(lyceum, resources)


# --------------------------------------------------------------------------
# check_credentials
# --------------------------------------------------------------------------
def test_check_credentials_signature_matches_skypilot_caller():
    """Prevents `sky check` dying with a TypeError instead of reporting status.

    `sky/check.py:206` calls `cloud.check_credentials(capability)` positionally
    for every registered cloud. A zero-argument override TypeErrors there.
    The damage is contained -- line 209 is a broad `except Exception: ok, reason
    = False, traceback.format_exc()`, so other clouds still report normally --
    but the containment is exactly what makes this worth pinning: Lyceum never
    passes `sky check`, and what an operator sees as the reason is a raw Python
    traceback rather than a credential message.
    """
    try:
        inspect.signature(Lyceum.check_credentials).bind(
            clouds.CloudCapability.COMPUTE)
    except TypeError as e:
        pytest.fail(
            'Lyceum.check_credentials must accept a CloudCapability '
            'positionally (sky/check.py:206). Prefer not overriding it at all '
            f'and implementing _check_compute_credentials instead. Got: {e}')


def test_check_credentials_succeeds_on_user_status(api_key, fixture):
    """Prevents a working key being reported as broken by `sky check`.

    `GET /user/status` is the cheapest authenticated call Lyceum has and the one
    the credential check uses. A 200 must yield `(True, None)`: SkyPilot treats a
    non-None message as a warning to print even when the boolean is True.
    """
    payload = fixture('user_status')
    with mock.patch.object(lyceum_api.LyceumClient,
                           'get_user_status',
                           return_value=payload) as m:
        valid, message = _call_check_credentials()
    assert valid is True, message
    assert message is None
    assert m.call_count == 1, 'check_credentials must actually call the API'


def test_check_credentials_missing_key_file_is_clean_and_actionable(
        monkeypatch, tmp_path):
    """Prevents an unhandled traceback out of `sky check`, and an unhelpful one.

    Adapted from upstream `test_hyperbolic_check_credentials_missing` /
    `test_verda_check_credentials_missing`. With no key anywhere the answer must
    be a tuple, not an exception, and the message must name the thing the
    operator has to create -- the `LYCEUM_API_KEY` environment variable, or the
    `~/.lyceum/api_key` file a deployment seeds from its own secret store.
    "Lyceum credentials not found" alone tells nobody what to do.
    """
    monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    assert not (tmp_path / '.lyceum' / 'api_key').exists()

    valid, message = _call_check_credentials()

    assert valid is False
    assert isinstance(message, str) and message.strip()
    lowered = message.lower()
    assert 'lyceum_api_key' in lowered or '.lyceum/api_key' in lowered, (
        'the message must name the env var or the key path so the operator '
        f'knows what to create; got: {message!r}')


@pytest.mark.parametrize('exc', [
    lyceum_api.LyceumAuthError('401 Unauthorized: invalid API key'),
    lyceum_api.LyceumServerError('502 Bad Gateway'),
    lyceum_api.LyceumNotFoundError('404'),
])
def test_check_credentials_never_raises(api_key, exc):
    """Prevents `sky check` reporting a traceback where a reason belongs.

    `sky/check.py:206` wraps the call in `try:` and line 209 is a broad
    `except Exception: ok, reason = False, traceback.format_exc()`. So an
    exception escaping here does NOT abort the sweep or affect AWS/Shadeform --
    it is contained to Lyceum. What it produces is worse than useless output:
    Lyceum's `sky check` reason becomes a formatted traceback, which buries the
    one line the operator needed. A rejected key, a vendor outage and a 404 must
    all degrade to `(False, message)` here, where we control the wording.
    """
    with mock.patch.object(lyceum_api.LyceumClient,
                           'get_user_status',
                           side_effect=exc):
        valid, message = _call_check_credentials()
    assert valid is False
    assert isinstance(message, str) and message.strip()


def test_check_credentials_auth_error_reports_cleanly(api_key):
    """Prevents a rejected key being reported as a transient failure.

    `LyceumAuthError` is the one outcome an operator can fix (rotate the key);
    conflating it with a server error sends them looking at the vendor status
    page instead.
    """
    with mock.patch.object(
            lyceum_api.LyceumClient,
            'get_user_status',
            side_effect=lyceum_api.LyceumAuthError('rejected by Lyceum')):
        valid, message = _call_check_credentials()
    assert valid is False
    assert 'lyceum' in message.lower()


@pytest.mark.parametrize('exc_factory', [
    lambda key: lyceum_api.LyceumAuthError(f'rejected token {key}'),
    lambda key: lyceum_api.LyceumServerError(f'upstream said no for {key}'),
])
def test_check_credentials_never_leaks_the_api_key(api_key, exc_factory):
    """Prevents a live credential landing in logs, tracebacks and support tickets.

    `sky check` output is routinely pasted into chat and captured in the API
    server's logs, which are not credential-grade storage. The exception raised
    by the API layer may legitimately contain the key (it echoes request
    context); this class is the boundary that must strip it. Also asserts no
    long suffix of the key survives, so masking must be real truncation, not a
    substring that a reader could still recover.
    """
    with mock.patch.object(lyceum_api.LyceumClient,
                           'get_user_status',
                           side_effect=exc_factory(api_key)):
        valid, message = _call_check_credentials()
    assert valid is False
    assert api_key not in message
    assert api_key[-16:] not in message


def test_check_compute_credentials_agrees_with_check_credentials(
        api_key, fixture):
    """Prevents the two entry points disagreeing about the same key.

    `Cloud.check_credentials(COMPUTE)` delegates to `_check_compute_credentials`
    in the base class; if Lyceum overrides both independently they can drift,
    and which one runs depends on the caller (`sky check` vs. the optimizer's
    enabled-clouds refresh).
    """
    payload = fixture('user_status')
    with mock.patch.object(lyceum_api.LyceumClient,
                           'get_user_status',
                           return_value=payload):
        assert Lyceum._check_compute_credentials() == _call_check_credentials()


# --------------------------------------------------------------------------
# get_credential_file_mounts
# --------------------------------------------------------------------------
def test_get_credential_file_mounts_carries_the_api_key_to_the_node(
        lyceum, monkeypatch, tmp_path):
    """Prevents a provisioned node being unable to talk back to Lyceum.

    `sky/check.py:566` reads the result as `{remote_path: local_path}` and drops
    any entry whose local path does not exist, so the mapping must name
    `~/.lyceum/api_key` on both sides -- the same path `api.API_KEY_PATH`
    documents and the same one a deployment seeds on the API server.
    """
    key_path = tmp_path / '.lyceum' / 'api_key'
    key_path.parent.mkdir(parents=True)
    key_path.write_text('lk_' + 'f' * 64)
    monkeypatch.setenv('HOME', str(tmp_path))

    mounts = lyceum.get_credential_file_mounts()

    assert mounts == {'~/.lyceum/api_key': '~/.lyceum/api_key'}
    assert lyceum_cloud_mod._CREDENTIAL_FILES == ['api_key'], (
        'the mount set must be derived from _CREDENTIAL_FILES, not hardcoded '
        'twice')


def test_get_credential_file_mounts_is_constant_when_the_key_is_absent(
        lyceum, monkeypatch, tmp_path):
    """Prevents `sky launch --fast` being busted for every other cloud too.

    This method runs on every launch for every enabled cloud, and its return
    value feeds the file-mounts hash SkyPilot uses to dedupe controller uploads
    and to decide whether setup can be skipped (see the docstring on
    `sky/clouds/nebius.py::_write_nebius_temp_credential_file`). Making the
    result depend on a `stat()` means the hash flips the moment the key is
    rotated or the check races, so it must be a pure constant; `sky/check.py`
    already filters non-existent local paths.
    """
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
    assert not (tmp_path / '.lyceum' / 'api_key').exists()

    mounts = lyceum.get_credential_file_mounts()

    assert mounts == {'~/.lyceum/api_key': '~/.lyceum/api_key'}
    assert all(
        isinstance(k, str) and isinstance(v, str) for k, v in mounts.items())


# --------------------------------------------------------------------------
# Catalog delegation
# --------------------------------------------------------------------------
def test_instance_type_exists_delegates_to_the_catalog(lyceum, monkeypatch):
    """Prevents a second, forked copy of the instance-type rules.

    The catalog is the single source of truth for the 48 `/pricing` rows (C3).
    A `Lyceum`-local reimplementation (e.g. a regex on 'h100.8x') would keep
    answering True for a SKU the vendor has withdrawn, because it never consults
    the price feed at all.
    """
    calls: List[Any] = []

    def _spy(instance_type):
        calls.append(instance_type)
        return True

    monkeypatch.setattr(lyceum_catalog, 'instance_type_exists', _spy)
    assert lyceum.instance_type_exists('h100.8x') is True
    assert calls == ['h100.8x']


def test_get_vcpus_mem_from_instance_type_delegates_to_the_catalog(
        lyceum, monkeypatch):
    """Prevents the measured-specs table being duplicated in the Cloud class.

    vCPU/RAM are exposed by no Lyceum endpoint and are carried from direct
    measurement; b200/b300 are still extrapolated. One copy means one place to
    correct when a real job finally lands on a b300.
    """
    calls: List[Any] = []

    def _spy(instance_type):
        calls.append(instance_type)
        return (256.0, 1448.0)

    monkeypatch.setattr(lyceum_catalog, 'get_vcpus_mem_from_instance_type',
                        _spy)
    assert lyceum.get_vcpus_mem_from_instance_type('h100.8x') == (256.0, 1448.0)
    assert calls == ['h100.8x']


def test_get_accelerators_from_instance_type_delegates_to_the_catalog(
        lyceum, monkeypatch):
    """Prevents accelerator counts being parsed out of the instance-type string.

    Splitting 'h100.8x' locally works right up until a profile is named with a
    dot or an 'x', and it bypasses `ACCELERATOR_NAMES`, which is what maps
    'h100' to SkyPilot's 'H100'.
    """
    calls: List[Any] = []

    def _spy(instance_type):
        calls.append(instance_type)
        return {'H100': 8}

    monkeypatch.setattr(lyceum_catalog, 'get_accelerators_from_instance_type',
                        _spy)
    assert lyceum.get_accelerators_from_instance_type('h100.8x') == {'H100': 8}
    assert calls == ['h100.8x']


def test_instance_type_to_hourly_cost_delegates_to_the_catalog(
        lyceum, monkeypatch):
    """The one delegation the optimizer's decision is actually made of.

    Every cross-cloud comparison reduces to this number plus
    `accelerators_to_hourly_cost` (which is 0.0 by design -- see
    `test_accelerators_to_hourly_cost_is_zero`), so a local price table here
    would not just be a duplicate: it would be the duplicate that decides
    whether Lyceum wins a plan. `region`/`zone` must be forwarded too -- the
    base class and `sky/optimizer.py` both call this with them by keyword, and
    silently dropping them is how a future multi-region catalog starts quoting
    the wrong region's price.
    """
    calls: List[Any] = []

    def _spy(instance_type, use_spot=False, region=None, zone=None):
        calls.append((instance_type, use_spot, region, zone))
        return 22.32

    monkeypatch.setattr(lyceum_catalog, 'get_hourly_cost', _spy)

    assert lyceum.instance_type_to_hourly_cost(
        'h100.8x', use_spot=False,
        region=lyceum_catalog.DEFAULT_REGION, zone=None) == 22.32
    assert calls == [('h100.8x', False, lyceum_catalog.DEFAULT_REGION, None)], (
        'instance_type_to_hourly_cost must forward instance_type, use_spot, '
        f'region and zone to the catalog; got {calls}')

    calls.clear()
    lyceum.instance_type_to_hourly_cost('h100.8x', use_spot=True)
    assert calls == [('h100.8x', True, None, None)], (
        'the spot flag must reach the catalog -- on-demand and spot are '
        f'separate price rows (C3/C9); got {calls}')


def test_instance_type_to_hourly_cost_real_values(lyceum, offline_api):
    """Prevents the delegation returning a price the catalog never quoted.

    Runs against the real catalog with the API cut (baked-CSV fallback, design
    S6) and compares to `lyceum_catalog.get_hourly_cost` directly, so the two
    cannot drift. Also pins that spot is strictly cheaper: if the method ignores
    `use_spot`, the optimizer's 2.5x saving silently disappears.
    """
    on_demand = lyceum.instance_type_to_hourly_cost('h100.8x', use_spot=False)
    assert on_demand == lyceum_catalog.get_hourly_cost('h100.8x',
                                                       use_spot=False)
    assert isinstance(on_demand, float) and on_demand > 0.0

    spot = lyceum.instance_type_to_hourly_cost('h100.8x', use_spot=True)
    assert spot == lyceum_catalog.get_hourly_cost('h100.8x', use_spot=True)
    assert spot < on_demand, (
        f'spot ({spot}) must be cheaper than on-demand ({on_demand}); equal '
        'prices mean use_spot never reached the catalog')


def test_get_accelerators_from_instance_type_real_values(lyceum, offline_api):
    """Prevents the optimizer being told an 8-GPU node has one GPU.

    Runs against the real catalog with the API cut, i.e. the baked CSV fallback
    -- the path that is actually live whenever the Lyceum API is
    unreachable, and the one that silently degrades to an empty frame if the
    fallback is wrong.
    """
    assert lyceum.get_accelerators_from_instance_type('h100.8x') == {'H100': 8}
    assert lyceum.get_accelerators_from_instance_type('l40s.1x') == {'L40S': 1}


def test_get_vcpus_mem_from_instance_type_real_values(lyceum, offline_api):
    """Prevents vCPU/RAM drifting from the measured hardware.

    Expected values are derived from `catalog.INSTANCE_SPECS` rather than
    retyped, so correcting a measurement in one place updates the expectation
    too. h100 is the interesting row: 32 vCPU per GPU, more than the pricier
    h200's 16, so any "vCPU tracks GPU class" shortcut fails here.
    """
    for instance_type, profile, count in [('h100.8x', 'h100', 8),
                                          ('l40s.1x', 'l40s', 1),
                                          ('h200.2x', 'h200', 2)]:
        vcpus_per_gpu, mem_per_gpu, _, measured = (
            lyceum_catalog.INSTANCE_SPECS[profile])
        assert measured, f'{profile} specs are extrapolated, not measured'
        assert lyceum.get_vcpus_mem_from_instance_type(instance_type) == (
            float(vcpus_per_gpu * count), float(mem_per_gpu * count))


# --------------------------------------------------------------------------
# Regions and zones -- Lyceum has neither
# --------------------------------------------------------------------------
def test_regions_with_offering_returns_exactly_one_synthetic_region(lyceum):
    """Prevents the failover loop iterating regions that do not exist.

    Lyceum is a single EU footprint with no region or zone concept; the one
    synthetic region exists only to satisfy `sky/catalog/common.py`. Claiming
    more would make SkyPilot retry a capacity failure (a fast, free 500 -- C7)
    once per fictional region instead of failing over to Shadeform.
    """
    regions = Lyceum.regions_with_offering('h100.8x', {'H100': 8},
                                           use_spot=False,
                                           region=None,
                                           zone=None)
    assert len(regions) == 1
    assert regions[0].name == lyceum_catalog.DEFAULT_REGION
    assert regions[0].zones is None, 'Lyceum must not advertise zones'


def test_regions_with_offering_filters_by_requested_region(lyceum):
    """Prevents a mistyped `region:` silently landing on the only region anyway.

    If the filter is skipped, `region: us-east-1` on a Lyceum task provisions in
    the EU without complaint -- a data-residency surprise rather than a launch
    error.
    """
    matching = Lyceum.regions_with_offering('h100.1x', {'H100': 1},
                                            use_spot=False,
                                            region=lyceum_catalog.
                                            DEFAULT_REGION,
                                            zone=None)
    assert [r.name for r in matching] == [lyceum_catalog.DEFAULT_REGION]

    assert Lyceum.regions_with_offering('h100.1x', {'H100': 1},
                                        use_spot=False,
                                        region='us-east-1',
                                        zone=None) == []


def test_zones_provision_loop_yields_none_exactly_once(lyceum):
    """Prevents the provisioner looping forever, or into a nonexistent zone.

    SkyPilot drives one provision attempt per yielded item, and `None` means
    "this cloud has no zones". Yielding a zone object sends a zone name into a
    create payload that has no field for it; yielding twice doubles the cost of
    every capacity failure.
    """
    yielded = list(
        Lyceum.zones_provision_loop(region=lyceum_catalog.DEFAULT_REGION,
                                    num_nodes=1,
                                    instance_type='h100.1x',
                                    accelerators={'H100': 1},
                                    use_spot=False))
    assert yielded == [None]


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
def test_accelerators_to_hourly_cost_is_zero(lyceum):
    """Prevents the optimizer double-counting the GPU and never picking Lyceum.

    `/pricing`'s `vm_running` rows are whole-instance prices that already
    include the GPUs (C3), and SkyPilot adds `instance_type_to_hourly_cost` and
    `accelerators_to_hourly_cost` together. Returning a per-GPU price here would
    inflate an h100.8x from $22.32/h to something no plan ever wins with, so
    Lyceum would be quietly dead in every `infra: "*"` comparison.
    """
    for accelerators in [{'H100': 1}, {'H100': 8}, {'B300': 8}]:
        for use_spot in (False, True):
            assert lyceum.accelerators_to_hourly_cost(accelerators,
                                                      use_spot=use_spot) == 0.0


def test_egress_cost_is_zero(lyceum):
    """Prevents an invented egress price skewing cross-cloud comparison.

    Lyceum publishes no egress meter; `/pricing` carries `vm_running` only.
    A guessed number would bias the optimizer against Lyceum for
    data-heavy jobs on evidence that does not exist.
    """
    assert lyceum.get_egress_cost(1024.0) == 0.0


# --------------------------------------------------------------------------
# _get_feasible_launchable_resources
# --------------------------------------------------------------------------
def test_feasible_resources_maps_h100_8_to_the_h100_8x_instance(
        lyceum, offline_api):
    """Prevents `accelerators: H100:8` failing to resolve to a launchable plan.

    This is the path essentially every real task takes: it asks for GPUs, never
    for a Lyceum instance type. The returned resources must be launchable
    (cloud + instance_type both set) or `sky launch` asserts deep inside the
    optimizer with no useful message.
    """
    requested = Resources(cloud=Lyceum(), accelerators={'H100': 8})
    feasible = lyceum._get_feasible_launchable_resources(requested)

    assert isinstance(feasible, resources_utils.FeasibleResources)
    instance_types = [r.instance_type for r in feasible.resources_list]
    assert 'h100.8x' in instance_types
    for r in feasible.resources_list:
        assert isinstance(r.cloud, Lyceum)
        assert r.accelerators == {'H100': 8}


def test_feasible_resources_is_empty_for_an_accelerator_lyceum_lacks(
        lyceum, offline_api):
    """Prevents an unavailable GPU crashing the optimizer instead of failing over.

    With `infra: "*"` the optimizer asks every enabled cloud about every
    accelerator. Lyceum has exactly six profiles (a100, b200, b300, h100, h200,
    l40s); anything else must come back as an empty `FeasibleResources`, not an
    exception -- an exception here takes down the plan for Shadeform too.
    """
    requested = Resources(cloud=Lyceum(), accelerators={'MI300X': 8})
    feasible = lyceum._get_feasible_launchable_resources(requested)

    assert isinstance(feasible, resources_utils.FeasibleResources)
    assert feasible.resources_list == []


def test_feasible_resources_does_not_raise_for_any_unknown_accelerator(
        lyceum, offline_api):
    """Same guarantee, swept over shapes that have historically broken clouds.

    A zero count and a lowercase name are the two spellings that reach clouds
    from hand-written task YAML; both must be answered, not raised on.
    """
    for accelerators in [{'MI300X': 1}, {'V100': 4}, {'h100': 8}]:
        requested = Resources(cloud=Lyceum(), accelerators=accelerators)
        feasible = lyceum._get_feasible_launchable_resources(requested)
        assert isinstance(feasible, resources_utils.FeasibleResources)


def test_query_status_does_not_raise(lyceum):
    """Prevents cluster-status refresh crashing for a SKYPILOT-version cloud.

    With `STATUS_VERSION is SKYPILOT` the real status query lives in
    `provision/instance.py::query_instances`; this method still exists on the
    Cloud interface and is called during validation, so it must return a list
    rather than inherit the base-class NotImplementedError.
    """
    result = Lyceum.query_status('sky-lyceum-test', {}, region=None, zone=None)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# The skylet has no plugin hook — which is WHY node_autodown ships a .pth
# ---------------------------------------------------------------------------
def test_the_skylet_still_has_no_plugin_hook_so_the_pth_is_still_needed():
    """A tripwire for the one hack in this package.

    Autodown runs on the node: the skylet's autostop event dispatches
    `terminate_instances('lyceum', ...)` locally, which needs our provisioner
    REGISTERED in the skylet process. Registration normally happens via
    `plugins.load_plugins()` -- and the skylet never calls it.

    This test used to conclude "therefore node-side autodown is impossible for
    any out-of-tree cloud", and that conclusion was WRONG: it assumed the plugin
    loader is the only way to run code in that process. A `.pth` file executes
    at interpreter startup with no framework hook at all, which is what
    `node_autodown.py` uses. Verified live 2026-08-07 -- the node deleted itself.

    So the observation below is still true and still load-bearing; only the
    conclusion changed. The day this test FAILS, upstream has grown skylet-side
    plugin loading, the `.pth` becomes unnecessary, and node_autodown collapses
    to an ordinary `enable()` call. Delete the hack then, not before.
    """
    import pathlib as _pathlib

    import sky

    sky_root = _pathlib.Path(sky.__file__).parent
    callers = set()
    for path in sky_root.rglob('*.py'):
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if 'plugins.load_plugins(' in text:
            callers.add(path.relative_to(sky_root).as_posix())

    assert callers, 'load_plugins has no callers at all -- did it move?'
    skylet_callers = {c for c in callers if c.startswith('skylet/')}
    assert not skylet_callers, (
        'the skylet now loads plugins, so node-side autodown may be possible; '
        f'the .pth hack can then be deleted. callers: {skylet_callers}')
