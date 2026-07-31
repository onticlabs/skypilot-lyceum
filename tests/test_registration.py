"""The seams `enable()` stands on, pinned.

Exactly which mechanisms are public SkyPilot API and which are patches:

    registry.CLOUD_REGISTRY.register              public
    provision.register_provisioner                public
    Provisioner.template_override                 public
    skylet.constants.ALL_CLOUDS += ('lyceum',)    PATCH
    sys.modules['sky.catalog.lyceum_catalog']     PATCH
    backend_utils._add_auth_to_cluster_config     PATCH

Every one of these fails silently or late:

* an unregistered CLOUD_REGISTRY entry surfaces as `ValueError: Cloud 'lyceum'
  is not a valid cloud` from whatever code path happens to call `from_str`;
* a missing ALL_CLOUDS entry surfaces as a JSON-schema rejection of the user's
  YAML -- client-side, before any of our code runs, with a regex in the message
  and no mention of Lyceum;
* a missing `sky.catalog.lyceum_catalog` surfaces as `ValueError: Cannot find
  module "sky.catalog.lyceum_catalog"` deep inside the optimizer;
* a missing provisioner surfaces as `AssertionError: Unknown provider: lyceum`
  in the middle of a launch, after the user has waited.

None of them is caught by importing the package. That is what this file is for.

Isolation: every test here mutates process-global state (a module-level
registry dict, a module constant, `sys.modules`). The `_lyceum_globals` fixture
snapshots and restores all four, and `_leak_detector` fails the module if
anything escapes anyway. Deliberately paranoid: a flaky global-state test is
worse than no test.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest
import sky
from sky import provision as sky_provision
from sky.skylet import constants
from sky.utils import registry

PKG = 'skypilot_lyceum'
CLOUD_NAME = 'lyceum'
CATALOG_MODULE = f'sky.catalog.{CLOUD_NAME}_catalog'

_MISSING = object()


# ---------------------------------------------------------------------------
# Global-state isolation
# ---------------------------------------------------------------------------
def _lyceum_module_names():
    return [
        name for name in list(sys.modules)
        if name == PKG or name.startswith(f'{PKG}.')
    ]


def _evict_lyceum_modules():
    """Drop our package from sys.modules so a later import re-executes it.

    `cloud.Lyceum` registers itself with a decorator at import time, so without
    eviction the second test in this file would import nothing, register
    nothing, and fail for a reason that has nothing to do with what it asserts.
    """
    for name in _lyceum_module_names():
        del sys.modules[name]


def _clear_lyceum_state():
    """Return SkyPilot's globals to their pre-`enable()` condition."""
    reg = registry.CLOUD_REGISTRY
    reg.pop(CLOUD_NAME, None)
    aliases = getattr(reg, '_aliases', None)
    if isinstance(aliases, dict):
        for alias, target in list(aliases.items()):
            if target == CLOUD_NAME:
                del aliases[alias]
    if isinstance(constants.ALL_CLOUDS, tuple):
        constants.ALL_CLOUDS = tuple(
            name for name in constants.ALL_CLOUDS if name != CLOUD_NAME)
    # pylint: disable=protected-access
    sky_provision._registered_provisioners.pop(CLOUD_NAME, None)
    sys.modules.pop(CATALOG_MODULE, None)
    _evict_lyceum_modules()


def _global_fingerprint():
    # pylint: disable=protected-access
    return (
        tuple(constants.ALL_CLOUDS),
        tuple(sorted(registry.CLOUD_REGISTRY)),
        tuple(sorted(sky_provision._registered_provisioners)),
        CATALOG_MODULE in sys.modules,
    )


@pytest.fixture(scope='module', autouse=True)
def _leak_detector():
    """Fail this module if any test leaks Lyceum state into the rest of the run.

    Set up before the first test's function-scoped fixtures, torn down after
    the last one, so `baseline` is genuinely pristine.
    """
    baseline = _global_fingerprint()
    yield
    after = _global_fingerprint()
    assert after == baseline, (
        'test_registration leaked global SkyPilot state into the rest of the '
        f'suite.\n  before: {baseline}\n  after:  {after}')


@pytest.fixture(autouse=True)
def _lyceum_globals():
    """Snapshot/restore every global `enable()` touches, and start clean."""
    reg = registry.CLOUD_REGISTRY
    # pylint: disable=protected-access
    saved = {
        'registry': dict(reg),
        'aliases': dict(getattr(reg, '_aliases', {}) or {}),
        'all_clouds': constants.ALL_CLOUDS,
        'provisioners': dict(sky_provision._registered_provisioners),
        'catalog_module': sys.modules.get(CATALOG_MODULE, _MISSING),
        'lyceum_modules': {n: sys.modules[n] for n in _lyceum_module_names()},
    }
    _clear_lyceum_state()
    try:
        yield
    finally:
        reg.clear()
        reg.update(saved['registry'])
        aliases = getattr(reg, '_aliases', None)
        if isinstance(aliases, dict):
            aliases.clear()
            aliases.update(saved['aliases'])
        constants.ALL_CLOUDS = saved['all_clouds']
        sky_provision._registered_provisioners.clear()
        sky_provision._registered_provisioners.update(saved['provisioners'])
        if saved['catalog_module'] is _MISSING:
            sys.modules.pop(CATALOG_MODULE, None)
        else:
            sys.modules[CATALOG_MODULE] = saved['catalog_module']
        _evict_lyceum_modules()
        sys.modules.update(saved['lyceum_modules'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _enable(**kwargs):
    """Fresh import + `enable()`. Returns the package module."""
    package = importlib.import_module(PKG)
    package.enable(**kwargs)
    return package


def _lyceum_cloud_cls():
    """The `Lyceum` class -- WITHOUT importing it ourselves.

    Reading it out of `sys.modules` rather than calling `import_module` matters:
    importing `skypilot_lyceum.cloud` runs the `@CLOUD_REGISTRY.register`
    decorator, so a test helper that imported it would paper over an `enable()`
    that never did.
    """
    module = sys.modules.get(f'{PKG}.cloud')
    assert module is not None, (
        'enable() did not import skypilot_lyceum.cloud, so the '
        '@CLOUD_REGISTRY.register decorator never ran')
    return module.Lyceum


def _parse(resources: dict):
    """Parse a one-resource task config the way a `sky launch` client does."""
    task = sky.Task.from_yaml_config({'resources': resources, 'run': 'echo hi'})
    resource_set = list(task.resources)
    assert len(resource_set) == 1, resource_set
    return resource_set[0]


# ---------------------------------------------------------------------------
# 1. CLOUD_REGISTRY
# ---------------------------------------------------------------------------
def test_enable_registers_cloud_in_registry():
    """Without this, every `CLOUD_REGISTRY.from_str('lyceum')` raises ValueError.

    `from_str` is what turns the string in a user's YAML, and the string
    stored in SkyPilot's cluster DB, back into a Cloud object;
    an unregistered cloud makes `sky status` on an existing Lyceum cluster
    explode, not just `sky launch`.
    """
    _enable()

    resolved = registry.CLOUD_REGISTRY.from_str(CLOUD_NAME)

    assert resolved is not None
    assert isinstance(resolved, _lyceum_cloud_cls()), (
        f'from_str({CLOUD_NAME!r}) returned {resolved!r} of type '
        f'{type(resolved)!r}, not a Lyceum instance')
    # The registry stores instances, not classes (see _Registry.register).
    assert not isinstance(resolved, type)
    # `from_str` lowercases; repr() is what lands in the cluster DB.
    assert isinstance(registry.CLOUD_REGISTRY.from_str('Lyceum'),
                      _lyceum_cloud_cls())
    assert repr(resolved) == 'Lyceum'


# ---------------------------------------------------------------------------
# 2. ALL_CLOUDS
# ---------------------------------------------------------------------------
def test_enable_adds_lyceum_to_all_clouds():
    """The patched constant that makes the task-YAML schema accept 'lyceum'.

    One of the unsupported patches. `ALL_CLOUDS` is read at call time by
    `sky.utils.schemas._get_infra_pattern()` and by the
    `cloud` enum, so this is a hard prerequisite for any YAML to parse.
    """
    _enable()

    assert isinstance(constants.ALL_CLOUDS, tuple), (
        'ALL_CLOUDS must stay a tuple -- schemas.py does list(ALL_CLOUDS) and '
        'other call sites index it')
    assert all(isinstance(name, str) for name in constants.ALL_CLOUDS)
    assert CLOUD_NAME in constants.ALL_CLOUDS
    assert constants.ALL_CLOUDS.count(CLOUD_NAME) == 1, (
        'lyceum appears more than once; the patch is appending unconditionally '
        'instead of guarding')


# ---------------------------------------------------------------------------
# 3. Both YAML spellings parse (highest blast radius)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('key', ['infra', 'cloud'])
def test_task_yaml_accepts_lyceum(key):
    """Both spellings are rejected without the ALL_CLOUDS patch. Verified.

    `infra:` goes through `_get_infra_pattern()`'s regex and `cloud:` through
    a `case_insensitive_enum`; both are built from
    `ALL_CLOUDS`, and both reject 'lyceum' unpatched with an
    InvalidSkyPilotConfigError raised client-side. `render.py` emits `infra:`
    today, but a hand-written `sky.yaml` may use either, so both must work.
    """
    _enable()

    resources = _parse({key: CLOUD_NAME})

    assert isinstance(resources.cloud, _lyceum_cloud_cls()), (
        f'resources.cloud for {key}: lyceum is {resources.cloud!r} '
        f'({type(resources.cloud)!r}), not a Lyceum instance')


def test_task_yaml_accepts_lyceum_with_accelerators():
    """The claim that was verified end-to-end.

    `{infra: lyceum, accelerators: H100:1}` must resolve to
    `Lyceum({'H100': 1})` -- i.e. the accelerator request survives alongside
    the cloud, which is the shape essentially every real task uses.
    """
    _enable()

    resources = _parse({'infra': CLOUD_NAME, 'accelerators': 'H100:1'})

    assert isinstance(resources.cloud, _lyceum_cloud_cls())
    assert resources.accelerators == {'H100': 1}


def test_client_only_enable_still_parses_infra():
    """The laptop half: parse `infra: lyceum` with no credentials, no provisioner.

    Optimization and provisioning are server-side; the client half exists purely
    to make `infra: lyceum` parse. A `client_only` enable that skipped the cloud
    registration would break the client before it ever reached the API server.
    """
    _enable(client_only=True)

    resources = _parse({'infra': CLOUD_NAME})
    assert isinstance(resources.cloud, _lyceum_cloud_cls())
    assert sky_provision.get_registered_provisioner(CLOUD_NAME) is None, (
        'client_only=True must not register the provisioner -- provisioning is '
        'server-side')


def test_client_only_enable_still_injects_the_catalog_module():
    """The trap in `client_only`: patching ALL_CLOUDS without the catalog module.

    INTENDED ANSWER, stated so this test pins a decision rather than describing
    an accident: the catalog module MUST be injected even client-side.

    Why the other side of the trade is not available. `sky.catalog.
    _map_clouds_catalog` with `clouds=None` does
    `clouds = list(constants.ALL_CLOUDS)` and then, for each one,
    `importlib.import_module(f'sky.catalog.{cloud}_catalog')`. `client_only`
    still applies the ALL_CLOUDS patch (it has to -- that is the whole point of
    the client half: making `infra: lyceum` parse). So the moment 'lyceum' is in
    ALL_CLOUDS without a `sky.catalog.lyceum_catalog` module, EVERY unfiltered
    catalog call on the laptop raises `ValueError: Cannot find module
    "sky.catalog.lyceum_catalog"` -- `sky show-gpus`, `list_accelerators`, the
    lot -- for every cloud, not just ours. A half-patch is strictly worse than
    either whole.

    Injecting the module costs nothing client-side: `skypilot_lyceum.catalog`
    reads no credentials at import, and its network calls only happen when a
    catalog function is actually called. So it is injected, and `client_only`
    means exactly one thing: no provisioner.
    """
    _enable(client_only=True)

    assert CATALOG_MODULE in sys.modules, (
        'client_only=True patched ALL_CLOUDS but did not inject '
        f'{CATALOG_MODULE}; _map_clouds_catalog iterates ALL_CLOUDS and will '
        'now fail for every cloud on the client')
    assert importlib.import_module(CATALOG_MODULE) is importlib.import_module(
        f'{PKG}.catalog')

    # Reproduce the dispatcher's own unfiltered sweep verbatim: `clouds=None`
    # becomes `list(constants.ALL_CLOUDS)`, and each entry is resolved with
    # `importlib.import_module(f'sky.catalog.{cloud.lower()}_catalog')`. This is
    # the resolution step only -- what the catalog functions then DO is
    # tests/test_catalog.py's business -- because resolution is the step the
    # missing injection breaks, and it breaks it for every cloud in the list.
    assert CLOUD_NAME in constants.ALL_CLOUDS
    for cloud in constants.ALL_CLOUDS:
        importlib.import_module(f'sky.catalog.{cloud.lower()}_catalog')


# ---------------------------------------------------------------------------
# 4. Catalog module dispatch
# ---------------------------------------------------------------------------
def test_catalog_module_is_importable_under_skys_name():
    """`sky.catalog` resolves catalogs by hardcoded import; there is no registry.

    The second unsupported patch. `_map_clouds_catalog` does
    `importlib.import_module(f'sky.catalog.{cloud.lower()}_catalog')` and turns
    a ModuleNotFoundError into `ValueError: Cannot find module ...`, which
    surfaces inside the optimizer rather than at import time.
    """
    _enable()

    imported = importlib.import_module(CATALOG_MODULE)

    assert imported is importlib.import_module(f'{PKG}.catalog'), (
        f'{CATALOG_MODULE} resolved to {imported!r}, which is not our catalog '
        'module -- two catalog objects means TTL caches diverge')
    # Reproduce the dispatcher's own lookup verbatim, not a paraphrase of it.
    dispatched = importlib.import_module(
        f'sky.catalog.{CLOUD_NAME.lower()}_catalog')
    assert dispatched is imported


# ---------------------------------------------------------------------------
# 5. Provisioner registration + dispatch
# ---------------------------------------------------------------------------
def test_provisioner_is_registered():
    """Unregistered means `AssertionError: Unknown provider: lyceum` mid-launch.

    `_route_to_cloud_impl` asserts a module exists for the provider name; for
    an out-of-tree cloud there is no `sky/provision/lyceum`
    fallback, so the registration is the only thing standing between us and an
    assertion error after the user has already waited for the optimizer.
    """
    _enable()

    provisioner = sky_provision.get_registered_provisioner(CLOUD_NAME)

    assert provisioner is not None
    assert provisioner.module is sys.modules[f'{PKG}.provision'], (
        f'registered module is {provisioner.module!r}, not our provision '
        'package')


def test_provision_dispatch_routes_to_our_module(monkeypatch):
    """Registration is not enough -- the routed call must actually reach us.

    Exercises the real `sky.provision` dispatcher (signature bind, provider
    lookup, `getattr(plugin_module, func.__name__)`) end-to-end with a recorder
    standing in for one of the nine functions. A registration under the wrong
    key, or a `Provisioner` whose module lacks the attribute, silently falls
    through to the in-tree module lookup instead.
    """
    _enable()
    calls = []

    def _recorder(cluster_name_on_cloud, ports, provider_config=None):
        calls.append((cluster_name_on_cloud, ports, provider_config))

    monkeypatch.setattr(sys.modules[f'{PKG}.provision'], 'open_ports',
                        _recorder)

    sky_provision.open_ports(CLOUD_NAME, 'sky-abc-lyceum', ['22'])

    assert calls == [('sky-abc-lyceum', ['22'], None)]


# ---------------------------------------------------------------------------
# 6. Idempotence
# ---------------------------------------------------------------------------
def test_enable_is_idempotent():
    """Two `enable()` calls in one process must not raise.

    Real path: a client imports and enables client-side while the API server's
    `plugins.yaml` enables the same package in the same interpreter.
    `_Registry.register` does
    `assert name not in self, f'{name} already registered'`, so an unguarded
    second registration is an AssertionError at import time -- and with `python
    -O` it would instead register silently twice. The ALL_CLOUDS count check
    below catches the other naive shape, an unconditional `+= ('lyceum',)`.
    """
    _enable()
    before = _global_fingerprint()

    _enable()  # must be a no-op, not an AssertionError

    assert _global_fingerprint() == before
    assert constants.ALL_CLOUDS.count(CLOUD_NAME) == 1
    assert isinstance(registry.CLOUD_REGISTRY.from_str(CLOUD_NAME),
                      _lyceum_cloud_cls())
    assert importlib.import_module(CATALOG_MODULE) is importlib.import_module(
        f'{PKG}.catalog')


def test_patches_apply_is_idempotent():
    """`patches.apply()` is reachable on its own; it must be re-runnable too."""
    _enable()
    patches = importlib.import_module(f'{PKG}.patches')

    patches.apply()
    patches.apply()

    assert constants.ALL_CLOUDS.count(CLOUD_NAME) == 1
    assert sys.modules[CATALOG_MODULE] is importlib.import_module(
        f'{PKG}.catalog')


# ---------------------------------------------------------------------------
# 7. Existing clouds keep working
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('cloud_name', ['shadeform', 'aws'])
def test_existing_clouds_still_parse_after_enable(cloud_name):
    """Lyceum is co-equal to the in-tree clouds, not a replacement for them.

    The ALL_CLOUDS patch rewrites the alternation the `infra` regex is built
    from. Get that wrong -- a stray separator, a replaced tuple, a lost entry --
    and today's `infra: shadeform` jobs stop parsing. This is a verified
    property of the patch, so it is pinned here.
    """
    _enable()

    resources = _parse({'infra': cloud_name})

    assert resources.cloud is not None
    # Equality, not a prefix: an earlier version compared `cloud_name[:3]`,
    # which would happily accept a *different* cloud whose repr starts the same
    # way -- exactly the mis-resolution this test claims to rule out.
    assert repr(resources.cloud).lower() == cloud_name, (
        f'infra: {cloud_name} resolved to {resources.cloud!r}')
    assert type(resources.cloud) is type(  # pylint: disable=unidiomatic-typecheck
        registry.CLOUD_REGISTRY.from_str(cloud_name)), (
            f'infra: {cloud_name} produced a '
            f'{type(resources.cloud).__name__}, not the registered class')


def test_unknown_cloud_is_still_rejected():
    """The patch must widen the schema by exactly one name, not defeat it.

    A patch that replaced the enum/regex with a wildcard would make every test
    above pass while letting `infra: typo` through to the optimizer, where it
    fails much later and much less clearly.
    """
    _enable()

    with pytest.raises(Exception) as exc_info:
        _parse({'infra': 'definitely-not-a-cloud'})
    assert 'definitely-not-a-cloud' in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. Anchor drift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'drifted',
    [
        pytest.param(['aws', 'gcp'], id='list-not-tuple'),
        pytest.param(None, id='none'),
        pytest.param(('aws', 42), id='tuple-of-non-str'),
    ],
)
def test_patch_all_clouds_refuses_on_anchor_drift(drifted, monkeypatch):
    """A drifted upstream must stop the build, not be guessed at.

    `patch_all_clouds`'s stated anchor is "ALL_CLOUDS is a tuple of str". This
    is the discipline every patch here follows: assert the anchor first, raise
    loudly, never half-apply. The failure mode this prevents is a SkyPilot
    version bump that reshapes ALL_CLOUDS and leaves us
    with a mangled constant that breaks *every* cloud's YAML parsing, in
    production, instead of failing CI.
    """
    patches = importlib.import_module(f'{PKG}.patches')
    monkeypatch.setattr(constants, 'ALL_CLOUDS', drifted, raising=True)

    with pytest.raises(patches.PatchDriftError):
        patches.patch_all_clouds()

    assert constants.ALL_CLOUDS is drifted, (
        'the patch mutated ALL_CLOUDS before/while refusing -- a half-applied '
        'patch is exactly what the anchor check exists to prevent')


def test_patch_catalog_module_refuses_on_anchor_drift(monkeypatch):
    """The second patch's anchor was documented but never tested.

    `patches.patch_catalog_module`'s docstring names a CHECKABLE anchor:
    `importlib.import_module('sky.catalog.shadeform_catalog')` must still
    succeed, i.e. the hardcoded `sky.catalog.<cloud>_catalog` convention we are
    exploiting is still how in-tree catalogs resolve. If upstream introduces a
    real catalog registry or renames the convention, injecting a module into
    `sys.modules` under that name is a no-op that nothing will ever look up --
    and the failure surfaces as `ValueError: Cannot find module
    "sky.catalog.lyceum_catalog"` inside the optimizer, in production.

    `patch_all_clouds` has drift coverage; this one had none, which is a
    documented anchor with nothing enforcing it -- a half-applied patch waiting
    to ship.

    Drift is simulated at the import-system level (a `sys.meta_path` finder that
    refuses the name, plus eviction from `sys.modules`) rather than by
    monkeypatching `importlib.import_module`. That way the test does not care
    whether the patch spells it `importlib.import_module(...)`, `from importlib
    import import_module`, or a bare `import`: the import genuinely fails, which
    is the shape upstream removing the convention would take.
    """
    patches = importlib.import_module(f'{PKG}.patches')
    anchor = 'sky.catalog.shadeform_catalog'

    class _Blocker:

        @staticmethod
        def find_module(fullname, path=None):  # py2-style hook, harmless
            return None

        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == anchor:
                raise ModuleNotFoundError(f'No module named {fullname!r}',
                                          name=fullname)
            return None

    monkeypatch.delitem(sys.modules, anchor, raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_Blocker] + list(sys.meta_path))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(anchor)  # the drift is real, not assumed

    with pytest.raises(patches.PatchDriftError):
        patches.patch_catalog_module()

    assert CATALOG_MODULE not in sys.modules, (
        'the patch injected sky.catalog.lyceum_catalog before/while refusing '
        '-- a half-applied patch is exactly what the anchor check prevents')


def test_enable_propagates_patch_drift(monkeypatch):
    """`enable()` must not swallow a PatchDriftError into a warning.

    Degrading to "Lyceum unavailable" would make the API server come up looking
    healthy while every Lyceum launch fails elsewhere; the design's rule is
    fail-loud (4.1).
    """
    patches = importlib.import_module(f'{PKG}.patches')
    monkeypatch.setattr(constants, 'ALL_CLOUDS', ['aws', 'gcp'], raising=True)

    with pytest.raises(patches.PatchDriftError):
        importlib.import_module(PKG).enable()


# ---------------------------------------------------------------------------
# 9. Cluster-auth patch (found by the first real provision attempt)
# ---------------------------------------------------------------------------
def test_upstream_still_refuses_lyceum_in_add_auth():
    """The third seam, and the reason `patch_cluster_auth` exists.

    `backend_utils._add_auth_to_cluster_config` is an isinstance chain over
    in-tree cloud classes ending in `assert False, cloud`. Unlike the cluster
    template (which has `template_override`) there is NO hook. A real
    `sky.launch(infra='lyceum')` died here with a bare `AssertionError: Lyceum`
    after the optimizer had already chosen the resource.

    If upstream ever adds a default branch or a hook, this test fails and the
    patch can be deleted -- which is exactly when we want to hear about it.
    """
    from sky.backends import backend_utils

    fn = getattr(backend_utils, '_add_auth_to_cluster_config', None)
    assert fn is not None, 'the function this patch wraps has moved'
    # Unwrap if a previous test already applied the patch: this assertion is
    # about UPSTREAM's behaviour, and inspecting our own wrapper would make the
    # test pass or fail on collection order rather than on what it claims.
    fn = getattr(fn, '_lyceum_original', fn)
    source = inspect.getsource(fn)
    assert 'assert False' in source, (
        'upstream no longer hard-asserts on an unknown cloud here; re-check '
        'whether patch_cluster_auth is still needed')


def test_patch_cluster_auth_routes_lyceum_to_the_generic_ssh_setup(tmp_path):
    """Lyceum needs `auth.configure_ssh_info`, not a bespoke setup.

    That generic helper substitutes `skypilot:ssh_public_key_content` (and
    `skypilot:ssh_user`) into the rendered config, which is precisely what
    `templates/lyceum-ray.yml.j2` contains. Shadeform needs a bespoke variant
    only because it registers the key with the account and gets an id back;
    Lyceum takes the key inline on create and has no key registry, so the
    generic path is the correct one.
    """
    import yaml as _yaml
    from sky.backends import backend_utils

    _enable()
    lyceum = registry.CLOUD_REGISTRY.from_str('lyceum')

    cfg = {'auth': {'ssh_user': 'lyceum', 'ssh_private_key': '~/.ssh/sky-key'},
           'available_node_types': {'ray_head_default': {'node_config': {
               'PublicKey': 'skypilot:ssh_public_key_content'}}}}
    path = tmp_path / 'cluster.yml'
    path.write_text(_yaml.safe_dump(cfg))

    backend_utils._add_auth_to_cluster_config(lyceum, str(path))

    out = _yaml.safe_load(path.read_text())
    injected = out['available_node_types']['ray_head_default']['node_config'][
        'PublicKey']
    assert injected != 'skypilot:ssh_public_key_content', (
        'the placeholder was never substituted — the VM would be created with '
        'a literal placeholder as its authorized key and be unreachable')
    assert injected.startswith(('ssh-', 'ecdsa-')), injected


def test_patch_cluster_auth_leaves_other_clouds_alone(tmp_path):
    """Wrapping a shared upstream function must not change any other cloud.

    This patch sits on the path EVERY cloud's launch takes. Getting it wrong
    breaks Shadeform, which is carrying the real work today.
    """
    import yaml as _yaml
    from sky.backends import backend_utils

    _enable()
    shadeform = registry.CLOUD_REGISTRY.from_str('shadeform')
    calls = []
    original = backend_utils._add_auth_to_cluster_config

    # The wrapper must delegate for anything that is not Lyceum. Prove it by
    # asserting the delegate is reached, rather than by inspecting internals.
    cfg = {'auth': {'ssh_user': 'shadeform'}}
    path = tmp_path / 'c.yml'
    path.write_text(_yaml.safe_dump(cfg))
    try:
        original(shadeform, str(path))
        calls.append('delegated')
    except Exception as e:  # noqa: BLE001 - shadeform's setup needs credentials
        calls.append(type(e).__name__)
    assert calls, 'the wrapper swallowed the call for a non-Lyceum cloud'
    assert calls[0] != 'AssertionError', (
        'a non-Lyceum cloud now hits the `assert False` branch — the wrapper '
        'is intercepting clouds it must not touch')
