"""Tests for `skypilot_lyceum.plugin.LyceumPlugin`.

This is how the package is loaded in PRODUCTION: a deployed SkyPilot API
server enables Lyceum through `~/.sky/plugins.yaml`, not through an explicit
`import skypilot_lyceum; enable()` -- that spelling is the client half only. So
every server-side capability (catalog, optimizer, provisioner, credentials)
reaches SkyPilot exclusively through `LyceumPlugin.install`. An `install` that
silently does nothing produces an API server that starts clean, reports healthy,
serves the dashboard -- and rejects every `infra: lyceum` job with
`ValueError: Cloud 'lyceum' is not a valid cloud`.

Harness modelled on skypilot 0.13.0's own
`tests/unit_tests/test_sky/server/test_plugins.py::test_load_plugins_registers_
and_installs`: build the config, write `plugins.yaml` to tmp_path, point the
config env var at it, call `plugins.load_plugins(ctx)`, assert `install()` ran.
Deliberately driven through `load_plugins` rather than by calling
`plugin.install(ctx)` directly, because the real failure modes here are the ones
`load_plugins` owns -- an unimportable module path, a class that is not a
`BasePlugin` subclass, a `load_contexts` that excludes the process actually
doing the work.

LOCAL FIXTURES: the global-state isolation below is the same discipline as
`tests/test_registration.py` (that file's docstring explains why it is
load-bearing). If a third module needs it, lift it verbatim.
"""
from __future__ import annotations

import importlib
import sys

import pytest
import yaml
from sky import provision as sky_provision
from sky.server import plugins
from sky.skylet import constants
from sky.utils import registry

PKG = 'skypilot_lyceum'
CLOUD_NAME = 'lyceum'
CATALOG_MODULE = f'sky.catalog.{CLOUD_NAME}_catalog'
PLUGIN_CLASS_PATH = f'{PKG}.plugin.LyceumPlugin'

_MISSING = object()


# ---------------------------------------------------------------------------
# Global-state isolation (mirrors tests/test_registration.py)
# ---------------------------------------------------------------------------
def _lyceum_module_names():
    return [
        name for name in list(sys.modules)
        if name == PKG or name.startswith(f'{PKG}.')
    ]


def _evict_lyceum_modules():
    """Drop our package so a later import re-runs the register decorator."""
    for name in _lyceum_module_names():
        del sys.modules[name]


def _clear_lyceum_state():
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
    """Fail this module if it leaks Lyceum state into the rest of the run."""
    baseline = _global_fingerprint()
    yield
    after = _global_fingerprint()
    assert after == baseline, (
        'test_plugin leaked global SkyPilot state into the rest of the suite.\n'
        f'  before: {baseline}\n  after:  {after}')


@pytest.fixture(autouse=True)
def _lyceum_globals(monkeypatch):
    """Snapshot/restore every global `install()` touches, and start clean.

    Covers SkyPilot's plugin registry too (`plugins._PLUGINS` and
    `plugins._plugins_loaded` are module-level and `load_plugins` writes both),
    which upstream's own tests reset with the same `monkeypatch.setattr` idiom.
    """
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
    monkeypatch.setattr(plugins, '_PLUGINS', {})
    monkeypatch.setattr(plugins, '_plugins_loaded', False)
    monkeypatch.setattr(plugins, '_EXTENSION_CONTEXT', None)
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
def _plugin_cls():
    """The `LyceumPlugin` class from a freshly imported package."""
    return importlib.import_module(f'{PKG}.plugin').LyceumPlugin


def _write_plugins_yaml(tmp_path, monkeypatch, class_path=PLUGIN_CLASS_PATH):
    """Write a `plugins.yaml` and point SkyPilot's config env var at it."""
    config_path = tmp_path / 'plugins.yaml'
    config_path.write_text(
        yaml.safe_dump({'plugins': [{
            'class': class_path
        }]}))
    monkeypatch.setenv(plugins._PLUGINS_CONFIG_ENV_VAR,  # pylint: disable=protected-access
                       str(config_path))
    return config_path


def _load(tmp_path, monkeypatch, context=plugins.PluginContext.EXECUTOR):
    """Run SkyPilot's real plugin loader against our plugins.yaml."""
    _write_plugins_yaml(tmp_path, monkeypatch)
    ctx = plugins.ExtensionContext(context=context)
    plugins.load_plugins(ctx)
    return ctx


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------
def test_plugin_name_is_lyceum():
    """The name is what an operator sees when the server reports its plugins.

    `sky/server/server.py`'s `/api/plugins` endpoint and the dashboard version
    tooltip both read `plugin.name`; `BasePlugin.name` defaults to `None`, which
    renders as a blank row. When a Lyceum launch fails, "is the Lyceum plugin
    even loaded on this server?" is the first question, and this string is the
    only place the answer is visible.
    """
    assert _plugin_cls()().name == CLOUD_NAME


def test_plugin_is_a_skypilot_base_plugin():
    """`load_plugins` refuses anything else, with a TypeError at server start.

    `plugins.load_plugins` does `if not issubclass(plugin_cls, BasePlugin):
    raise TypeError(...)`, and `install` is an `abc.abstractmethod`, so a class
    that forgets it cannot even be instantiated. Both are start-time failures on
    the API server -- cheap to pin here instead.
    """
    cls = _plugin_cls()
    assert issubclass(cls, plugins.BasePlugin)
    assert cls(), 'LyceumPlugin() must be constructible with no parameters'


def test_plugin_version_is_reported_and_non_empty():
    """A blank version makes "which build is deployed?" unanswerable.

    `BasePlugin.version` defaults to `None`. The `/api/plugins` payload carries
    it verbatim, and it is how a Lyceum bug gets pinned to a specific installed
    build. It must also agree with the package's own `__version__`, or the two
    drift and the number in the dashboard stops meaning anything.
    """
    version = _plugin_cls()().version

    assert isinstance(version, str) and version.strip(), (
        f'LyceumPlugin.version must be a non-empty string; got {version!r}')
    assert version == importlib.import_module(PKG).__version__, (
        'the plugin version must be the package version, not a second '
        'hand-maintained string')


# ---------------------------------------------------------------------------
# 2. load_contexts
# ---------------------------------------------------------------------------
def test_plugin_declares_load_contexts_as_plugin_context_members():
    """A stray string in `load_contexts` disables the plugin silently.

    `BasePlugin.should_load` is `context in cls.load_contexts`, where `context`
    is a `PluginContext` enum member. `'executor' in frozenset({'executor'})` is
    True but `PluginContext.EXECUTOR in frozenset({'executor'})` is False, so a
    string-valued declaration makes `load_plugins` skip us with a debug-level
    log line and nothing else.
    """
    load_contexts = _plugin_cls().load_contexts

    assert isinstance(load_contexts, frozenset), (
        'load_contexts must be a frozenset (BasePlugin declares it as a '
        f'ClassVar[FrozenSet[PluginContext]]); got {type(load_contexts)}')
    assert load_contexts, 'an empty load_contexts disables the plugin entirely'
    unexpected = {c for c in load_contexts
                  if not isinstance(c, plugins.PluginContext)}
    assert not unexpected, (
        f'load_contexts contains non-PluginContext values {unexpected}; '
        '`context in load_contexts` will never match for those')


@pytest.mark.parametrize('context', [
    pytest.param(plugins.PluginContext.EXECUTOR, id='executor'),
    pytest.param(plugins.PluginContext.UVICORN, id='uvicorn'),
])
def test_plugin_loads_in_the_contexts_that_do_the_work(context):
    """The two contexts where excluding Lyceum breaks a real request.

    EXECUTOR is the request-executor subprocess that runs API request bodies --
    optimization and provisioning happen there (`sky/core.py`,
    `sky/execution.py`), so this is where the catalog, the optimizer and the
    provisioner are actually consulted. Skip it and every `sky launch` on Lyceum
    dies in the executor, after the request was accepted.

    UVICORN is the API-server worker that parses and schema-validates the
    submitted task YAML. Skip it and `infra: lyceum` is rejected at the door,
    before the executor ever sees it.

    Not asserted as an exact set: MAIN and CONTROLLER are defensible either way
    (CONTROLLER matters the day managed jobs target Lyceum), and
    `BasePlugin.load_contexts` defaults to all four, which satisfies this.
    """
    assert _plugin_cls().should_load(context), (
        f'LyceumPlugin opts out of {context.value}; load_plugins will skip it '
        'there with only a debug log')


# ---------------------------------------------------------------------------
# 3. install() actually registers the cloud
# ---------------------------------------------------------------------------
def test_load_plugins_installs_lyceum_plugin(tmp_path, monkeypatch):
    """The end-to-end production path, driven by SkyPilot's own loader.

    Everything from `~/.sky/plugins.yaml` to a registered cloud: config schema
    validation, `importlib.import_module('skypilot_lyceum.plugin')`, the
    `issubclass` check, the context filter, `plugin.install(ctx)`. A typo in the
    class path or a schema the config no longer satisfies fails here rather than
    at server boot.
    """
    ctx = _load(tmp_path, monkeypatch)

    loaded = plugins.get_plugins()
    assert len(loaded) == 1, (
        f'expected exactly one plugin loaded, got {loaded}')
    assert isinstance(loaded[0], _plugin_cls())
    assert loaded[0].name == CLOUD_NAME
    assert plugins.get_extension_context() is ctx


def test_install_registers_the_cloud(tmp_path, monkeypatch):
    """`install()` must call `enable()` -- a no-op install is the failure mode.

    This is the whole point of the plugin. Asserts the observable consequence
    (the cloud is registered and resolvable) rather than that some function was
    called, so an `install` that inlines the registration a different way still
    passes, and an `install` that merely logs still fails.
    """
    _load(tmp_path, monkeypatch)

    lyceum_cloud = sys.modules.get(f'{PKG}.cloud')
    assert lyceum_cloud is not None, (
        'install() did not import skypilot_lyceum.cloud, so the '
        '@CLOUD_REGISTRY.register decorator never ran')
    resolved = registry.CLOUD_REGISTRY.from_str(CLOUD_NAME)
    assert isinstance(resolved, lyceum_cloud.Lyceum)


def test_install_applies_the_full_server_side_enable(tmp_path, monkeypatch):
    """Server-side means all four seams, not just the cloud class.

    The API server is the half that owns the catalog, the optimizer and the
    provisioner, so `install()` must be the FULL `enable()`,
    not `enable(client_only=True)`. A client-only install would parse
    `infra: lyceum` on the server and then fail with `AssertionError: Unknown
    provider: lyceum` in the middle of a launch.
    """
    _load(tmp_path, monkeypatch)

    assert CLOUD_NAME in constants.ALL_CLOUDS
    assert CATALOG_MODULE in sys.modules
    assert sky_provision.get_registered_provisioner(CLOUD_NAME) is not None, (
        'install() must register the provisioner -- the API server is the side '
        'that provisions')


def test_install_is_idempotent_across_contexts(tmp_path, monkeypatch):
    """The API server loads plugins more than once per interpreter.

    `sky/server/plugins.py` is driven per process context, and with
    `--deploy=false` uvicorn runs in-process, giving the main process a second
    plugin load (see the `PluginContext.UVICORN` docstring upstream). Two
    installs must not trip `_Registry.register`'s
    `assert name not in self` or append 'lyceum' to ALL_CLOUDS twice.
    """
    _write_plugins_yaml(tmp_path, monkeypatch)

    plugins.load_plugins(
        plugins.ExtensionContext(context=plugins.PluginContext.MAIN))
    after_first = _global_fingerprint()

    plugins.load_plugins(
        plugins.ExtensionContext(context=plugins.PluginContext.UVICORN))

    assert _global_fingerprint() == after_first
    assert constants.ALL_CLOUDS.count(CLOUD_NAME) == 1


def test_install_propagates_patch_drift(tmp_path, monkeypatch):
    """A broken install must stop the server, not degrade it silently.

    `load_plugins` does not wrap `plugin.install(...)`, so an exception
    propagates and the API server fails to start -- the design's fail-loud rule
    (4.1). Pinned here because the tempting "defensive" fix is a try/except
    inside `install`, which would produce a server that boots clean, reports the
    Lyceum plugin as loaded, and rejects every Lyceum job.

    Drift is induced through the package's own documented failure -- ALL_CLOUDS
    reshaped so `patch_all_clouds` refuses -- rather than by monkeypatching
    `enable` itself, so the test does not depend on how `plugin.py` happens to
    import it. `tests/test_registration.py::test_enable_propagates_patch_drift`
    pins the same rule one layer down, at `enable()`.
    """
    _write_plugins_yaml(tmp_path, monkeypatch)
    patch_drift_error = importlib.import_module(f'{PKG}.patches').PatchDriftError
    monkeypatch.setattr(constants, 'ALL_CLOUDS', ['aws', 'gcp'], raising=True)

    with pytest.raises(patch_drift_error):
        plugins.load_plugins(
            plugins.ExtensionContext(context=plugins.PluginContext.EXECUTOR))
