"""Lyceum cloud provider plugin for SkyPilot 0.13.0.

Usage:
    import skypilot_lyceum; skypilot_lyceum.enable()

Server-side (the SkyPilot API server) this is driven by `plugin.LyceumPlugin`
via `~/.sky/plugins.yaml`. Client-side (a laptop, or a CLI wrapping the SkyPilot
SDK) it is an explicit `enable(client_only=True)`: the client needs only the
cloud NAME and class to parse `infra: lyceum`; the catalog, optimizer and
credentials are all server-side, so no Lyceum API key ever reaches a laptop.
"""
from __future__ import annotations

import importlib
import sys

__version__ = '0.1.0.dev0'

CLOUD_NAME = 'lyceum'

_CLOUD_MODULE = f'{__name__}.cloud'
_PROVISION_MODULE = f'{__name__}.provision'


def enable(client_only: bool = False) -> None:
    """Register the Lyceum cloud with the ambient SkyPilot.

    Applies the three anchored patches, imports the Cloud subclass (which
    registers itself via the CLOUD_REGISTRY decorator), and -- unless
    `client_only` -- registers the provisioner.

    `client_only` means exactly one thing: no provisioner. Both patches still
    apply, including the catalog-module injection. Skipping the latter would be
    strictly worse than skipping both: `sky.catalog._map_clouds_catalog` with
    `clouds=None` sweeps every name in `ALL_CLOUDS`, so 'lyceum' present in
    ALL_CLOUDS without a `sky.catalog.lyceum_catalog` module breaks catalog
    lookups for EVERY cloud on the client. Injecting it is free: our catalog
    module reads no credentials at import.

    Idempotent: repeated calls in one process are a no-op. This is a real path,
    not a nicety -- the API server loads plugins once per process context, and
    `_Registry.register` asserts the name is not already present.

    Raises:
        patches.PatchDriftError: an upstream anchor moved. Deliberately NOT
            downgraded to a warning: a server that boots "healthy" but rejects
            every Lyceum job is worse than one that refuses to boot.
    """
    # Imported here rather than at module scope so that importing this package
    # (which the plugin loader does before it can even read `__version__`) does
    # not drag in half of SkyPilot.
    # pylint: disable=import-outside-toplevel
    from skypilot_lyceum import patches

    patches.apply()
    _register_cloud()
    if not client_only:
        _register_provisioner()


def _register_cloud() -> None:
    """Import `cloud.py`; its `@CLOUD_REGISTRY.register` decorator does the work.

    The registration is a side effect of module execution, so "already enabled"
    is observable as "module imported AND name in the registry" -- no private
    flag, which would go stale the moment anything reloads or evicts us.
    """
    # pylint: disable=import-outside-toplevel
    from sky.utils import registry

    already_registered = CLOUD_NAME in registry.CLOUD_REGISTRY
    if already_registered and _CLOUD_MODULE in sys.modules:
        return
    if already_registered:
        # Registered by an earlier incarnation of `cloud.py` that has since been
        # dropped from `sys.modules` (importlib.reload, a test harness). Running
        # the decorator again would trip its `assert name not in self`, so drop
        # the stale entry and let the fresh import own the name.
        _unregister_cloud()

    importlib.import_module(_CLOUD_MODULE)

    if CLOUD_NAME not in registry.CLOUD_REGISTRY:
        raise RuntimeError(
            f'importing {_CLOUD_MODULE} did not register {CLOUD_NAME!r} in '
            'sky.utils.registry.CLOUD_REGISTRY -- the @CLOUD_REGISTRY.register '
            'decorator on the Lyceum class is missing or registers under a '
            'different name.')


def _unregister_cloud() -> None:
    """Drop a stale registry entry (and its aliases) for our cloud name."""
    # pylint: disable=import-outside-toplevel,protected-access
    from sky.utils import registry

    reg = registry.CLOUD_REGISTRY
    reg.pop(CLOUD_NAME, None)
    aliases = getattr(reg, '_aliases', None)
    if isinstance(aliases, dict):
        for alias, target in list(aliases.items()):
            if target == CLOUD_NAME:
                del aliases[alias]


def _register_provisioner() -> None:
    """Point `sky.provision`'s dispatcher at our provision package.

    Server-side only. `register_provisioner` is last-registration-wins, so it is
    already idempotent; the guard exists so a repeat `enable()` does not swap in
    a fresh `Provisioner` object that callers holding the old one would miss.
    """
    # pylint: disable=import-outside-toplevel
    from sky import provision as sky_provision

    module = importlib.import_module(_PROVISION_MODULE)
    registered = sky_provision.get_registered_provisioner(CLOUD_NAME)
    if registered is not None and registered.module is module:
        return
    sky_provision.register_provisioner(
        CLOUD_NAME,
        module,
        # The cluster-config template we ship (`templates/lyceum-ray.yml.j2`)
        # reaches the backend through this hook. Read off the provision package
        # rather than imported by name so the hook stays that package's
        # business: it is the half that knows what the template needs.
        template_override=getattr(module, 'template_override', None),
    )
