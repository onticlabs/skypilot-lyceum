"""SkyPilot API-server plugin entry point (`~/.sky/plugins.yaml`)."""
from __future__ import annotations

from typing import ClassVar, FrozenSet

from sky.server import plugins

import skypilot_lyceum


class LyceumPlugin(plugins.BasePlugin):
    """Registers the Lyceum cloud in every server process context."""

    #: Every context, spelled out. `BasePlugin.should_load` is
    #: `context in cls.load_contexts` against `PluginContext` MEMBERS, so a
    #: frozenset of strings would silently disable the plugin.
    #:
    #: * UVICORN parses and schema-validates submitted task YAML -- without it
    #:   `infra: lyceum` is rejected at the door.
    #: * EXECUTOR runs the request bodies: catalog, optimizer, provisioner.
    #: * MAIN registers before main-process bootstrap consumes the registry.
    #: * CONTROLLER matters the day managed jobs target Lyceum.
    load_contexts: ClassVar[FrozenSet[plugins.PluginContext]] = frozenset({
        plugins.PluginContext.MAIN,
        plugins.PluginContext.UVICORN,
        plugins.PluginContext.EXECUTOR,
        plugins.PluginContext.CONTROLLER,
    })

    @property
    def name(self):
        return 'lyceum'

    @property
    def version(self):
        """The package version, never a second hand-maintained string.

        Surfaces in `/api/plugins` and the dashboard version tooltip; it is how
        a Lyceum bug gets pinned to a wheel.
        """
        return skypilot_lyceum.__version__

    def install(self, extension_context: 'plugins.ExtensionContext'):
        """The full server-side enable -- this side owns catalog + provisioner.

        NOT `client_only`: the API server is the half that optimizes and
        provisions, so a client-only install would parse `infra: lyceum` here
        and then die with `AssertionError: Unknown provider: lyceum` mid-launch.

        No try/except by design. `load_plugins` does not wrap `install`, so a
        `PatchDriftError` propagates and the server fails to start -- which is
        the point. Swallowing it produces a server that boots clean, reports
        this plugin as loaded, and rejects every Lyceum job.
        """
        del extension_context  # Nothing to register on the FastAPI app.
        skypilot_lyceum.enable()
