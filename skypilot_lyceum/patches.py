"""The unsupported seams, isolated and anchored.

Everything else this package needs is public SkyPilot 0.13 API. The three
patches below are not, so each one CHECKS its upstream anchor before touching
anything and raises `PatchDriftError` if the anchor has moved.

Two rules make monkeypatching a long-running server survivable, and both are
load-bearing here:

  * Check the anchor FIRST, mutate second. A patch that discovers the drift
    halfway through leaves the process in a state neither the patched nor the
    unpatched code path expects, which is far worse than not starting.
  * Fail loudly, never degrade to a warning. A server that boots "healthy",
    advertises this cloud, and then rejects every job for it is much harder to
    diagnose than one that refuses to boot with the reason on stderr.
"""
from __future__ import annotations

import importlib
import sys

from sky.skylet import constants

CLOUD_NAME = 'lyceum'

#: The name `sky.catalog._map_clouds_catalog` will try to import for us.
CATALOG_MODULE = f'sky.catalog.{CLOUD_NAME}_catalog'

#: Our own catalog implementation, injected under the name above.
_OUR_CATALOG_MODULE = f'{__name__.rsplit(".", 1)[0]}.catalog'

#: An IN-TREE cloud whose catalog module must still resolve through the exact
#: `sky.catalog.<cloud>_catalog` convention `patch_catalog_module` exploits.
#: Shadeform specifically because it is a long-lived in-tree cloud whose
#: catalog is an ordinary module of that name -- if it ever stops resolving,
#: the convention this patch exploits has been replaced and injecting our
#: module would be a silent no-op.
_CATALOG_ANCHOR_MODULE = 'sky.catalog.shadeform_catalog'

#: Names that must still be in ALL_CLOUDS for it to be the constant we think it
#: is. Deliberately a handful of long-lived clouds rather than the whole tuple:
#: upstream adds and removes providers routinely, and this patch does not care
#: which ones are present -- only that the constant is still the flat tuple of
#: cloud names that `schemas.py` builds the `cloud`/`infra` validators from.
_ALL_CLOUDS_ANCHOR_NAMES = frozenset({'aws', 'gcp', 'kubernetes', 'shadeform'})


class PatchDriftError(RuntimeError):
    """An upstream anchor no longer matches. Refuse to run rather than guess."""


def patch_all_clouds() -> None:
    """Append 'lyceum' to `sky.skylet.constants.ALL_CLOUDS`.

    Needed because the task-YAML JSON schema builds its `cloud`/`infra` regex
    from this tuple; without it BOTH `cloud: lyceum` and `infra: lyceum` are
    rejected client-side before any of our code runs. Idempotent.

    Anchor: ALL_CLOUDS is a tuple of str containing known in-tree clouds.
    """
    all_clouds = constants.ALL_CLOUDS

    # Anchor first, and completely: nothing below this block may mutate
    # anything, so a drifted upstream leaves the constant exactly as found.
    if not isinstance(all_clouds, tuple):
        raise PatchDriftError(
            'sky.skylet.constants.ALL_CLOUDS is no longer a tuple (got '
            f'{type(all_clouds).__name__}). skypilot-lyceum patches it by '
            'appending one name; refusing to guess at the new shape.')
    non_str = [name for name in all_clouds if not isinstance(name, str)]
    if non_str:
        raise PatchDriftError(
            'sky.skylet.constants.ALL_CLOUDS is no longer a tuple of str '
            f'(offending entries: {non_str!r}). Refusing to append to it.')
    missing = sorted(_ALL_CLOUDS_ANCHOR_NAMES - set(all_clouds))
    if missing:
        raise PatchDriftError(
            'sky.skylet.constants.ALL_CLOUDS no longer contains the in-tree '
            f'clouds {missing}, so it is probably not the constant the '
            'cloud/infra schema is built from any more. Refusing to patch it.')

    if CLOUD_NAME in all_clouds:
        # Already applied (or upstream has adopted us). Appending again would
        # duplicate the name in every schema enum built from this tuple.
        return
    constants.ALL_CLOUDS = all_clouds + (CLOUD_NAME,)


def patch_catalog_module() -> None:
    """Register `skypilot_lyceum.catalog` as `sky.catalog.lyceum_catalog`.

    Needed because `sky.catalog._map_clouds_catalog` resolves catalogs with a
    hardcoded `importlib.import_module(f'sky.catalog.{cloud}_catalog')` and
    there is no catalog registry. Idempotent.

    Anchor (must be CHECKABLE, not merely asserted in prose): an in-tree cloud's
    catalog must still resolve through the exact convention we are exploiting --
    `importlib.import_module('sky.catalog.shadeform_catalog')` succeeds. If
    upstream introduces a real catalog registry, or renames the module
    convention, that import changes shape and we refuse to run rather than
    inject a module nothing will ever look up.
    """
    # A genuine import attempt, not a string or attribute comparison: the point
    # is that the convention still RESOLVES, which is the property
    # `_map_clouds_catalog` depends on.
    try:
        importlib.import_module(_CATALOG_ANCHOR_MODULE)
    except ImportError as e:
        raise PatchDriftError(
            f'{_CATALOG_ANCHOR_MODULE} no longer imports ({e}), so SkyPilot no '
            'longer resolves catalogs by the sky.catalog.<cloud>_catalog '
            f'convention. Injecting {CATALOG_MODULE} would be a no-op nothing '
            'looks up; refusing.') from e

    our_catalog = importlib.import_module(_OUR_CATALOG_MODULE)
    if sys.modules.get(CATALOG_MODULE) is our_catalog:
        # Already applied. Re-assigning would be harmless, but returning here
        # keeps "idempotent" a property of the code and not of luck.
        return
    sys.modules[CATALOG_MODULE] = our_catalog


def patch_cluster_auth() -> None:
    """Route Lyceum through SkyPilot's generic SSH-key substitution.

    `sky.backends.backend_utils._add_auth_to_cluster_config` is an isinstance
    chain over in-tree cloud classes that ends in `assert False, cloud`. Unlike
    the cluster-config template — which upstream DOES expose a hook for
    (`Provisioner.template_override`) — there is no extension point here at all.
    A real `sky.launch(infra='lyceum')` dies with a bare `AssertionError: Lyceum`
    at that line, after the optimizer has already chosen the resource, so the
    failure looks like it comes from nowhere.

    Lyceum needs exactly the GENERIC branch, `authentication.configure_ssh_info`:
    it substitutes `skypilot:ssh_public_key_content` and `skypilot:ssh_user` into
    the rendered config, which is what `templates/lyceum-ray.yml.j2` contains.
    Shadeform needs a bespoke variant only because it registers the key with the
    account and gets an id back; Lyceum takes the public key inline on every
    create and has no key registry (that is why our template has no
    `ssh_key_id`).

    Anchors, both checked before anything is wrapped:
      * the function still exists, and
      * it still contains the `assert False` fallthrough — i.e. upstream has
        not grown a default branch or a hook that would make this unnecessary.
    Idempotent.
    """
    # pylint: disable=import-outside-toplevel
    import inspect

    from sky import authentication
    from sky.backends import backend_utils
    from sky.utils import yaml_utils

    original = getattr(backend_utils, '_add_auth_to_cluster_config', None)
    if original is None:
        raise PatchDriftError(
            'sky.backends.backend_utils._add_auth_to_cluster_config no longer '
            'exists. Lyceum launches need SkyPilot to substitute the SSH key '
            'into the rendered cluster config; refusing to guess where that '
            'moved to.')
    if getattr(original, '_lyceum_patched', False):
        return
    if not hasattr(authentication, 'configure_ssh_info'):
        raise PatchDriftError(
            'sky.authentication.configure_ssh_info is gone — that is the '
            'generic SSH-key substitution Lyceum relies on. Refusing to patch.')
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):  # pragma: no cover - source always available
        source = ''
    if 'assert False' in source:
        pass  # The gap we are filling is still there, as expected.
    else:
        raise PatchDriftError(
            '_add_auth_to_cluster_config no longer falls through to '
            '`assert False` for unknown clouds. It may now handle out-of-tree '
            'clouds natively, in which case this patch should be DELETED '
            'rather than silently kept.')

    def patched(cloud, tmp_yaml_path):
        # Only Lyceum is intercepted. This function is on every cloud's launch
        # path, and Shadeform is carrying the real work today.
        if type(cloud).__name__ == 'Lyceum' and type(cloud).__module__.startswith(
                'skypilot_lyceum'):
            config = yaml_utils.read_yaml(tmp_yaml_path)
            config = authentication.configure_ssh_info(config)
            yaml_utils.dump_yaml(tmp_yaml_path, config)
            return
        return original(cloud, tmp_yaml_path)

    patched._lyceum_patched = True
    # Keep the original reachable. Without it, anything inspecting this function
    # after `enable()` sees OUR wrapper -- so the drift anchor above becomes
    # un-checkable the moment the patch is applied, which is precisely when you
    # still want to be able to check it.
    patched._lyceum_original = original
    backend_utils._add_auth_to_cluster_config = patched


def apply() -> None:
    """Apply all three patches. Safe to call repeatedly."""
    patch_all_clouds()
    patch_catalog_module()
    patch_cluster_auth()
