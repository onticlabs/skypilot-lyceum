"""Structural conformance to the provisioner contract SkyPilot checks at runtime.

`sky.provision._route_to_cloud_impl` does
`inspect.signature(func).bind(*args, **kwargs)` and then
`getattr(plugin_module, func.__name__)(*args, **kwargs)`. Nothing verifies that
our function accepts what the dispatcher forwards, so a parameter mismatch is a
TypeError raised in the middle of a real launch -- after the optimizer ran,
possibly after a VM was created and started billing. Several dispatcher call
sites pass by keyword (`get_cluster_info(..., region=..., cluster_name_on_cloud=
..., provider_config=...)` in cloud_vm_ray_backend.py, `state=` on
wait_instances, `worker_only=` on terminate_instances, `retry_if_missing=` on
query_instances), so parameter *names* are load-bearing, not just arity.

`sky.catalog._map_clouds_catalog` has the same shape one layer up, but that
contract is checked in `tests/test_catalog.py` (see section 4 below for why it
is not duplicated here).

This is cheap to check statically. None of these tests calls into the package,
so they stay meaningful once the bodies are implemented.
"""
from __future__ import annotations

import inspect

import pytest
from sky import provision as sky_provision

import skypilot_lyceum.provision as lyceum_provision

#: The nine names `sky.provision` routes to a cloud module.
PROVISION_FUNCTIONS = (
    'bootstrap_instances',
    'cleanup_ports',
    'get_cluster_info',
    'open_ports',
    'query_instances',
    'run_instances',
    'stop_instances',
    'terminate_instances',
    'wait_instances',
)

#: Parameters where we deliberately supply a default the dispatcher does not.
#:
#: The dispatcher declares `provider_config: Dict[str, Any]` with no default on
#: stop/terminate, but every in-tree provider (shadeform, hyperbolic, runpod)
#: writes `Optional[Dict[str, Any]] = None`. Adding a default cannot break
#: dispatch -- the caller always passes it positionally -- so we match the
#: in-tree convention rather than the dispatcher here, and pin the exception
#: list so that a *new* divergence still fails.
KNOWN_EXTRA_DEFAULTS = frozenset({
    ('stop_instances', 'provider_config'),
    ('terminate_instances', 'provider_config'),
})


def _dispatcher_parameters(name: str):
    """`sky.provision.<name>`'s parameters, minus the leading provider_name."""
    parameters = list(inspect.signature(getattr(sky_provision,
                                                name)).parameters.values())
    assert parameters and parameters[0].name == 'provider_name', (
        f'sky.provision.{name} no longer starts with provider_name: '
        f'{parameters!r} -- upstream dispatch contract changed')
    return parameters[1:]


# ---------------------------------------------------------------------------
# 1. The nine names exist and are callable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', PROVISION_FUNCTIONS)
def test_provision_package_exports_function(name):
    """A missing name silently falls through to the in-tree module lookup.

    `_route_to_cloud_impl` only uses `getattr(plugin_module, func.__name__,
    None)`; when that is None it tries `globals().get('lyceum')` in
    `sky.provision`, finds nothing, and ends in the decorator's default body --
    `raise NotImplementedError` with no hint about which of the nine is absent.
    """
    assert hasattr(lyceum_provision, name), (
        f'skypilot_lyceum.provision does not export {name!r}; exports are '
        f'{sorted(n for n in dir(lyceum_provision) if not n.startswith("_"))}')
    assert callable(getattr(lyceum_provision, name))


def test_provision_all_names_are_actually_importable():
    """`__all__` must describe the module, not a second hand-written list.

    Replaces an earlier `set(__all__) == set(PROVISION_FUNCTIONS)` check, which
    compared one literal against another literal and was proven inert: renaming
    a provisioner function left it green, because neither side of the
    comparison reads the module. `hasattr` is the assertion with teeth --
    `_route_to_cloud_impl` does `getattr(plugin_module, name)`, so a name in
    `__all__` that the module does not define is a promise the dispatcher will
    call in.

    The nine-names-are-present direction is covered by
    `test_provision_package_exports_function` above; this is the converse.
    """
    missing = sorted(n for n in lyceum_provision.__all__
                     if not hasattr(lyceum_provision, n))
    assert not missing, (
        f'skypilot_lyceum.provision.__all__ promises {missing} but the module '
        'does not define them')


# ---------------------------------------------------------------------------
# 2. Signature conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name', PROVISION_FUNCTIONS)
def test_provision_signature_matches_dispatcher(name):
    """Prevents a TypeError raised mid-launch by `_route_to_cloud_impl`.

    Compares our signature against `sky.provision.<name>` with the leading
    `provider_name` dropped, on parameter names, order and kind. Defaults are
    checked one-sidedly: where the dispatcher declares a default the value must
    match exactly (the dispatcher omits defaulted arguments from the forwarded
    call, so a differing or missing default changes behaviour or raises), while
    adding a default the dispatcher lacks is permitted and pinned separately in
    `test_no_unexpected_added_defaults`.

    Annotations are deliberately not compared: this package uses
    `from __future__ import annotations`, so ours are strings and upstream's are
    objects. Only the binding surface matters here.
    """
    expected = _dispatcher_parameters(name)
    actual = list(
        inspect.signature(getattr(lyceum_provision, name)).parameters.values())

    assert [p.name for p in actual] == [p.name for p in expected], (
        f'{name}: parameter names/order differ from sky.provision.{name} '
        f'(minus provider_name).\n'
        f'  dispatcher forwards: {[p.name for p in expected]}\n'
        f'  ours accepts:        {[p.name for p in actual]}')

    for ours, theirs in zip(actual, expected):
        assert ours.kind == theirs.kind, (
            f'{name}: parameter {ours.name!r} is {ours.kind}, dispatcher has '
            f'{theirs.kind}')
        if theirs.default is not inspect.Parameter.empty:
            assert ours.default == theirs.default, (
                f'{name}: default for {ours.name!r} is {ours.default!r}, '
                f'dispatcher has {theirs.default!r}; the dispatcher omits this '
                'argument when the caller does, so the values must agree')


@pytest.mark.parametrize('name', PROVISION_FUNCTIONS)
def test_no_unexpected_added_defaults(name):
    """Guards the one-sided default rule above from becoming a loophole.

    A default the dispatcher does not have is only acceptable where the in-tree
    providers all do the same (see KNOWN_EXTRA_DEFAULTS). Anywhere else it hides
    an argument the dispatcher is expected to supply.
    """
    expected = {p.name: p for p in _dispatcher_parameters(name)}
    actual = inspect.signature(getattr(lyceum_provision, name)).parameters

    added = {
        param_name for param_name, param in actual.items()
        if param.default is not inspect.Parameter.empty and
        param_name in expected and
        expected[param_name].default is inspect.Parameter.empty
    }
    allowed = {p for (fn, p) in KNOWN_EXTRA_DEFAULTS if fn == name}
    assert added == allowed, (
        f'{name}: defaults added on {sorted(added)}, expected exactly '
        f'{sorted(allowed)}')


# ---------------------------------------------------------------------------
# 3. The specific parameter that has already bitten us once
# ---------------------------------------------------------------------------
def test_query_instances_accepts_retry_if_missing():
    """The one parameter mismatch that has already been paid for in production.

    `sky.provision.query_instances` forwards `retry_if_missing` by keyword from
    `backend_utils._query_cluster_status_via_cloud_api`. SkyPilot 0.13.0's own
    in-tree Shadeform provider omits the parameter, so that call raises
    TypeError on every status refresh for a Shadeform cluster -- and because
    there is no hook for it, the only fix available from outside SkyPilot is to
    monkeypatch a corrected `query_instances` into the running API server at
    start-up. Matching the dispatcher here means this package never needs an
    equivalent patch.
    """
    dispatcher = inspect.signature(sky_provision.query_instances).parameters
    assert 'retry_if_missing' in dispatcher, (
        'upstream dropped retry_if_missing; this regression test needs '
        'revisiting')

    ours = inspect.signature(lyceum_provision.query_instances).parameters
    assert 'retry_if_missing' in ours, (
        'query_instances must accept retry_if_missing -- omitting it is the '
        'in-tree Shadeform bug this provider must not reproduce')
    assert ours['retry_if_missing'].default == dispatcher[
        'retry_if_missing'].default
    assert ours['retry_if_missing'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        'retry_if_missing is passed by keyword by backend_utils')


# ---------------------------------------------------------------------------
# 4. Catalog method surface -- NOT CHECKED HERE.
#
# `tests/test_catalog.py::test_catalog_implements_every_function_the_dispatcher_
# may_call` owns this check. It is the file to edit if the catalog surface
# changes.
#
# This file used to derive the required set as
# `shadeform_catalog ∩ hyperbolic_catalog`, which is provably too weak:
# `sky/catalog/shadeform_catalog.py` defines no `regions`, so the intersection
# silently omits it -- yet `sky.catalog.regions()` dispatches exactly that name
# (`sky/catalog/__init__.py:177`). Anything only one reference catalog
# implements falls out of an intersection, which is the wrong direction for a
# "required" floor.
#
# test_catalog.py derives it correctly instead: the names the dispatcher
# actually dispatches (parsed out of `_map_clouds_catalog(clouds, '<name>', ...)`
# call sites), intersected with what a reference catalog implements. That set is
# a superset of this one and is anchored on the caller rather than on two
# arbitrary callees. Two derivations of the same contract is one too many, and
# the weaker one was the one that would have passed while `regions` was missing.
# ---------------------------------------------------------------------------
