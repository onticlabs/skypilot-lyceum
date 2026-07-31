"""The shipped Jinja template must actually reach the backend.

This file exists because the first 363 tests missed it. `test_cloud_class.py`
checks that `make_deploy_resources_variables` supplies every variable the
template references -- but nothing checked that SkyPilot ever *reads* our
template. It does not, by default: `cloud_vm_ray_backend._get_cluster_config_template`
ends in `return cloud_to_template[type(cloud)]`, a bare dict lookup over
in-tree clouds only, so an out-of-tree cloud raises KeyError on every launch.

The only escape is `Provisioner.template_override`, consulted at
`cloud_vm_ray_backend.py:1121` and documented in `sky/provision/__init__.py` as
supporting an ABSOLUTE `template_path` for exactly this "plugin-shipped
template" case.

A gap here is invisible to every other test in the suite and would surface as a
KeyError at the first real `sky launch`.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest
import sky.provision as provision_lib
import skypilot_lyceum
from sky.backends import cloud_vm_ray_backend
from skypilot_lyceum import provision as lyceum_provision


@pytest.fixture
def enabled(lyceum_enabled):
    """Register the cloud + provisioner, as the API server would.

    Delegates to conftest's `lyceum_enabled`, which snapshots and restores every
    global `enable()` mutates. An earlier version of this fixture just called
    `enable()` and leaked -- caught by `test_plugin.py`'s leak detector under a
    reversed file order, which is precisely why that detector exists.
    """
    yield lyceum_enabled


def test_upstream_has_no_template_for_lyceum(enabled):
    """The override is load-bearing, not belt-and-braces.

    Pins WHY `template_override` must exist: SkyPilot's default resolution is a
    dict keyed by in-tree cloud classes and raises on anything else. If upstream
    ever grows a registry and this test starts failing, the override may become
    optional -- but until then, removing it breaks every launch.
    """
    from sky.utils import registry

    lyceum = registry.CLOUD_REGISTRY.from_str('lyceum')
    with pytest.raises(KeyError):
        cloud_vm_ray_backend._get_cluster_config_template(lyceum)


def test_provision_package_exposes_template_override(enabled):
    """`register_provisioner` only picks the hook up if the module defines it.

    `enable()` wires `template_override=getattr(module, 'template_override',
    None)`. A missing attribute is not an error -- it silently yields None, and
    the backend then falls through to the KeyError above.
    """
    assert hasattr(lyceum_provision, 'template_override')
    assert callable(lyceum_provision.template_override)

    registered = provision_lib.get_registered_provisioner('lyceum')
    assert registered is not None
    assert registered.template_override is not None


def test_template_override_matches_the_backend_call_signature(enabled):
    """The backend calls it with two positionals and two private keywords.

    `cloud_vm_ray_backend.py:1121-1129` passes `_extra_launch_context` and
    `_is_launched_by_jobs_controller` by keyword. A signature that omits them is
    a TypeError in the middle of a launch -- the same failure mode
    `test_signature_conformance.py` guards for the provisioner functions.
    """
    sig = inspect.signature(lyceum_provision.template_override)
    sig.bind(object(), object(), _extra_launch_context=None,
             _is_launched_by_jobs_controller=False)


def test_template_override_returns_an_absolute_path_that_exists(enabled):
    """An absolute path is the documented contract for a plugin template.

    `sky/provision/__init__.py`'s TemplateSpec docstring: the path is "either an
    absolute path (for plugin-shipped templates) or a bare filename relative to
    sky/templates/". A bare filename would be looked up inside SkyPilot's own
    package, where our template does not live.
    """
    spec = lyceum_provision.template_override(
        object(), object(), _extra_launch_context=None,
        _is_launched_by_jobs_controller=False)

    assert isinstance(spec, provision_lib.TemplateSpec)
    path = pathlib.Path(spec.template_path)
    assert path.is_absolute(), f'{path} must be absolute, not package-relative'
    assert path.is_file(), f'{path} does not exist'
    assert path.name == 'lyceum-ray.yml.j2'


def test_template_override_points_at_the_packaged_template(enabled):
    """It must resolve inside the installed package, not a source checkout.

    Building the path from `__file__` keeps it correct under `pip install`,
    where the repo layout is gone. A hardcoded repo path works in a source
    checkout on a laptop and fails on the deployed API server, which is the only
    place it matters.
    """
    spec = lyceum_provision.template_override(
        object(), object(), _extra_launch_context=None,
        _is_launched_by_jobs_controller=False)

    packaged = (pathlib.Path(skypilot_lyceum.__file__).parent / 'templates' /
                'lyceum-ray.yml.j2')
    assert pathlib.Path(spec.template_path).resolve() == packaged.resolve()


def test_template_override_declares_the_provider_module(enabled):
    """The rendered config's `provider.module` must name OUR package.

    The template writes `module: skypilot_lyceum.provision`, which is what
    SkyPilot's external-provider path imports to reach our `run_instances`.
    Pointing at `sky.provision.shadeform` (an easy copy-paste) would provision
    the wrong cloud entirely.
    """
    spec = lyceum_provision.template_override(
        object(), object(), _extra_launch_context=None,
        _is_launched_by_jobs_controller=False)
    text = pathlib.Path(spec.template_path).read_text()

    # Match the DIRECTIVE, not the file. A whole-file substring search for
    # 'shadeform' would trip on the header comment recording which template
    # this was adapted from -- pinning prose rather than behaviour, which is
    # the over-fitting this suite's review flagged elsewhere.
    modules = re.findall(r'^\s*module:\s*(\S+)\s*$', text, re.MULTILINE)
    assert modules == ['skypilot_lyceum.provision'], (
        f'provider.module must name our package exactly once, got {modules}')
