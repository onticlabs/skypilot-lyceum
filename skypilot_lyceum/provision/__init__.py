"""Lyceum provisioner.

Re-exports exactly the nine names `sky.provision` dispatches to, matching the
shape of `sky/provision/shadeform/__init__.py`, plus the `template_override`
hook that makes our shipped cluster template reachable at all.
"""
import os
import typing
from typing import Any, Dict, Optional

import sky.provision as _sky_provision

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky import task as task_lib

from skypilot_lyceum import node_autodown
from skypilot_lyceum.provision.config import bootstrap_instances
from skypilot_lyceum.provision.instance import cleanup_ports
from skypilot_lyceum.provision.instance import get_cluster_info
from skypilot_lyceum.provision.instance import open_ports
from skypilot_lyceum.provision.instance import query_instances
from skypilot_lyceum.provision.instance import run_instances
from skypilot_lyceum.provision.instance import stop_instances
from skypilot_lyceum.provision.instance import terminate_instances
from skypilot_lyceum.provision.instance import wait_instances

#: Absolute path to the cluster config template shipped inside this package.
#: Built from `__file__` so it stays correct under `pip install`, where the
#: repo layout is gone -- a hardcoded repo path works in a source checkout on a
#: laptop and fails on the deployed API server, which is the only place it
#: matters.
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                             'templates', 'lyceum-ray.yml.j2')
TEMPLATE_PATH = os.path.normpath(TEMPLATE_PATH)


def template_override(
    task: 'task_lib.Task',
    to_provision: 'resources_lib.Resources',
    *,
    _extra_launch_context: Optional[Dict[str, Any]] = None,
    _is_launched_by_jobs_controller: bool = False,
) -> Optional[_sky_provision.TemplateSpec]:
    """Point the backend at our packaged Jinja template.

    NOT optional wiring. SkyPilot's default resolution is
    `cloud_vm_ray_backend._get_cluster_config_template`, which ends in
    `return cloud_to_template[type(cloud)]` -- a bare dict lookup over in-tree
    cloud classes. An out-of-tree cloud is a KeyError on EVERY launch. This hook
    (consulted at `cloud_vm_ray_backend.py:1121`) is the only escape, and
    `TemplateSpec`'s docstring names an absolute path as the plugin-shipped case.

    Keyword defaults are ours, not the protocol's: `TemplateOverrideFn` declares
    both keywords as required, and the backend always passes them. Defaulting
    them costs nothing and keeps the hook callable from a test or a REPL.

    The extra variables carry node-side autodown onto the machine (the wheel
    mount and its install command). They come from here rather than from
    `Lyceum.make_deploy_resources_variables` because they describe the SERVER's
    filesystem, not the resource being provisioned -- `make_deploy_resources_variables`
    is a pure function of the Resources and must stay that way.
    """
    del task, to_provision, _extra_launch_context, _is_launched_by_jobs_controller
    return _sky_provision.TemplateSpec(
        template_path=TEMPLATE_PATH,
        variables=node_autodown.template_variables())


__all__ = [
    'bootstrap_instances', 'cleanup_ports', 'get_cluster_info', 'open_ports',
    'query_instances', 'run_instances', 'stop_instances',
    'template_override', 'terminate_instances', 'wait_instances',
]
