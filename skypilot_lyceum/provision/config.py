"""Bootstrap hook. Lyceum needs no pre-provision cloud setup (no VPC, no
security groups, no SSH-key registry), so this is a pass-through."""
from __future__ import annotations

from sky.provision import common


def bootstrap_instances(region: str, cluster_name_on_cloud: str,
                        config: common.ProvisionConfig) -> common.ProvisionConfig:
    """Pass-through.

    NOTE the parameter name. `sky.provision.bootstrap_instances` declares
    `cluster_name_on_cloud`, and `sky/provision/provisioner.py:74` really does
    pass `cluster_name.name_on_cloud`. Every in-tree provider (shadeform,
    hyperbolic, runpod) calls it `cluster_name`, which is a misnomer -- dispatch
    is positional so nothing breaks, but the name lies about the value. We match
    the dispatcher instead of copying the bug.

    Returned unchanged, not rebuilt: the result is what `run_instances` is then
    called with, so dropping or rewriting a field here silently changes what
    gets provisioned.
    """
    del region, cluster_name_on_cloud
    return config
