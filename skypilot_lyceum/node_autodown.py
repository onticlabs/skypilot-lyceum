"""Let a Lyceum node delete itself when it goes idle.

SkyPilot's autodown is executed by the skylet ON the node: when the cluster has
been idle past its threshold, `StopEvent._stop_cluster` looks up the cloud and
dispatches `terminate_instances`. For an out-of-tree cloud that lookup finds
nothing, because a provisioned node gets a stock SkyPilot install and the skylet
never calls `plugins.load_plugins()`.

Lyceum makes that expensive rather than merely untidy: there is no stop and no
cloud-side TTL, so a VM bills from `ready` until someone issues DELETE.

Two halves, both here:

  * `template_variables()` runs on the API SERVER at launch time. It mounts this
    package's wheel onto the node and emits a setup command that installs it and
    drops a `.pth` file into SkyPilot's runtime environment.
  * `bootstrap()` runs on the NODE, at every interpreter start, because that is
    what a `.pth` does. It registers the cloud and provisioner -- but only
    inside the skylet, which is the one process that needs them.

A `.pth` is an unusual thing to ship, and it is here for one reason: it is the
only place code can run inside the skylet process without a SkyPilot hook, and
there is no such hook in 0.13.0. If upstream ever loads plugins in the skylet,
this whole module collapses to a normal `enable()` call.
"""
from __future__ import annotations

import os
import pathlib
import shlex
from typing import Dict, List, Optional

#: Where the wheel lands on the node. Under `~/.sky` so it sits beside the rest
#: of SkyPilot's per-node state rather than in the user's working tree.
REMOTE_WHEEL_DIR = '~/.sky/lyceum'

#: The single line written into the runtime's site-packages. `site.py` executes
#: a .pth line ONLY when it starts with `import `; every other line is treated
#: as a directory to add to sys.path. A typo here is silent and total.
PTH_LINE = 'import skypilot_lyceum.node_autodown as _l; _l.bootstrap()'

#: Filename chosen to sort last, so anything else the runtime relies on is
#: already on sys.path by the time we import.
PTH_NAME = 'zz-skypilot-lyceum.pth'

#: Overridable so the image can build the wheel wherever it likes. Set by the
#: API server's Dockerfile; absent on a laptop, where node-side autodown is not
#: a thing that can happen anyway.
WHEEL_DIR_ENV = 'SKYPILOT_LYCEUM_WHEEL_DIR'

_DEFAULT_WHEEL_DIR = '/opt/lyceum-wheels'

#: Evidence on the node that the bootstrap ran, and what it decided. The skylet
#: log is the natural place, but the .pth executes before logging is configured,
#: so a file is the only channel that always exists.
_MARKER = pathlib.Path('~/.sky/lyceum-node-bootstrap').expanduser()


# --------------------------------------------------------------------------
# Server side: getting this package onto the node
# --------------------------------------------------------------------------
def find_wheel() -> Optional[pathlib.Path]:
    """The wheel to install on the node, or None if this server has none.

    None means this server ships no node autodown at all, which is a build
    configuration rather than a runtime failure -- a laptop, or an image built
    without the wheel step. It renders no mount and no setup command, so nothing
    is installed and nothing is verified.

    Note the asymmetry with `_install_command`: NOT shipping autodown is a
    deployment choice, while shipping it and having it not work is a defect, and
    only the second fails a launch. The API server's Dockerfile asserts the
    wheel is present, so on the control plane this cannot silently be None.
    """
    directory = pathlib.Path(os.environ.get(WHEEL_DIR_ENV, _DEFAULT_WHEEL_DIR))
    if not directory.is_dir():
        return None
    wheels = sorted(directory.glob('skypilot_lyceum-*.whl'))
    if not wheels:
        return None
    if len(wheels) > 1:
        # The image builds exactly one. More than one means the build changed
        # and nothing here can say which is intended -- a lexicographic "latest"
        # would pick 0.9.0 over 0.10.0 and ship the wrong plugin to every node.
        raise RuntimeError(
            f'{directory} holds {len(wheels)} skypilot_lyceum wheels '
            f'({[w.name for w in wheels]}); expected exactly one, so the node '
            'would get an arbitrary build. Fix the image.')
    return wheels[0]


def template_variables() -> Dict[str, object]:
    """Extra variables for the cluster template, via `TemplateSpec.variables`.

    Returns a file-mount mapping and a setup command. Both are empty when no
    wheel is available, which renders to nothing in the template.
    """
    wheel = find_wheel()
    if wheel is None:
        return {'lyceum_file_mounts': {}, 'lyceum_node_setup_command': ''}
    remote = f'{REMOTE_WHEEL_DIR}/{wheel.name}'
    return {
        'lyceum_file_mounts': {remote: str(wheel)},
        'lyceum_node_setup_command': _install_command(remote),
    }


def _install_command(remote_wheel: str) -> str:
    """Install the plugin into SkyPilot's runtime env, arm the .pth, and PROVE it.

    Fails the setup step, and so the whole launch, if any of that does not work.
    That is the deliberate trade: a node that cannot delete itself is worse than
    a launch that did not happen. Lyceum has no stop and no cloud-side TTL, so
    such a node bills from `ready` until a human notices -- unbounded in time,
    up to $63.92/h.

    What failing here does NOT do is delete the VM. SkyPilot's teardown-on-error
    wraps provisioning, not runtime setup, so the machine is left running as a
    visible INIT cluster. On the ontic path `launch._reconcile_orphan` then sees
    it and calls `sky down`; off that path a human must. So the honest claim is
    that this converts an INVISIBLE, unbounded leak -- a node that looks healthy
    and can never delete itself -- into a loud, visible, usually self-cleaning
    one. Not that the node never exists.

    The last step is the point of the whole thing. `pip install` exiting 0 says a
    file was copied; it does not say the skylet will be able to resolve the cloud
    when it tries to tear this node down half a day later. So we run the exact
    registration the skylet's autodown path depends on, in the exact interpreter
    that will run it, and require the two lookups `StopEvent._stop_cluster` makes
    to succeed. Verifying the capability rather than the artifact is what makes
    "loud" mean anything.

    `--no-deps` is deliberate. Our declared dependencies are `requests`, `pandas`
    and `skypilot` itself; the first two are already SkyPilot dependencies and so
    present, and resolving them again could upgrade versions SkyPilot pinned
    inside its own runtime. Installing `skypilot` over the runtime's own install
    would be worse still.

    Contains no `": "` and no `" #"`, and must not. The setup block is a plain
    YAML scalar: a colon-space starts a mapping and makes the whole cluster
    config unparseable, and a mid-line hash silently TRUNCATES the command --
    the second is worse, because the launch then succeeds with autodown quietly
    missing. `test_node_autodown.py` pins both.
    """
    # Same python discovery SkyPilot uses for its own remote commands
    # (`skylet.constants.SKY_PYTHON_CMD`), so we install into, and verify
    # against, the interpreter that will actually run the skylet.
    py = ('$([ -s ${SKY_RUNTIME_DIR:-$HOME}/.sky/python_path ] && '
          'cat ${SKY_RUNTIME_DIR:-$HOME}/.sky/python_path 2> /dev/null || '
          'which python3)')
    pth = shlex.quote(PTH_LINE)
    verify = shlex.quote(VERIFY_SNIPPET)
    return (
        f'{py} -m pip install --no-deps --quiet {remote_wheel} && '
        f'SP=$({py} -c "import site; print(site.getsitepackages()[0])") && '
        f'echo {pth} > "$SP/{PTH_NAME}" && '
        f'{py} -c {verify};')


#: Run on the node at setup time, in the runtime interpreter, to prove autodown
#: will work. Mirrors what `StopEvent._stop_cluster` does: register, then resolve
#: the cloud and the provisioner. A failure here fails the launch.
VERIFY_SNIPPET = (
    'import os, site; '
    'from skypilot_lyceum import node_autodown; '
    'node_autodown._register(); '
    'from sky.utils import registry; '
    'import sky.provision as p; '
    'assert registry.CLOUD_REGISTRY.from_str("lyceum") is not None, '
    '"lyceum did not register in the node runtime"; '
    'assert p.get_registered_provisioner("lyceum") is not None, '
    '"no lyceum provisioner in the node runtime"; '
    'assert os.path.exists(os.path.join(site.getsitepackages()[0], '
    f'"{PTH_NAME}")), "pth missing"; '
    'print("lyceum node autodown verified")'
)


# --------------------------------------------------------------------------
# Node side: registering inside the skylet
# --------------------------------------------------------------------------
def _process_cmdline() -> List[str]:
    """This process's argv. Read from /proc rather than `sys.argv` because the
    .pth runs during interpreter startup, where argv is not yet meaningful for
    a `-m` invocation."""
    try:
        with open('/proc/self/cmdline', 'rb') as handle:
            return handle.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return []


def _inside_skylet() -> bool:
    """Exactly SkyPilot's own test (`skylet/attempt_skylet.py`): the skylet is
    the process whose command line names `sky.skylet.skylet`. Matching it means
    we cannot drift from what SkyPilot considers the skylet.

    Containment, not `endswith`, because that is what upstream does. The two
    agree on today's launch line, but they diverge the moment SkyPilot glues
    arguments into one argv element or launches by script path -- upstream would
    still find its skylet and we would silently stop registering, which is the
    exact drift this is supposed to be immune to.

    `attempt_skylet` -- the launcher -- deliberately does NOT match: it exits
    immediately after starting the real skylet, so registering there is waste.
    """
    return any('sky.skylet.skylet' in arg for arg in _process_cmdline())


def _register() -> None:
    """Register the cloud and the provisioner in this process.

    Deliberately NOT `enable()`. `enable()` also applies the anchored patches,
    two of which are a matched pair: `patch_all_clouds` puts 'lyceum' into
    ALL_CLOUDS, which makes catalog sweeps look for a `sky.catalog.lyceum_catalog`
    module that `patch_catalog_module` then supplies. On the node we need
    neither -- nothing sweeps catalogs there -- and skipping both together keeps
    that invariant. The third patch is launch-path only.

    What the skylet needs is precisely the two lookups in
    `StopEvent._stop_cluster`: a CLOUD_REGISTRY entry, and a registered
    provisioner to dispatch `terminate_instances` through.
    """
    # pylint: disable=import-outside-toplevel
    from skypilot_lyceum import _register_cloud, _register_provisioner

    _register_cloud()
    _register_provisioner()


def bootstrap() -> None:
    """Entry point for the .pth. Never raises.

    An exception escaping into `site.py` is printed and swallowed, so it would
    leave a node that looks configured and cannot delete itself -- the exact
    silent failure this module exists to end. Contain it and write down what
    happened, so `cat ~/.sky/lyceum-node-bootstrap` answers the question on any
    node in one command.
    """
    try:
        if not _inside_skylet():
            return
        _register()
    except BaseException as exc:  # noqa: BLE001 - see docstring
        _note(f'failed: {exc!r}')
    else:
        _note('ok')


def _note(text: str) -> None:
    try:
        _MARKER.parent.mkdir(parents=True, exist_ok=True)
        _MARKER.write_text(text)
    except OSError:
        pass          # evidence is a nicety; never let it break the skylet
