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

    None is a supported state, not an error: it degrades to the behaviour we
    already have (no node-side autodown, teardown left to the reaper). Making it
    fatal would turn a missing build artifact into an inability to launch.
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
    """Install the wheel into SkyPilot's runtime env and arm the .pth.

    `--no-deps` is deliberate. Our declared dependencies are `requests`,
    `pandas` and `skypilot` itself; the first two are already SkyPilot
    dependencies and therefore present, and resolving them again could upgrade
    versions SkyPilot pinned inside its own runtime. Installing `skypilot` from
    PyPI over the runtime's own install would be worse still.

    Contains no `": "` anywhere, and must not. The setup block is a plain YAML
    scalar, where a colon-followed-by-space starts a mapping and makes the whole
    cluster config unparseable -- for every launch, not just Lyceum ones. That
    is what broke the first attempt at this; `test_node_autodown.py` now pins it.

    The whole thing ends in `|| true`: a node that cannot install the plugin
    must still run its job. Failing the setup step would trade a cost bug for an
    outage. The failure is not silent -- it prints, and `_MARKER` on the node
    records that the bootstrap never ran.
    """
    # Same python discovery SkyPilot uses for its own remote commands
    # (`skylet.constants.SKY_PYTHON_CMD`), so we install into the interpreter
    # that will actually run the skylet rather than whatever `python3` resolves
    # to on the vendor image.
    py = ('$([ -s ${SKY_RUNTIME_DIR:-$HOME}/.sky/python_path ] && '
          'cat ${SKY_RUNTIME_DIR:-$HOME}/.sky/python_path 2> /dev/null || '
          'which python3)')
    pth = shlex.quote(PTH_LINE)
    return (
        f'({py} -m pip install --no-deps --quiet {remote_wheel} && '
        f'SP=$({py} -c "import site; print(site.getsitepackages()[0])") && '
        f'echo {pth} > "$SP/{PTH_NAME}") '
        f'|| echo "WARNING - lyceum node autodown NOT installed, this node '
        f'cannot delete itself" >&2 || true;')


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
