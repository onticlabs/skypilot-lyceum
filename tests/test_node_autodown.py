"""Node-side autodown: the machine must be able to delete itself.

Lyceum has no stop and no cloud-side TTL (C5, re-verified against the live
OpenAPI schema 2026-08-07: `CreateVMRequest` takes five fields, none of them an
expiry). A VM bills from `ready` until someone issues DELETE. SkyPilot's design
answer is autodown, executed by the skylet ON the node -- which for this cloud
has never once worked, at a measured cost of 157.5 idle node-hours.

Two independent faults caused that, and both are covered here:

  1. `cluster_utils.get_provider_name` parses the cloud's name out of the
     template's `provider.module` with a regex that requires the in-tree
     `sky.provision.<name>` spelling. Ours said `skypilot_lyceum.provision`,
     which does not match, so the skylet died on an assertion before it reached
     any Lyceum code.
  2. Past that, dispatch needs this package REGISTERED in the skylet process.
     The skylet never calls `plugins.load_plugins()`, so being installed is not
     enough -- registration has to happen some other way.

Every one of these failures is silent in production: `SkyletEvent.run` catches
and retries every 60s, and a `.pth` that raises is swallowed by `site.py`. The
tests are the only place they are loud.
"""
from __future__ import annotations

import pathlib

import pytest
import sky.provision as provision_lib
from sky import clouds
from sky.utils import cluster_utils, registry

from skypilot_lyceum import CLOUD_NAME, node_autodown
from skypilot_lyceum.node_autodown import PTH_NAME as PTH_NAME_FOR_TEST
from skypilot_lyceum.cloud import Lyceum

TEMPLATE = (pathlib.Path(node_autodown.__file__).parent / 'templates' /
            'lyceum-ray.yml.j2')


# --------------------------------------------------------------------------
# Fault 1 -- the provider name the skylet parses
# --------------------------------------------------------------------------
def test_skylets_own_parser_extracts_lyceum_from_our_template():
    """The regression test for the whole outage, run through the REAL parser.

    Asserting the literal template string would pass just as well with a
    spelling SkyPilot cannot parse -- which is exactly the bug. So this feeds
    the rendered value to `cluster_utils.get_provider_name`, the function the
    skylet actually calls, and requires it to yield the name our cloud and
    provisioner are registered under.
    """
    module = None
    for line in TEMPLATE.read_text().splitlines():
        if line.strip().startswith('module:'):
            module = line.split(':', 1)[1].strip()
    assert module, 'template declares no provider.module'

    assert cluster_utils.get_provider_name({'provider': {
        'module': module
    }}) == CLOUD_NAME


def test_the_old_spelling_would_still_fail_that_parser():
    """Pins WHY the spelling matters, so a future 'tidy-up' cannot undo it.

    `skypilot_lyceum.provision` has no dot after `provision`, so the regex
    `(?:providers|provision)\\.(\\w+)\\.?` finds nothing and the assert fires
    inside the skylet -- where nobody sees it.
    """
    with pytest.raises(AssertionError):
        cluster_utils.get_provider_name(
            {'provider': {
                'module': 'skypilot_lyceum.provision'
            }})


# --------------------------------------------------------------------------
# Fault 2 -- registration inside the skylet process
# --------------------------------------------------------------------------
def test_bootstrap_does_nothing_outside_the_skylet(monkeypatch):
    """The .pth runs at EVERY interpreter start on the node -- the user's job,
    every subprocess, every `python -c`. Only the skylet needs Lyceum, and
    importing sky into unrelated processes is both slow and a risk they did not
    sign up for."""
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['python', 'train.py'])
    calls = []
    monkeypatch.setattr(node_autodown, '_register', lambda: calls.append(1))
    node_autodown.bootstrap()
    assert calls == []


def test_bootstrap_registers_inside_the_skylet(monkeypatch):
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['python', '-m', 'sky.skylet.skylet'])
    calls = []
    monkeypatch.setattr(node_autodown, '_register', lambda: calls.append(1))
    node_autodown.bootstrap()
    assert calls == [1]


def test_the_skylet_test_matches_skypilots_own(monkeypatch):
    """SkyPilot identifies its skylet by looking for `sky.skylet.skylet` in the
    command line (`skylet/attempt_skylet.py`). Using a different test would
    drift from it silently."""
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['/usr/bin/python3', '-m', 'sky.skylet.skylet'])
    assert node_autodown._inside_skylet()
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['/usr/bin/python3', '-m', 'sky.skylet.attempt_skylet'])
    assert not node_autodown._inside_skylet()


def test_bootstrap_never_raises(monkeypatch, tmp_path):
    """An exception escaping here is printed by `site.py` and swallowed, so it
    would degrade to today's behaviour with no signal. Contain it and leave
    evidence on the node instead."""
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['python', '-m', 'sky.skylet.skylet'])
    monkeypatch.setattr(node_autodown, '_MARKER', tmp_path / 'marker')

    def boom():
        raise RuntimeError('registry exploded')

    monkeypatch.setattr(node_autodown, '_register', boom)
    node_autodown.bootstrap()          # must not raise
    assert 'registry exploded' in (tmp_path / 'marker').read_text()


def test_bootstrap_records_success_on_the_node(monkeypatch, tmp_path):
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['python', '-m', 'sky.skylet.skylet'])
    monkeypatch.setattr(node_autodown, '_MARKER', tmp_path / 'marker')
    monkeypatch.setattr(node_autodown, '_register', lambda: None)
    node_autodown.bootstrap()
    assert 'ok' in (tmp_path / 'marker').read_text()


def test_register_makes_the_stop_path_resolvable(lyceum_enabled):
    """What `_register` has to achieve, stated as the skylet's own two lookups:
    `StopEvent._stop_cluster` asserts the CLOUD_REGISTRY entry is non-None, then
    dispatch routes `terminate_instances` through the registered provisioner.

    `lyceum_enabled` already registered both, so this pins the postcondition
    rather than re-running registration -- the point is that these two lookups,
    and not merely an importable module, are what the node needs.
    """
    assert registry.CLOUD_REGISTRY.from_str(CLOUD_NAME) is not None
    prov = provision_lib.get_registered_provisioner(CLOUD_NAME)
    assert prov is not None
    assert hasattr(prov.module, 'terminate_instances')


# --------------------------------------------------------------------------
# Getting this package onto the node
# --------------------------------------------------------------------------
def test_no_wheel_means_no_mounts_and_no_setup(monkeypatch):
    """Degrade cleanly. A server built without the wheel still launches jobs;
    it just cannot self-terminate, and falls back to the reaper."""
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: None)
    v = node_autodown.template_variables()
    assert v['lyceum_file_mounts'] == {}
    assert v['lyceum_node_setup_command'] == ''


def test_wheel_produces_a_mount_and_an_install(monkeypatch, tmp_path):
    wheel = tmp_path / 'skypilot_lyceum-0.1.0-py3-none-any.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    v = node_autodown.template_variables()
    assert list(v['lyceum_file_mounts'].values()) == [str(wheel)]
    remote = next(iter(v['lyceum_file_mounts']))
    cmd = v['lyceum_node_setup_command']
    assert remote in cmd
    assert '--no-deps' in cmd, (
        'installing our deps would drag pandas/requests into the node runtime '
        'and could upgrade what SkyPilot pinned; both are already SkyPilot '
        'dependencies, so --no-deps is both safe and required')


def test_install_writes_a_pth_that_python_will_execute(monkeypatch, tmp_path):
    """`site.py` executes a .pth line only when it begins with `import `.
    Anything else is silently treated as a path to add -- which would look
    installed and do nothing."""
    wheel = tmp_path / 'w.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    cmd = node_autodown.template_variables()['lyceum_node_setup_command']
    assert node_autodown.PTH_LINE.startswith('import ')
    assert node_autodown.PTH_LINE in cmd
    assert '.pth' in cmd


def test_two_wheels_is_an_error_not_a_coin_flip(monkeypatch, tmp_path):
    """A lexicographic 'latest' picks 0.9.0 over 0.10.0. Since the image builds
    exactly one wheel, more than one means the build changed in a way nothing
    here can adjudicate -- so refuse rather than ship an arbitrary plugin."""
    monkeypatch.setenv(node_autodown.WHEEL_DIR_ENV, str(tmp_path))
    (tmp_path / 'skypilot_lyceum-0.9.0-py3-none-any.whl').write_bytes(b'')
    assert node_autodown.find_wheel().name.endswith('0.9.0-py3-none-any.whl')
    (tmp_path / 'skypilot_lyceum-0.10.0-py3-none-any.whl').write_bytes(b'')
    with pytest.raises(RuntimeError, match='expected exactly one'):
        node_autodown.find_wheel()


def test_skylet_detection_matches_upstream_containment(monkeypatch):
    """Upstream tests `'sky.skylet.skylet' in arg`. If SkyPilot ever glues the
    args into one element, `endswith` stops matching while upstream still finds
    its skylet -- we would silently stop registering."""
    monkeypatch.setattr(node_autodown, '_process_cmdline',
                        lambda: ['python -m sky.skylet.skylet --port=1234'])
    assert node_autodown._inside_skylet()


def test_install_command_cannot_contain_a_yaml_mapping_indicator(monkeypatch,
                                                                 tmp_path):
    """The setup block is a plain YAML scalar, so a `": "` in the command turns
    it into a mapping and the ENTIRE cluster config stops parsing -- on every
    launch. This is the trap that killed the previous attempt at node-side
    autodown; the failure looks nothing like its cause, so pin it directly.
    """
    wheel = tmp_path / 'w.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    cmd = node_autodown.template_variables()['lyceum_node_setup_command']
    assert ': ' not in cmd, f'plain-scalar-breaking ": " in: {cmd}'
    assert ': ' not in node_autodown.PTH_LINE
    # ' #' is the WORSE trap: it does not fail to parse, it silently truncates
    # the command at that point. The launch then succeeds with autodown quietly
    # missing -- indistinguishable from the bug this whole module fixes.
    assert ' #' not in cmd, f'comment indicator truncates the command: {cmd}'
    assert ' #' not in node_autodown.PTH_LINE


def test_install_failure_does_not_fail_the_launch(monkeypatch, tmp_path):
    """A node that cannot install the plugin should still run the job. The
    alternative -- aborting the launch -- trades a cost bug for an outage."""
    wheel = tmp_path / 'w.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    cmd = node_autodown.template_variables()['lyceum_node_setup_command']
    assert '||' in cmd and 'true' in cmd.split('||')[-1], (
        'the install must not be able to fail the setup step')


def test_template_override_supplies_the_node_install_variables(monkeypatch,
                                                               tmp_path):
    """The variables have to arrive through `TemplateSpec.variables`; the
    template cannot reach into this module by itself."""
    from skypilot_lyceum import provision as lyceum_provision
    wheel = tmp_path / 'w.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    spec = lyceum_provision.template_override(None, None)
    assert 'lyceum_file_mounts' in spec.variables
    assert 'lyceum_node_setup_command' in spec.variables


def test_rendered_template_is_valid_yaml_with_and_without_the_wheel(
        monkeypatch, tmp_path):
    """The install lines are spliced into a YAML block scalar and a flow
    mapping. A quoting slip there produces a cluster config that fails to parse
    on every launch -- for both clouds, since setup runs before anything else.
    """
    jinja2 = pytest.importorskip('jinja2')
    yaml = pytest.importorskip('yaml')
    from tests.test_cloud_class import _deploy_vars
    from sky.resources import Resources
    lyceum = Lyceum()

    wheel = tmp_path / 'skypilot_lyceum-0.1.0-py3-none-any.whl'
    wheel.write_bytes(b'')
    base = _deploy_vars(
        lyceum, Resources(cloud=Lyceum(), instance_type='h100.1x',
                          accelerators={'H100': 1}))
    backend_stub = {
        'cluster_name_on_cloud': 'c-1234abcd', 'num_nodes': 1,
        'credentials': {}, 'ssh_private_key': '/k', 'sky_pip_cmd': 'pip',
        'sky_ray_yaml_remote_path': '/r', 'sky_ray_yaml_local_path': '/l',
        'sky_remote_path': '/sr', 'sky_local_path': '/sl',
        'sky_wheel_hash': 'h', 'initial_setup_commands': [],
        'conda_installation_commands': 'true;',
        'uv_installation_commands': 'true;',
        'ray_skypilot_installation_commands': 'true;',
        'copy_skypilot_templates_commands': 'true;',
        'ssh_max_sessions_config': 'true;',
    }
    for present in (None, wheel):
        monkeypatch.setattr(node_autodown, 'find_wheel', lambda p=present: p)
        rendered = jinja2.Template(TEMPLATE.read_text()).render(
            **base, **backend_stub, **node_autodown.template_variables())
        doc = yaml.safe_load(rendered)
        assert doc['provider']['module'] == 'sky.provision.lyceum'
        assert isinstance(doc['setup_commands'], list)
        if present is not None:
            assert 'pip' in rendered and '.pth' in rendered


# --------------------------------------------------------------------------
# The launch-time leak that autodown does not cover
# --------------------------------------------------------------------------
def test_autostop_without_down_is_refused_at_launch():
    """`sky launch -i N` WITHOUT `--down` asks the node to STOP, which Lyceum
    cannot do: `stop_instances` raises, the skylet swallows it and retries
    forever, and the VM bills forever.

    `execution.py` maps launch-time autostop-without-down to the AUTOSTOP
    feature, which was absent from the unsupported dict -- only STOP was listed,
    and that gate covers `sky autostop`, not `sky launch -i`. So the launch
    validated and the leak was guaranteed rather than merely possible.
    """
    assert (clouds.CloudImplementationFeatures.AUTOSTOP
            in Lyceum._CLOUD_UNSUPPORTED_FEATURES)
    reason = Lyceum._CLOUD_UNSUPPORTED_FEATURES[
        clouds.CloudImplementationFeatures.AUTOSTOP]
    assert '--down' in reason, 'the message must name the thing that DOES work'


def test_site_really_executes_the_pth_line(tmp_path):
    """Prove the .pth mechanism against the real `site` machinery.

    `site.addsitedir` is the function `site.py` applies to every site-packages
    directory at startup, and executing `import ...` lines is what it does. So
    this exercises the exact code path the node will, without needing a venv.

    Worth having: the whole design rests on that line being executed, and the
    first version of this test used PYTHONPATH and failed -- .pth files are
    processed ONLY in site directories. That distinction is why the install
    command asks Python for `site.getsitepackages()[0]` rather than guessing.
    """
    import site

    site_dir = tmp_path / 'site-packages'
    site_dir.mkdir()
    (site_dir / 'lyceum_pth_proof.py').write_text(
        'import pathlib\n'
        f'pathlib.Path({str(tmp_path / "ran")!r}).write_text("yes")\n')
    (site_dir / PTH_NAME_FOR_TEST).write_text('import lyceum_pth_proof\n')

    import sys
    sys.path.insert(0, str(site_dir))
    try:
        site.addsitedir(str(site_dir))
    finally:
        sys.path.remove(str(site_dir))
        sys.modules.pop('lyceum_pth_proof', None)

    assert (tmp_path / 'ran').read_text() == 'yes', (
        'site.addsitedir did not execute the .pth line — node-side autodown '
        'would install and silently do nothing')


def test_the_install_targets_a_directory_where_pth_files_are_honoured(
        monkeypatch, tmp_path):
    """.pth files are inert outside a site directory. The command must ask
    Python where that is rather than guessing a path."""
    wheel = tmp_path / 'w.whl'
    wheel.write_bytes(b'')
    monkeypatch.setattr(node_autodown, 'find_wheel', lambda: wheel)
    cmd = node_autodown.template_variables()['lyceum_node_setup_command']
    assert 'site.getsitepackages()' in cmd
