"""Behavioural tests for `skypilot_lyceum.provision.instance`.

Shape follows SkyPilot's own `tests/unit_tests/kubernetes/test_provision.py`:
a `_make_provision_config()` helper that builds a minimal
`sky.provision.common.ProvisionConfig`, a single `_patch_lyceum_client()`
helper that stubs *only* the SDK-touching boundary, then behavioural tests.

The boundary is `LyceumClient`, not `requests`. Everything below is about the
provisioner's state machine -- adoption vs. duplication, the ready-but-null-IP
poll, status mapping, teardown blast radius -- and none of it is about HTTP.
`tests/test_api_client.py` owns the HTTP layer.

Every correction number (C1, C2, C4, C7, C10) refers to the Lyceum API quirk
table in README.md.
"""
from __future__ import annotations

import collections
import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from sky import exceptions as sky_exceptions
from sky.provision import common
from sky.utils import status_lib

from skypilot_lyceum import api
from skypilot_lyceum.provision import config as config_lib
from skypilot_lyceum.provision import instance

# The cluster under test. `vm_list_mixed.json` deliberately carries a
# terminated VM and a live VM sharing this display_name (C4), plus one
# unrelated cluster that teardown must never touch.
CLUSTER = 'sky-cluster'
CLUSTER_ON_CLOUD = 'sky-cluster-abc'
OTHER_CLUSTER_ON_CLOUD = 'sky-cluster-other'
#: A different cluster whose name merely STARTS WITH ours. SkyPilot builds
#: cluster_name_on_cloud as `<name>-<hash>`, so two live clusters sharing a
#: hash prefix is ordinary, not contrived. Any prefix/substring match on
#: display_name adopts or terminates this one by mistake.
PREFIX_DECOY_ON_CLOUD = CLUSTER_ON_CLOUD + 'd'
REGION = 'lyceum'

#: Exception types for which `sky/provision/provisioner.py:172-188` re-raises
#: WITHOUT tearing the cluster down. Raising one of these after a VM has been
#: created converts a failed launch into a permanent leak, because there is no
#: cloud-side TTL to catch it (C5).
TEARDOWN_SKIPPING_ERRORS = (
    sky_exceptions.NoClusterLaunchedError,
    sky_exceptions.InvalidCloudCredentials,
    sky_exceptions.InconsistentHighAvailabilityError,
    sky_exceptions.ExecutionPausedError,
)

PUBLIC_KEY = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI0000000000000000 sky-key'


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _VM(api.VM):
    """`api.VM` with its two predicates filled in.

    `api.VM.is_terminal` / `is_usable` are still stubs, and this file is not
    about them -- `tests/test_api_client.py` pins their behaviour. Implementing them
    here with the documented semantics (TERMINAL_STATUSES; ready AND non-null
    IP) keeps a provisioner failure from masquerading as an api.py failure.
    """

    @property
    def is_terminal(self) -> bool:
        return self.status in api.TERMINAL_STATUSES

    @property
    def is_usable(self) -> bool:
        return self.status in api.READY_STATUSES and self.ip is not None


def _split_host_port(value: Optional[str]) -> Tuple[Optional[str], int]:
    """Local stand-in for `api.parse_ip_address` (C2), which is still a stub."""
    if not value:
        return None, 22
    if ':' in value:
        host, _, port = value.rpartition(':')
        return host, int(port)
    return value, 22


def _vm(payload: Dict[str, Any], **overrides: Any) -> _VM:
    """Build a `_VM` from a captured API payload, with field overrides."""
    data = dict(payload)
    data.update(overrides)
    specs = data.get('instance_specs') or {}
    host, port = _split_host_port(data.get('ip_address'))
    return _VM(
        vm_id=data['vm_id'],
        status=data['status'],
        display_name=data.get('display_name'),
        hardware_profile=data.get('hardware_profile') or specs.get('gpu_type'),
        gpu_count=data.get('gpu_count') or specs.get('gpu_count'),
        instance_type=data.get('instance_type'),
        created_at=data.get('created_at'),
        ip=host,
        ssh_port=port,
        raw=data,
    )


class FakeLyceumClient:
    """In-memory stand-in for `api.LyceumClient`.

    One VM store, three read paths over it (`get_vm`, `list_vms`,
    `find_vms_by_display_name`), so the provisioner can resolve its cluster by
    whichever path it likes and the assertions still hold.

    `poll_script` is a sequence of store snapshots for the polling tests. Each
    read method walks the script on its *own* cursor, so one poll round that
    happens to call two different read methods still observes one consistent
    snapshot. A provisioner that returns after a single observation therefore
    never advances past snapshot 0 -- which is exactly the C10 trap.

    Three opt-in failure/sloppiness modes, each defaulting off:

    `loose_find` makes `find_vms_by_display_name` return a deliberately sloppy
    superset -- prefix matches, terminal VMs included, unordered. See
    `test_terminate_instances_never_touches_another_cluster` for why the
    provisioner is required to survive it.

    `create_commits` means "the server created the VM, then the client raised":
    used with `create_error`, `create_vm` appends the VM to the store *and
    then* raises, reproducing a timeout or reset after the POST committed. The
    caller never learns the vm_id, so display_name is the only handle left.

    `post_create_error` fails the first read that happens after a successful
    create -- the API having a bad minute in the window between create and
    whatever run_instances does next.
    """

    def __init__(self,
                 vms: Optional[Sequence[_VM]] = None,
                 *,
                 poll_script: Optional[Sequence[Sequence[_VM]]] = None,
                 create_error: Optional[BaseException] = None,
                 create_commits: bool = False,
                 post_create_error: Optional[BaseException] = None,
                 terminate_error: Optional[BaseException] = None,
                 loose_find: bool = False) -> None:
        if poll_script is not None:
            self._script: List[List[_VM]] = [list(s) for s in poll_script]
        else:
            self._script = [list(vms or [])]
        self._cursor: Dict[str, int] = collections.defaultdict(int)
        self._create_error = create_error
        self._create_commits = create_commits
        self._post_create_error = post_create_error
        self._terminate_error = terminate_error
        self._loose_find = loose_find
        self._created = 0

        self.create_calls: List[Dict[str, Any]] = []
        self.terminate_calls: List[str] = []
        self.reads: List[Tuple[str, Any]] = []
        #: vm_ids the *server* holds because a create committed, whether or not
        #: the caller ever saw them. Anything left here at the end of a failed
        #: run_instances is a VM on the meter that nobody is tracking.
        self.committed_vm_ids: List[str] = []

    # ---- store -----------------------------------------------------------

    def _snapshot(self, method: str, arg: Any = None) -> List[_VM]:
        self.reads.append((method, arg))
        if self._post_create_error is not None and self._created:
            raise self._post_create_error
        index = min(self._cursor[method], len(self._script) - 1)
        self._cursor[method] += 1
        return self._script[index]

    @property
    def polls(self) -> int:
        """How far the furthest read path walked the script."""
        return max(self._cursor.values(), default=0)

    def _mutate_all(self, vm_id: str, **changes: Any) -> None:
        for i, snapshot in enumerate(self._script):
            self._script[i] = [
                dataclasses.replace(v, **changes) if v.vm_id == vm_id else v
                for v in snapshot
            ]

    def _append_all(self, vm: _VM) -> None:
        for snapshot in self._script:
            snapshot.append(vm)

    # ---- LyceumClient surface -------------------------------------------

    def create_vm(self,
                  *,
                  public_key: Optional[str] = None,
                  hardware_profile: Optional[str] = None,
                  gpu_count: int = 1,
                  display_name: Optional[str] = None,
                  use_spot: bool = False,
                  **extra: Any) -> _VM:
        self.create_calls.append({
            'public_key': public_key,
            'hardware_profile': hardware_profile,
            'gpu_count': gpu_count,
            'display_name': display_name,
            'use_spot': use_spot,
            **extra,
        })
        if self._create_error is not None and not self._create_commits:
            raise self._create_error
        self._created += 1
        vm_id = f'created-vm-{self._created}'
        raw = {
            'vm_id': vm_id,
            'status': 'pending',
            'ip_address': None,
            'created_at': f'2026-07-31T16:0{self._created}:00.000000Z',
            'instance_specs': {
                'gpu_type': hardware_profile,
                'gpu_count': gpu_count,
            },
            # C6: both are null in the create response.
            'hardware_profile': None,
            'gpu_count': None,
            'instance_type': 'spot' if use_spot else 'on-demand',
            'display_name': display_name,
        }
        # The API answers `pending`; the store holds the VM as it will look a
        # few minutes later, so a provisioner that blocks inside run_instances
        # makes progress instead of spinning against a permanently-pending VM.
        self._append_all(
            _vm(raw, status='ready', ip_address='203.0.113.11'))
        self.committed_vm_ids.append(vm_id)
        if self._create_error is not None:
            # The server committed the VM; the response never made it back.
            raise self._create_error
        return _vm(raw)

    def get_vm(self, vm_id: str) -> _VM:
        for vm in self._snapshot('get_vm', vm_id):
            if vm.vm_id == vm_id:
                return vm
        raise api.LyceumNotFoundError(f'VM not found: {vm_id}')

    def list_vms(self, *, include_terminated: bool = False) -> List[_VM]:
        vms = self._snapshot('list_vms', include_terminated)
        if include_terminated:
            return list(vms)
        return [v for v in vms if not v.is_terminal]

    def find_vms_by_display_name(self, display_name: str) -> List[_VM]:
        vms = self._snapshot('find_vms_by_display_name', display_name)
        if self._loose_find:
            # Deliberately sloppy: prefix match, terminal VMs kept, no
            # ordering. The provisioner must not trust any of it.
            return [
                v for v in vms
                if (v.display_name or '').startswith(display_name)
            ]
        matches = [
            v for v in vms
            if v.display_name == display_name and not v.is_terminal
        ]
        return sorted(matches, key=lambda v: v.created_at or '', reverse=True)

    def terminate_vm(self, vm_id: str) -> None:
        self.terminate_calls.append(vm_id)
        if self._terminate_error is not None:
            raise self._terminate_error
        self._mutate_all(vm_id, status='terminated')
        for i, committed in enumerate(list(self.committed_vm_ids)):
            if committed == vm_id:
                del self.committed_vm_ids[i]
                break

    def get_user_status(self) -> Dict[str, Any]:
        return {'status': 'authenticated'}


class _Clock:
    """A fake wall clock. No test in this file may sleep for real."""

    #: Trips instead of hanging if the code under test ignores its own budget.
    MAX_SLEEPS = 10_000

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.start = start
        self.now = start
        self.sleeps: List[float] = []

    @property
    def elapsed(self) -> float:
        return self.now - self.start

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) > self.MAX_SLEEPS:
            raise AssertionError(
                f'slept {len(self.sleeps)} times without terminating -- the '
                'poll loop is not bounded by PROVISION_TIMEOUT_S')
        self.now += seconds

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(time, 'sleep', self.sleep)
        monkeypatch.setattr(time, 'time', lambda: self.now)
        monkeypatch.setattr(time, 'monotonic', lambda: self.now)
        monkeypatch.setattr(time, 'perf_counter', lambda: self.now)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_provision_config(count: int = 1,
                           **node_config_overrides: Any
                           ) -> common.ProvisionConfig:
    """A minimal ProvisionConfig sufficient to drive `run_instances`.

    `node_config` mirrors what `Lyceum.make_deploy_resources_variables` feeds
    through `templates/lyceum-ray.yml.j2`, in the shape the in-tree RunPod and
    Shadeform providers use (`InstanceType`, `PublicKey`). The public key is
    also mirrored into `authentication_config`, since either is a defensible
    place for the provisioner to read it from.
    """
    node_config: Dict[str, Any] = {
        'InstanceType': 'l40s.1x',
        'PublicKey': PUBLIC_KEY,
        'DiskSize': 512,
        'UseSpot': False,
    }
    node_config.update(node_config_overrides)
    return common.ProvisionConfig(
        provider_config={
            'region': REGION,
            'use_spot': node_config['UseSpot'],
        },
        authentication_config={
            'ssh_public_key': PUBLIC_KEY,
            'ssh_private_key': '~/.ssh/sky-key',
            'ssh_user': instance.SSH_USER,
        },
        docker_config={},
        node_config=node_config,
        count=count,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


def _patch_lyceum_client(monkeypatch, fake: FakeLyceumClient) -> None:
    """Route every way `provision/instance.py` could reach a client to `fake`.

    Deliberately covers each plausible seam -- `api.LyceumClient()`, a
    name imported into the instance module, or a `_get_client()`-style factory
    -- because the provisioner is what these tests are about and the exact
    construction idiom is an implementation detail. Also stubs credential
    lookup so no test can touch a real `~/.lyceum/api_key`.
    """
    monkeypatch.setenv('LYCEUM_API_KEY', 'lk_' + '0' * 64)
    monkeypatch.setattr(api, 'read_api_key', lambda *a, **k: 'lk_' + '0' * 64)
    monkeypatch.setattr(api, 'LyceumClient', lambda *a, **k: fake)

    for name in ('LyceumClient', '_client', '_get_client', 'get_client',
                 'make_client'):
        if hasattr(instance, name):
            monkeypatch.setattr(instance, name, lambda *a, **k: fake)
    # Defeat any module-level memoisation of the client.
    for name in ('_CLIENT', '_CACHED_CLIENT', '_client_instance'):
        if hasattr(instance, name):
            monkeypatch.setattr(instance, name, None)


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> _Clock:
    """Freeze time for every test in this module. Nothing sleeps for real."""
    c = _Clock()
    c.install(monkeypatch)
    return c


@pytest.fixture
def vm(fixture):
    """`vm('vm_status_ready_bare_ip', display_name=...)` -> `_VM`."""

    def _make(name: str, **overrides: Any) -> _VM:
        return _vm(fixture(name), **overrides)

    return _make


@pytest.fixture
def mixed_vms(fixture) -> List[_VM]:
    """The three VMs of `vm_list_mixed.json`, as `_VM`s.

    [0] terminated h100, display_name sky-cluster-abc  (C4: keeps its name)
    [1] live l40s,       display_name sky-cluster-abc  (the real node)
    [2] live a100,       display_name sky-cluster-other (must never be touched)
    """
    return [_vm(p) for p in fixture('vm_list_mixed')['vms']]


@pytest.fixture
def prefix_decoy(vm) -> _VM:
    """A live VM of a *different* cluster whose name starts with ours."""
    return vm('vm_status_ready_bare_ip',
              vm_id='decoy-prefix-vm',
              display_name=PREFIX_DECOY_ON_CLOUD,
              created_at='2026-07-31T15:10:00.000000Z')


# --------------------------------------------------------------------------
# run_instances
# --------------------------------------------------------------------------


def test_run_instances_adopts_an_existing_live_vm(monkeypatch, vm):
    """Prevents a duplicate GPU VM billing on SkyPilot's failover retry.

    SkyPilot calls run_instances again for the same cluster during failover
    and after a transient error. Creating a second VM leaves the first one
    running and unreferenced -- a second GPU on the meter until the orphan
    reaper notices.
    """
    live = vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD)
    fake = FakeLyceumClient([live])
    _patch_lyceum_client(monkeypatch, fake)

    record = instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                                    _make_provision_config())

    assert fake.create_calls == []
    assert record.head_instance_id == live.vm_id
    assert record.created_instance_ids == []


def test_run_instances_ignores_a_terminated_vm_with_the_same_name(
        monkeypatch, mixed_vms, prefix_decoy):
    """C4: a dead VM keeps its display_name forever; adopting it is a hang.

    `vm_list_mixed` has a terminated h100 and a live l40s both named
    sky-cluster-abc. Resolving the terminated one would hand SkyPilot an
    instance id that can never come up; resolving `sky-cluster-abcd` would
    hand it someone else's running node.

    The client runs in `loose_find` mode, so `find_vms_by_display_name` hands
    back the sloppiest superset a client could produce -- prefix matches,
    terminal VMs kept, unordered. That is what makes this test mean what its
    name says: against the contract-honouring fake, api.py's own filter had
    already removed both traps, so any implementation that merely forwarded the
    call passed and the assertion proved nothing about the provisioner. The
    requirement pinned here is deliberately belt-and-braces -- check exact
    display_name equality and non-terminality before adopting -- because C4 is
    the review's most dangerous finding, the check is two comparisons, and the
    failure modes are a 15-minute wait on a machine that no longer exists or a
    launch quietly landing on a neighbouring cluster's GPU.
    """
    terminated, live = mixed_vms[0], mixed_vms[1]
    fake = FakeLyceumClient(mixed_vms + [prefix_decoy], loose_find=True)
    _patch_lyceum_client(monkeypatch, fake)

    record = instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                                    _make_provision_config())

    assert record.head_instance_id != terminated.vm_id
    assert record.head_instance_id != prefix_decoy.vm_id
    assert record.head_instance_id == live.vm_id
    assert prefix_decoy.vm_id not in (record.created_instance_ids or [])
    assert fake.create_calls == []


@pytest.mark.parametrize('status', ['pending', 'provisioning'])
def test_run_instances_adopts_a_vm_that_is_still_provisioning(
        monkeypatch, vm, status):
    """The duplicate-GPU window: adoption must gate on "not terminal", not "usable".

    Provisioning takes 130 s (spot) to 221 s (on-demand). SkyPilot calls
    run_instances again for the same cluster on a failover retry or after a
    transient error, and that retry can easily land inside that window. If
    adoption is gated on `VM.is_usable` -- ready AND a non-null IP -- the
    still-provisioning VM looks like "nothing here", a second create fires, and
    the first VM keeps billing with nothing referencing it. Per C5 there is no
    cloud-side TTL, so nothing but the reaper ever notices; at b300 x8 that is
    $63.92/h.

    Every other adoption test in this file uses a VM that is already `ready`,
    so none of them can see this. The gate for adoption is `not is_terminal`;
    `is_usable` is the gate for `wait_instances` (C10), and conflating the two
    is what costs the money.
    """
    provisioning = vm('vm_status_ready_null_ip',
                      display_name=CLUSTER_ON_CLOUD,
                      status=status)
    assert provisioning.ip is None, 'fixture drift: this VM must have no IP yet'
    usable = dataclasses.replace(provisioning, status='ready',
                                 ip='198.51.100.30', ssh_port=22)
    # Later snapshots let an implementation that blocks inside run_instances
    # finish, so a hang is not mistaken for a failed adoption.
    fake = FakeLyceumClient(
        poll_script=[[provisioning], [provisioning], [usable]])
    _patch_lyceum_client(monkeypatch, fake)

    record = instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                                    _make_provision_config())

    assert fake.create_calls == [], (
        f'a VM in {status!r} for this cluster was not adopted; a second GPU '
        'is now billing with nothing referencing it')
    assert record.head_instance_id == provisioning.vm_id
    assert record.created_instance_ids == []


@pytest.mark.parametrize('failure', ['response_lost', 'readback_failed'])
def test_run_instances_never_strands_a_vm_it_created(monkeypatch, failure):
    """Create succeeded, a later step did not: the VM must not be left running.

    Two shapes, both observed in the wild against flaky APIs:

    `response_lost` -- the POST committed server-side and the response never
    came back (timeout, reset). The caller has no vm_id at all, so the only
    handle left is the display_name it sent.

    `readback_failed` -- create returned, and the next call fails.

    The invariant asserted is the same for both, and it is the only one that
    matters: **every VM the server created is either referenced by the returned
    record or terminated before the exception propagates.** Nothing is left
    unaccounted for.

    Why cleanup rather than "leave it for teardown": SkyPilot does normally
    tear down a failed provision, but `sky/provision/provisioner.py:172-188`
    re-raises without teardown for four exception types -- and
    `NoClusterLaunchedError` is a natural-looking choice for "create failed",
    one edit away from turning this into a permanent leak. Even on the happy
    path, teardown goes through the same API that just failed and gives up
    after three retries with StopFailoverError. Per C5 there is no cloud-side
    TTL, so a stranded b300 x8 bills $63.92/h until the reaper finds it. The
    cheapest place to stop the meter is the function that started it, while it
    still holds the handle. So: clean up, and also do not raise a type that
    disables SkyPilot's own second line of defence.
    """
    boom = api.LyceumServerError('connection reset (502)')
    if failure == 'response_lost':
        fake = FakeLyceumClient([], create_error=boom, create_commits=True)
    else:
        fake = FakeLyceumClient([], post_create_error=boom)
    _patch_lyceum_client(monkeypatch, fake)

    record = None
    raised = None
    try:
        record = instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                                        _make_provision_config())
    except Exception as exc:  # noqa: BLE001 - the type is asserted below
        raised = exc

    # Exactly one: zero means the scenario was never reached and the rest of
    # this test would pass vacuously; more than one means create was retried
    # while its outcome was unknown, which is how the duplicate billing GPU
    # happens.
    assert len(fake.create_calls) == 1, (
        f'create was issued {len(fake.create_calls)} times for an empty '
        'one-node cluster')
    for call in fake.create_calls:
        assert call['display_name'] == CLUSTER_ON_CLOUD, (
            'the created VM carries no recoverable name, so neither teardown '
            'nor the reaper can ever find it')

    unaccounted = set(fake.committed_vm_ids)
    if record is not None:
        unaccounted -= set(record.created_instance_ids or [])
        unaccounted.discard(record.head_instance_id)
    assert not unaccounted, (
        f'{sorted(unaccounted)} was created and then neither terminated nor '
        'returned in the ProvisionRecord -- a GPU billing with nothing '
        'referencing it')

    if raised is not None:
        assert not isinstance(raised, TEARDOWN_SKIPPING_ERRORS), (
            f'{type(raised).__name__} makes SkyPilot skip teardown entirely, '
            'removing the only backstop after a VM was already created')


def test_run_instances_names_the_created_vm_after_the_cluster(monkeypatch):
    """The whole identity scheme is display_name == cluster_name_on_cloud.

    Lyceum has no tags and no server-side filtering, so a VM created under any
    other name is unreachable by every later
    operation: it can never be adopted, queried, or terminated, and only the
    orphan reaper would ever find it.
    """
    fake = FakeLyceumClient([])
    _patch_lyceum_client(monkeypatch, fake)

    instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                           _make_provision_config())

    assert len(fake.create_calls) == 1
    assert fake.create_calls[0]['display_name'] == CLUSTER_ON_CLOUD


def test_run_instances_returns_a_provision_record_for_the_created_vm(
        monkeypatch):
    """A record whose head_instance_id is not the real VM strands the cluster.

    Everything downstream -- get_cluster_info, wait_instances, teardown --
    keys off this record. `provider_name` must be the registered name or
    `sky.provision` dispatches the follow-up calls to the wrong provider.
    """
    fake = FakeLyceumClient([])
    _patch_lyceum_client(monkeypatch, fake)

    record = instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                                    _make_provision_config())

    assert isinstance(record, common.ProvisionRecord)
    assert record.provider_name == instance.PROVIDER_NAME == 'lyceum'
    assert record.cluster_name in (CLUSTER, CLUSTER_ON_CLOUD)
    assert record.head_instance_id
    assert record.created_instance_ids == [record.head_instance_id]
    assert record.resumed_instance_ids == []


def test_run_instances_converts_capacity_error_to_resources_unavailable(
        monkeypatch, fixture):
    """C7: capacity exhaustion must fail over, not retry.

    Lyceum signals "no capacity" with an HTTP 500 whose only distinguishing
    mark is the detail string. If that does not surface as
    ResourcesUnavailableError, the optimizer never fails over to Shadeform and
    the launch dies on an exhausted SKU.
    """
    detail = fixture('error_500_capacity')['detail']
    fake = FakeLyceumClient([], create_error=api.LyceumCapacityError(detail))
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(sky_exceptions.ResourcesUnavailableError):
        instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                               _make_provision_config())


def test_run_instances_does_not_convert_server_error_to_unavailable(
        monkeypatch):
    """C7, the other direction: a real 5xx must not burn the failover budget.

    Conflating a transient server fault with capacity exhaustion spends a
    failover attempt (and eventually the whole optimizer plan) on a cloud that
    is merely having a bad minute.
    """
    fake = FakeLyceumClient([],
                            create_error=api.LyceumServerError('boom (502)'))
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(Exception) as exc_info:  # noqa: B017 - type asserted below
        instance.run_instances(REGION, CLUSTER, CLUSTER_ON_CLOUD,
                               _make_provision_config())

    assert not isinstance(exc_info.value,
                          sky_exceptions.ResourcesUnavailableError)
    assert isinstance(exc_info.value,
                      (api.LyceumServerError, common.ProvisionerError))


# --------------------------------------------------------------------------
# wait_instances
# --------------------------------------------------------------------------


def test_wait_instances_does_not_return_on_ready_with_null_ip(
        monkeypatch, clock, vm):
    """C10, the null-IP trap -- observed live on an h200 at 104 s.

    A VM reports `status: "ready"` while `ip_address` is still null; the IP
    appears on a later poll. Returning at the first `ready` hands SkyPilot a
    null host and fails the launch on an arbitrary subset of nodes. The gate
    is `VM.is_usable`, never `status` alone.
    """
    not_yet = vm('vm_status_ready_null_ip', display_name=CLUSTER_ON_CLOUD)
    usable = dataclasses.replace(not_yet, ip='198.51.100.30', ssh_port=22)
    fake = FakeLyceumClient(poll_script=[[not_yet], [not_yet], [usable]])
    _patch_lyceum_client(monkeypatch, fake)

    instance.wait_instances(REGION, CLUSTER_ON_CLOUD,
                            status_lib.ClusterStatus.UP)

    assert fake.polls >= 2, (
        'returned after a single observation of status="ready" while '
        'ip_address was still null')
    assert clock.sleeps, 'polled without sleeping between attempts'


def test_wait_instances_returns_once_the_vm_is_usable(monkeypatch, vm):
    """The positive half of C10: a usable VM must not be polled forever."""
    usable = vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD)
    fake = FakeLyceumClient([usable])
    _patch_lyceum_client(monkeypatch, fake)

    assert instance.wait_instances(REGION, CLUSTER_ON_CLOUD,
                                   status_lib.ClusterStatus.UP) is None


def test_wait_instances_raises_on_a_terminal_status(monkeypatch, clock, vm):
    """A failed VM must abort the wait, not burn the full 15-minute budget.

    `provisioner.py` retries wait_instances on RuntimeError and then gives up,
    so ProvisionerError (a RuntimeError subclass) is what SkyPilot handles
    sanely here. Polling a dead VM to timeout delays failover by 15 minutes
    per attempt.
    """
    dead = vm('vm_status_ready_null_ip',
              display_name=CLUSTER_ON_CLOUD,
              status='failed')
    fake = FakeLyceumClient([dead])
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(common.ProvisionerError):
        instance.wait_instances(REGION, CLUSTER_ON_CLOUD,
                               status_lib.ClusterStatus.UP)

    assert clock.elapsed < instance.PROVISION_TIMEOUT_S, (
        'waited out the full timeout on a VM that was already terminal')


def test_wait_instances_respects_the_provision_timeout(monkeypatch, clock, vm):
    """An unbounded poll loop wedges the launch and holds a billing VM.

    Time is monkeypatched, so this asserts on the loop's own accounting: it
    must give up somewhere around PROVISION_TIMEOUT_S rather than never.
    """
    stuck = vm('vm_status_ready_null_ip', display_name=CLUSTER_ON_CLOUD)
    fake = FakeLyceumClient([stuck])
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(common.ProvisionerError):
        instance.wait_instances(REGION, CLUSTER_ON_CLOUD,
                               status_lib.ClusterStatus.UP)

    assert clock.sleeps, 'busy-waited without sleeping'
    assert clock.elapsed >= instance.PROVISION_TIMEOUT_S * 0.9, (
        f'gave up after {clock.elapsed}s, well short of the '
        f'{instance.PROVISION_TIMEOUT_S}s budget')
    assert clock.elapsed <= instance.PROVISION_TIMEOUT_S * 1.5, (
        f'overshot the {instance.PROVISION_TIMEOUT_S}s budget by too much')


# --------------------------------------------------------------------------
# get_cluster_info
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('fixture_name', 'ip_address', 'expected_host', 'expected_port'),
    [
        # Captured live, minutes apart, from the same org (C2).
        ('vm_status_ready_bare_ip', None, '203.0.113.10', 22),
        ('vm_status_ready_host_port', None, '198.51.100.20', 22),
        # Derived from the host:port fixture: proves the port is parsed, not
        # assumed. Lyceum has not been observed on a non-22 port, but the
        # field is free-form and RunPod/Vast both use one.
        ('vm_status_ready_host_port', '198.51.100.20:2222', '198.51.100.20',
         2222),
    ],
)
def test_get_cluster_info_parses_the_polymorphic_ip_address(
        monkeypatch, vm, fixture_name, ip_address, expected_host,
        expected_port):
    """C1 + C2: wrong ssh_user or an unsplit host:port is an unreachable node.

    The docs say `ssh root@<ip>`; root and ubuntu are both refused and the
    real user is `lyceum`. And `ip_address` is bare on some VMs and
    `host:port` on others -- feeding the raw string through as a host produces
    an intermittent, per-node SSH failure.
    """
    overrides: Dict[str, Any] = {'display_name': CLUSTER_ON_CLOUD}
    if ip_address is not None:
        overrides['ip_address'] = ip_address
    live = vm(fixture_name, **overrides)
    fake = FakeLyceumClient([live])
    _patch_lyceum_client(monkeypatch, fake)

    info = instance.get_cluster_info(REGION, CLUSTER_ON_CLOUD)

    assert isinstance(info, common.ClusterInfo)
    assert info.provider_name == instance.PROVIDER_NAME == 'lyceum'
    assert info.ssh_user == instance.SSH_USER == 'lyceum'
    assert info.head_instance_id in info.instances, (
        'head_instance_id must be a key of instances')
    assert info.num_instances == 1
    head = info.instances[info.head_instance_id][0]
    assert isinstance(head, common.InstanceInfo)
    assert head.instance_id == live.vm_id
    assert head.ssh_port == expected_port
    assert head.get_feasible_ip() == expected_host
    assert ':' not in head.get_feasible_ip()


def test_get_cluster_info_skips_a_terminated_namesake(monkeypatch, mixed_vms):
    """C4: resolving the dead namesake points SkyPilot at a machine that is gone.

    The terminated h100 and the live l40s share display_name sky-cluster-abc.
    Only the live one may appear.
    """
    fake = FakeLyceumClient(mixed_vms)
    _patch_lyceum_client(monkeypatch, fake)
    terminated, live = mixed_vms[0], mixed_vms[1]

    info = instance.get_cluster_info(REGION, CLUSTER_ON_CLOUD)

    assert terminated.vm_id not in info.instances
    assert info.head_instance_id == live.vm_id
    assert info.num_instances == 1


def test_get_cluster_info_with_no_live_vm_is_empty_not_an_error(monkeypatch):
    """`sky status` calls this on a torn-down cluster; raising breaks the CLI.

    An empty ClusterInfo with head_instance_id None is the documented "nothing
    here" answer (see the in-tree Shadeform provider).
    """
    fake = FakeLyceumClient([])
    _patch_lyceum_client(monkeypatch, fake)

    info = instance.get_cluster_info(REGION, CLUSTER_ON_CLOUD)

    assert info.instances == {}
    assert info.head_instance_id is None
    assert info.provider_name == 'lyceum'


# --------------------------------------------------------------------------
# query_instances
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('lyceum_status', 'expected'),
    [
        ('pending', status_lib.ClusterStatus.INIT),
        ('provisioning', status_lib.ClusterStatus.INIT),
        ('ready', status_lib.ClusterStatus.UP),
        ('running', status_lib.ClusterStatus.UP),
        # None is SkyPilot's "gone" signal. STOPPED would be a lie: Lyceum has
        # no stop, and a STOPPED cluster looks restartable in `sky status`.
        ('terminated', None),
        ('failed', None),
        ('error', None),
    ],
)
def test_query_instances_maps_the_lyceum_status_vocabulary(
        monkeypatch, vm, lyceum_status, expected):
    """C4: a mis-mapped status either hides a live VM or resurrects a dead one.

    In particular a terminal VM must map to None, not STOPPED -- SkyPilot
    treats STOPPED as restartable, and Lyceum has no start.
    """
    node = vm('vm_status_ready_bare_ip',
              display_name=CLUSTER_ON_CLOUD,
              status=lyceum_status)
    fake = FakeLyceumClient([node])
    _patch_lyceum_client(monkeypatch, fake)

    result = instance.query_instances(CLUSTER,
                                      CLUSTER_ON_CLOUD,
                                      non_terminated_only=False)

    assert node.vm_id in result, (
        'non_terminated_only=False must report every VM of the cluster')
    status, reason = result[node.vm_id]
    assert status == expected
    assert reason is None or isinstance(reason, str)


def test_query_instances_excludes_terminated_when_asked(monkeypatch,
                                                        mixed_vms):
    """C4: terminated VMs linger in /vms/list forever, keeping their name.

    With non_terminated_only=True the dead namesake must be absent entirely --
    not present with a None status, which would make SkyPilot believe the
    cluster is partially gone.
    """
    fake = FakeLyceumClient(mixed_vms)
    _patch_lyceum_client(monkeypatch, fake)
    terminated, live = mixed_vms[0], mixed_vms[1]

    result = instance.query_instances(CLUSTER,
                                      CLUSTER_ON_CLOUD,
                                      non_terminated_only=True)

    assert terminated.vm_id not in result
    assert result[live.vm_id][0] == status_lib.ClusterStatus.UP


def test_query_instances_returns_only_the_requested_cluster(
        monkeypatch, mixed_vms):
    """Leaking another cluster's VM into this cluster's status is a teardown risk.

    `sky status` and the autostop path both act on what this returns.
    """
    fake = FakeLyceumClient(mixed_vms)
    _patch_lyceum_client(monkeypatch, fake)
    other = mixed_vms[2]
    assert other.display_name == OTHER_CLUSTER_ON_CLOUD

    result = instance.query_instances(CLUSTER, CLUSTER_ON_CLOUD)

    assert other.vm_id not in result
    assert set(result) <= {v.vm_id for v in mixed_vms[:2]}


def test_query_instances_accepts_retry_if_missing(monkeypatch, mixed_vms):
    """The dispatcher binds this kwarg; omitting it is a TypeError mid-launch.

    `sky.provision._route_to_cloud_impl` does `inspect.signature(fn).bind(...)`,
    so a missing parameter fails inside a launch rather than at import. The
    in-tree Shadeform provider in SkyPilot 0.13.0 omits this parameter, which is
    a TypeError on every status refresh for its clusters.
    """
    fake = FakeLyceumClient(mixed_vms)
    _patch_lyceum_client(monkeypatch, fake)

    with_flag = instance.query_instances(CLUSTER,
                                         CLUSTER_ON_CLOUD,
                                         retry_if_missing=True)
    without_flag = instance.query_instances(CLUSTER, CLUSTER_ON_CLOUD)

    assert with_flag == without_flag


# --------------------------------------------------------------------------
# stop_instances
# --------------------------------------------------------------------------


def test_stop_instances_raises_and_terminates_nothing(monkeypatch, mixed_vms):
    """A silent no-op leaves a GPU billing while SkyPilot believes it stopped.

    Lyceum has no stop API at all. NotImplementedError specifically:
    `provisioner.py` catches it around teardown_cluster and
    converts it into StopFailoverError with an actionable message -- any other
    type falls into the generic retry path and eventually a bare traceback.
    Terminating instead would be worse still: silent data loss on a `sky stop`.

    The message is checked for meaning, not phrasing. An earlier regex required
    the word "stop" to appear *before* the negation, which rejected the
    perfectly natural "Lyceum does not support stopping instances; use sky
    down". What must hold is that the message names the operation and says it
    is unavailable, in whatever order reads best.
    """
    fake = FakeLyceumClient(mixed_vms)
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(NotImplementedError) as exc_info:
        instance.stop_instances(CLUSTER_ON_CLOUD)

    message = str(exc_info.value).lower()
    assert 'stop' in message, (
        f'the message never mentions stopping: {message!r}')
    assert any(phrase in message for phrase in (
        'not support', 'unsupported', 'no support', 'does not', "doesn't",
        'cannot', "can't", 'not available', 'unavailable', 'no stop')), (
            f'the message does not say the operation is unavailable: {message!r}')

    assert fake.terminate_calls == []


# --------------------------------------------------------------------------
# terminate_instances
# --------------------------------------------------------------------------


def test_terminate_instances_terminates_every_live_vm(monkeypatch, vm):
    """A VM left behind bills until the reaper catches it -- up to $63.92/h.

    There is no cloud-side TTL (C5), so an explicit DELETE is the only thing
    that stops the meter.
    """
    first = vm('vm_status_ready_bare_ip',
               display_name=CLUSTER_ON_CLOUD,
               vm_id='live-1')
    second = vm('vm_status_ready_host_port',
                display_name=CLUSTER_ON_CLOUD,
                vm_id='live-2')
    fake = FakeLyceumClient([first, second])
    _patch_lyceum_client(monkeypatch, fake)

    instance.terminate_instances(CLUSTER_ON_CLOUD)

    assert set(fake.terminate_calls) == {'live-1', 'live-2'}


def test_terminate_instances_never_touches_another_cluster(
        monkeypatch, mixed_vms, prefix_decoy):
    """Terminating a stranger's VM destroys someone else's running job.

    display_name is the only handle Lyceum offers; a prefix match, a substring
    match, or a forgotten filter would take
    sky-cluster-abcd down with sky-cluster-abc.

    `loose_find` is what gives that claim teeth. The contract-honouring fake
    matched display_name exactly and dropped terminal VMs, so the provisioner
    could never *see* a near-miss and the test passed no matter what it did.
    Here the client hands back the sloppy superset and the provisioner's own
    filtering is what is under test -- justified because this is the one
    operation in the package that destroys someone else's running job, and the
    guard is a single `==`.
    """
    live, other = mixed_vms[1], mixed_vms[2]
    fake = FakeLyceumClient(mixed_vms + [prefix_decoy], loose_find=True)
    _patch_lyceum_client(monkeypatch, fake)

    instance.terminate_instances(CLUSTER_ON_CLOUD)

    assert other.vm_id not in fake.terminate_calls
    assert prefix_decoy.vm_id not in fake.terminate_calls, (
        'terminated sky-cluster-abcd, a different live cluster, by matching '
        'display_name as a prefix instead of exactly')
    assert live.vm_id in fake.terminate_calls
    # The terminated namesake may be DELETEd again (idempotent, harmless); no
    # VM outside this cluster's name may be.
    assert set(fake.terminate_calls) <= {v.vm_id for v in mixed_vms[:2]}


def test_terminate_instances_is_idempotent_when_nothing_is_live(monkeypatch):
    """`sky down` on an already-gone cluster must succeed, not error.

    SkyPilot's failover path tears down after every failed attempt, including
    ones that created nothing; raising there converts a recoverable failover
    into StopFailoverError.
    """
    fake = FakeLyceumClient([])
    _patch_lyceum_client(monkeypatch, fake)

    assert instance.terminate_instances(CLUSTER_ON_CLOUD) is None
    assert fake.terminate_calls == []


def test_terminate_instances_swallows_not_found(monkeypatch, vm, fixture):
    """Already gone is success; a raised 404 aborts teardown mid-cluster.

    Two callers can race on `sky down`, and the reaper races with autodown by
    design. Propagating the 404 leaves any remaining VM un-terminated.
    """
    detail = fixture('error_404_vm_not_found')['detail']
    live = vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD)
    fake = FakeLyceumClient([live],
                            terminate_error=api.LyceumNotFoundError(detail))
    _patch_lyceum_client(monkeypatch, fake)

    assert instance.terminate_instances(CLUSTER_ON_CLOUD) is None
    assert fake.terminate_calls == [live.vm_id]


# --------------------------------------------------------------------------
# open_ports / cleanup_ports / bootstrap_instances
# --------------------------------------------------------------------------


@pytest.mark.parametrize('fn_name', ['open_ports', 'cleanup_ports'])
@pytest.mark.parametrize('ports', [[], ['22'], ['8080', '10000-10010']])
def test_port_hooks_are_no_ops(monkeypatch, fn_name, ports):
    """Lyceum exposes no firewall API; raising here fails an otherwise-good launch.

    `OPEN_PORTS_VERSION = LAUNCH_ONLY` means the backend calls open_ports on
    every launch. The in-tree Shadeform provider raises NotImplementedError
    from open_ports, which is exactly the trap being avoided.
    """
    fake = FakeLyceumClient([])
    _patch_lyceum_client(monkeypatch, fake)
    fn = getattr(instance, fn_name)

    assert fn(CLUSTER_ON_CLOUD, ports) is None
    assert fn(CLUSTER_ON_CLOUD, ports, {'region': REGION}) is None
    assert fake.terminate_calls == []
    assert fake.create_calls == []


def test_bootstrap_instances_returns_the_config_unchanged():
    """Lyceum has no VPC, security groups, or SSH-key registry to set up.

    The returned config is what run_instances is then called with, so dropping
    or rewriting a field here silently changes what gets provisioned.
    """
    config = _make_provision_config()

    result = config_lib.bootstrap_instances(REGION, CLUSTER, config)

    assert result == config
    assert result.node_config == config.node_config
    assert result.authentication_config == config.authentication_config


# --------------------------------------------------------------------------
# Transient API failures during the polling loop (found in production)
# --------------------------------------------------------------------------
class _FlakyReadClient(FakeLyceumClient):
    """Raises on the first `fail_reads` read calls, then behaves normally.

    Models the real thing: Lyceum returned HTTP 500 from `/vms/list` and
    `/vms/availability` twice within ten minutes during phase-3 smoke testing.
    """

    def __init__(self, vms, error, fail_reads, **kw):
        super().__init__(vms, **kw)
        self._error = error
        self._fail_reads = fail_reads
        self.read_attempts = 0

    def _maybe_fail(self):
        self.read_attempts += 1
        if self.read_attempts <= self._fail_reads:
            raise self._error

    def list_vms(self, **kw):
        self._maybe_fail()
        return super().list_vms(**kw)

    def find_vms_by_display_name(self, name):
        self._maybe_fail()
        return super().find_vms_by_display_name(name)


def test_wait_instances_survives_a_transient_5xx_while_polling(monkeypatch, vm):
    """A retryable 5xx mid-poll must not kill a provision that is going fine.

    THIS IS A PRODUCTION BUG, not a hypothetical: a real launch died with
    `LyceumServerError: GET /vms/list failed with HTTP 500` while the VM was
    healthy and had already been assigned an IP. The whole point of separating
    LyceumServerError from LyceumCapacityError (C7) is that the former is
    retryable -- so the poll loop has to actually retry it.
    """
    live = vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD,
              ip_address='203.0.113.12')
    fake = _FlakyReadClient([live], api.LyceumServerError('HTTP 500: Failed to list VMs'),
                            fail_reads=3)
    _patch_lyceum_client(monkeypatch, fake)

    instance.wait_instances(REGION, CLUSTER_ON_CLOUD, status_lib.ClusterStatus.UP)

    assert fake.read_attempts > 3, (
        'the loop gave up on the first transient failure instead of retrying')


def test_wait_instances_still_gives_up_on_a_persistent_5xx(monkeypatch, vm):
    """Retrying must be bounded, and must surface as ProvisionerError.

    An unbounded retry would hold a GPU on the meter with no cloud-side TTL to
    stop it (C5). The type matters too: `provisioner.py` catches RuntimeError to
    drive its own retry, and a raw LyceumServerError is not one.
    """
    fake = _FlakyReadClient(
        [vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD)],
        api.LyceumServerError('HTTP 500'), fail_reads=10**6)
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises(common.ProvisionerError):
        instance.wait_instances(REGION, CLUSTER_ON_CLOUD, status_lib.ClusterStatus.UP)


def test_wait_instances_does_not_retry_an_auth_failure(monkeypatch, vm):
    """A bad credential is not transient; spinning on it just burns the budget.

    Retrying here would hold the provision open for the full timeout while a
    GPU bills, for an error that cannot possibly resolve itself.
    """
    fake = _FlakyReadClient(
        [vm('vm_status_ready_bare_ip', display_name=CLUSTER_ON_CLOUD)],
        api.LyceumAuthError('no key'), fail_reads=10**6)
    _patch_lyceum_client(monkeypatch, fake)

    with pytest.raises((api.LyceumAuthError, common.ProvisionerError)):
        instance.wait_instances(REGION, CLUSTER_ON_CLOUD, status_lib.ClusterStatus.UP)
    assert fake.read_attempts <= 2, (
        f'auth error retried {fake.read_attempts} times; it can never succeed')
