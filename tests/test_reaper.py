"""Orphan reaper: the only thing that ends billing when the control plane forgets.

Lyceum has no stop and no cloud-side TTL (C5). A VM bills until someone issues
DELETE, and at b300x8 that is $63.92/h. SkyPilot's autodown covers the normal
case; this covers the abnormal one -- an API server that died mid-provision, a
crashed executor, a cluster record lost with the volume.

This module is DESTRUCTIVE AUTOMATION pointed at production infrastructure, so
most of these tests are about what it must REFUSE to do. The happy path is one
test; the refusals are the rest.
"""
from __future__ import annotations

import pytest
from skypilot_lyceum import api, reaper


def _vm(name, *, status='ready', age_s=7200, vm_id=None):
    """A VM `age_s` seconds old. Times are ISO-8601 Z, as the API returns."""
    import datetime
    created = (datetime.datetime(2026, 7, 31, 12, 0, 0,
                                 tzinfo=datetime.timezone.utc) -
               datetime.timedelta(seconds=age_s))
    return api.VM(
        vm_id=vm_id or f'id-{name}', status=status, display_name=name,
        hardware_profile='l40s', gpu_count=1, instance_type='on-demand',
        created_at=created.isoformat().replace('+00:00', 'Z'),
        ip='1.2.3.4', ssh_port=22, raw={})


NOW = 1785500000.0  # matches the _vm base instant; tests pass it explicitly


def _now():
    import datetime
    return datetime.datetime(2026, 7, 31, 12, 0, 0,
                             tzinfo=datetime.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------
def test_a_vm_belonging_to_a_live_cluster_is_never_reaped():
    """The whole point: do not kill work that is running.

    `display_name == cluster_name_on_cloud`, so membership is
    an exact set lookup. Getting this wrong destroys a running training job.
    """
    vms = [_vm('sky-train-abc12345')]
    orphans = reaper.select_orphans(vms, known_names={'sky-train-abc12345'},
                                    now=_now())
    assert orphans == []


def test_a_vm_younger_than_the_grace_window_is_never_reaped():
    """A VM mid-provision is not yet in the cluster DB.

    Provisioning took 130-221s in measurement and the provisioner allows 900s.
    Reaping inside that window would kill launches at random, and the reaper
    would look like flaky infrastructure rather than a safety net.
    """
    vms = [_vm('sky-new-deadbeef', age_s=60)]
    assert reaper.select_orphans(vms, known_names=set(), now=_now()) == []


def test_a_vm_that_does_not_look_like_a_skypilot_cluster_is_never_reaped():
    """Never touch a machine a human made by hand in the dashboard.

    SkyPilot appends a user hash, so `cluster_name_on_cloud` ends in
    `-<8 hex>`. Anything else is somebody's manual VM, and terminating it would
    be both destructive and completely outside this tool's remit.
    """
    vms = [_vm('someones-debug-box'), _vm('scratch'), _vm('test-vm-2')]
    assert reaper.select_orphans(vms, known_names=set(), now=_now()) == []


def test_already_terminal_vms_are_ignored():
    """Terminated VMs linger in /vms/list forever (C4) and bill nothing."""
    vms = [_vm('sky-old-aaaaaaaa', status='terminated'),
           _vm('sky-bad-bbbbbbbb', status='failed'),
           _vm('sky-err-cccccccc', status='error')]
    assert reaper.select_orphans(vms, known_names=set(), now=_now()) == []


def test_an_unknown_cluster_set_refuses_to_reap_anything():
    """THE catastrophic case. An empty set must never mean 'reap everything'.

    If the cluster DB read fails and yields an empty or None set, every live VM
    looks like an orphan and the reaper deletes the entire fleet mid-training.
    `known_names=None` means 'I could not find out' and must raise, never
    proceed. This is the single most dangerous line in the package.
    """
    with pytest.raises(reaper.UnknownClusterStateError):
        reaper.select_orphans([_vm('sky-x-aaaaaaaa')], known_names=None,
                              now=_now())


# ---------------------------------------------------------------------------
# The one thing it must actually do
# ---------------------------------------------------------------------------
def test_an_old_unknown_skypilot_vm_is_an_orphan():
    """The case this exists for: control plane forgot, VM still billing."""
    orphan = _vm('sky-lost-ada52f01', age_s=7200)
    got = reaper.select_orphans([orphan], known_names={'sky-other-11111111'},
                                now=_now())
    assert [v.display_name for v in got] == ['sky-lost-ada52f01']


def test_selection_is_reported_with_enough_context_to_audit():
    """An operator must be able to see WHY something was chosen, before it dies."""
    orphan = _vm('sky-lost-ada52f01', age_s=7200)
    lines = reaper.describe(reaper.select_orphans(
        [orphan], known_names=set(), now=_now()), now=_now())
    joined = '\n'.join(lines)
    assert 'sky-lost-ada52f01' in joined
    assert 'id-sky-lost-ada52f01' in joined
    assert 'l40s' in joined


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class _FakeClient:
    def __init__(self, vms, fail_list=False, fail_terminate=False):
        self.vms = list(vms)
        self.terminated = []
        self.fail_list = fail_list
        self.fail_terminate = fail_terminate

    def list_vms(self, include_terminated=False):
        if self.fail_list:
            raise api.LyceumServerError('HTTP 500')
        return list(self.vms)

    def terminate_vm(self, vm_id):
        if self.fail_terminate:
            raise api.LyceumServerError('HTTP 500')
        self.terminated.append(vm_id)


def test_reap_is_dry_run_by_default():
    """Destructive automation must opt IN to destroying things.

    A reaper that terminates by default turns a config mistake, a stale
    deployment, or an accidental invocation into deleted GPUs.
    """
    client = _FakeClient([_vm('sky-lost-ada52f01', age_s=7200)])
    result = reaper.reap(client, known_names=set(), now=_now())
    assert client.terminated == []
    assert result.would_terminate and not result.terminated


def test_reap_terminates_only_when_explicitly_told_to():
    client = _FakeClient([_vm('sky-lost-ada52f01', age_s=7200)])
    result = reaper.reap(client, known_names=set(), now=_now(), dry_run=False)
    assert client.terminated == ['id-sky-lost-ada52f01']
    assert result.terminated == ['id-sky-lost-ada52f01']


def test_a_failed_listing_reaps_nothing():
    """No information is not the same as 'nothing is running'.

    If /vms/list is down -- and it demonstrably goes down (two endpoints 500'd
    within ten minutes during phase-3 smoke) -- the reaper must do nothing at
    all rather than act on a partial or empty view.
    """
    client = _FakeClient([_vm('sky-lost-ada52f01', age_s=7200)], fail_list=True)
    with pytest.raises(api.LyceumServerError):
        reaper.reap(client, known_names=set(), now=_now(), dry_run=False)
    assert client.terminated == []


def test_one_failed_termination_does_not_abandon_the_rest():
    """Each leaked VM is money; a failure on one must not strand the others."""
    client = _FakeClient([_vm('sky-a-aaaaaaaa', age_s=7200, vm_id='a'),
                          _vm('sky-b-bbbbbbbb', age_s=7200, vm_id='b')],
                         fail_terminate=True)
    result = reaper.reap(client, known_names=set(), now=_now(), dry_run=False)
    assert len(result.failed) == 2, 'both attempts should be recorded'


# ---------------------------------------------------------------------------
# Clusters stuck in AUTOSTOPPING (found in production, phase 3/5)
# ---------------------------------------------------------------------------
def test_a_cluster_stuck_autostopping_is_reapable():
    """The failure mode that made autodown useless on Lyceum.

    Observed live: the skylet fired autostop on schedule, the cluster went to
    AUTOSTOPPING, and the node then tried `terminate_instances('lyceum', ...)`
    -- which it cannot do, because SkyPilot never ships out-of-tree plugins to
    ordinary cluster nodes. The VM billed for 30+ minutes after its job was
    killed, and the plain orphan rule could not touch it: the cluster is still
    in SkyPilot's DB, so its name IS 'known'.

    So a known name is no longer automatically safe. A cluster that has been
    trying to die past the grace window is exactly what must be collected.
    """
    vm = _vm('sky-stuck-ada52f01', age_s=7200)
    got = reaper.select_orphans([vm], known_names={'sky-stuck-ada52f01'},
                                now=_now(),
                                stuck_names={'sky-stuck-ada52f01'})
    assert [v.display_name for v in got] == ['sky-stuck-ada52f01']


def test_a_healthy_known_cluster_is_still_never_reaped():
    """The stuck rule must not widen the blast radius.

    A cluster that is merely known -- running real work -- stays untouchable.
    Only membership in `stuck_names` moves it.
    """
    vm = _vm('sky-busy-ada52f01', age_s=7200)
    assert reaper.select_orphans([vm], known_names={'sky-busy-ada52f01'},
                                 now=_now(), stuck_names=set()) == []


def test_a_recently_stuck_cluster_is_left_alone():
    """AUTOSTOPPING is a legitimate transient state for a few seconds.

    Reaping the instant it appears would race a teardown that is about to
    succeed on its own, and on a cloud that CAN self-terminate it always will.
    """
    vm = _vm('sky-stuck-ada52f01', age_s=120)
    assert reaper.select_orphans([vm], known_names={'sky-stuck-ada52f01'},
                                 now=_now(),
                                 stuck_names={'sky-stuck-ada52f01'}) == []


def test_stuck_names_defaults_to_empty_so_callers_opt_in():
    """An old caller must not suddenly start reaping live clusters."""
    vm = _vm('sky-busy-ada52f01', age_s=7200)
    assert reaper.select_orphans([vm], known_names={'sky-busy-ada52f01'},
                                 now=_now()) == []


def test_a_stuck_cluster_gets_a_shorter_grace_than_an_unknown_one():
    """A cluster stuck AUTOSTOPPING is DEFINITIVELY finished, so wait less.

    The one-hour grace exists for a VM that might still be provisioning and not
    yet in the cluster DB. A stuck cluster is the opposite case: the job ended,
    autostop fired on schedule, and the teardown then failed. There is nothing
    left to wait for, and on Lyceum nothing else will ever collect it -- the
    node cannot self-terminate, because the skylet never loads plugins.

    Waiting a full hour here is pure burn: at b300x8 that is $63.92.
    """
    vm = _vm('sky-stuck-ada52f01', age_s=reaper.STUCK_GRACE_SECONDS + 60)
    got = reaper.select_orphans([vm], known_names={'sky-stuck-ada52f01'},
                                now=_now(), stuck_names={'sky-stuck-ada52f01'})
    assert [v.display_name for v in got] == ['sky-stuck-ada52f01']
    assert reaper.STUCK_GRACE_SECONDS < reaper.DEFAULT_GRACE_SECONDS


def test_an_unknown_vm_still_gets_the_full_grace():
    """The shorter stuck-grace must not leak into the provisioning case.

    An unknown VM may be mid-provision (measured 130-340s, timeout 900s); the
    long grace is what stops the reaper killing launches at random.
    """
    vm = _vm('sky-new-ada52f01', age_s=reaper.STUCK_GRACE_SECONDS + 60)
    assert reaper.select_orphans([vm], known_names=set(), now=_now()) == []
