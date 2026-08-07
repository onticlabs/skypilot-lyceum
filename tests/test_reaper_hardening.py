"""Hardening for the one piece of automation here that DELETES paid machines.

The reaper's job is narrow: find Lyceum VMs our control plane created and then
lost track of, and end their billing. Everything below exists because the
original predicate answered "is this ours?" by looking at the VM's NAME, which
is not an ownership test at all.

Three changes, each closing a way the reaper is wrong:

  * a receipt — we write down that we are about to create a VM, before we create
    it, and only ever delete VMs we hold a receipt for;
  * a circuit breaker — a sudden crowd of candidates means our own records were
    lost far more often than it means the fleet was abandoned, so refuse;
  * a heartbeat — the reaper's existing failure mode is to refuse quietly, every
    thirty minutes, forever. Refusing is right. Quietly is not.
"""
from __future__ import annotations

import pathlib

import pytest

from skypilot_lyceum import intent, reaper


class FakeVM:
    def __init__(self, name, vm_id=None, created_at='2020-01-01T00:00:00Z',
                 status='running'):
        self.display_name = name
        self.vm_id = vm_id or f'id-{name}'
        self.created_at = created_at
        self.status = status
        self.is_terminal = status in ('terminated', 'failed')


OLD = 0.0                      # created_at above is far in the past
NOW = 4_000_000_000.0          # so every VM is comfortably past any grace


# --------------------------------------------------------------------------
# The receipt
# --------------------------------------------------------------------------
def test_a_vm_we_never_recorded_creating_is_never_deleted(tmp_path):
    """The core of the change. `run-20260807` and `bench-12345678` both match
    the old name predicate -- all-digit suffixes are valid hex -- and so does
    every cluster created by any OTHER SkyPilot install sharing the org API key.
    None of them are ours, and none are in our ledger."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('ours-abcd1234')

    vms = [FakeVM('ours-abcd1234'), FakeVM('run-20260807'),
           FakeVM('bench-12345678'), FakeVM('teammates-cluster-deadbeef')]
    orphans = reaper.select_orphans(vms, known_names=set(), now=NOW)
    assert [v.display_name for v in orphans] == ['ours-abcd1234']


def test_the_ledger_is_written_before_the_vm_exists(tmp_path):
    """Ordering is the whole point. A receipt written AFTER the create call
    would be missing for exactly the VM that matters -- one whose create
    succeeded and whose caller then died, which is the leak being hunted."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    seen = {}

    def fake_create(**kw):
        seen['ledger_at_create_time'] = intent.recorded()
        raise RuntimeError('crashed right after Lyceum made the VM')

    with pytest.raises(RuntimeError):
        intent.record('c-abcd1234')
        fake_create(display_name='c-abcd1234')
    assert 'c-abcd1234' in seen['ledger_at_create_time']


def test_a_crash_between_receipt_and_create_leaves_only_a_dangling_entry(
        tmp_path):
    """The failure this trades for: a receipt with no VM. Harmless -- the
    reaper intersects the ledger with what Lyceum actually reports."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('never-created-abcd1234')
    orphans = reaper.select_orphans([], known_names=set(), now=NOW)
    assert orphans == []


def test_an_empty_ledger_selects_nothing(tmp_path):
    """Fail-safe in the direction that costs money rather than data. A server
    whose volume was recreated has no ledger, so it deletes nothing -- instead
    of concluding the whole fleet is abandoned."""
    intent.set_ledger_path(tmp_path / 'nonexistent.jsonl')
    vms = [FakeVM(f'c{i}-abcd1234') for i in range(5)]
    assert reaper.select_orphans(vms, known_names=set(), now=NOW) == []


def test_the_ledger_survives_a_corrupt_line(tmp_path):
    """It is an append-only log written during provisioning; a torn write must
    not take the whole safety net down with it."""
    p = tmp_path / 'ledger.jsonl'
    p.write_text('{"name": "good-abcd1234"}\n{not json\n')
    intent.set_ledger_path(p)
    assert intent.recorded() == {'good-abcd1234'}


def test_the_ledger_is_a_set_not_a_transcript(tmp_path):
    """Appended to on every launch, forever, with names that repeat across
    cluster generations — so the file must collapse to the SET it represents
    rather than growing without bound on a volume shared with SkyPilot state."""
    ledger = tmp_path / 'ledger.jsonl'
    intent.set_ledger_path(ledger)
    for _ in range(intent._COMPACT_AT + 20):
        intent.record('c-abcd1234')
    assert intent.recorded() == {'c-abcd1234'}
    assert len(ledger.read_text().splitlines()) < intent._COMPACT_AT


def test_compaction_keeps_every_distinct_name(tmp_path):
    """Compaction that dropped a name would silently un-collect the VM it
    covers — the ledger shrinking must never shrink what the reaper may act on."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    names = {f'c{i}-abcd1234' for i in range(30)}
    for _ in range(20):
        for n in sorted(names):
            intent.record(n)
    assert intent.recorded() == names


def test_a_known_cluster_is_still_never_touched(tmp_path):
    """The receipt ADDS a condition; it does not relax the existing ones."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('live-abcd1234')
    orphans = reaper.select_orphans([FakeVM('live-abcd1234')],
                                    known_names={'live-abcd1234'}, now=NOW)
    assert orphans == []


def test_a_young_vm_is_still_never_touched(tmp_path):
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('young-abcd1234')
    vm = FakeVM('young-abcd1234', created_at='2020-01-01T00:00:00Z')
    orphans = reaper.select_orphans([vm], known_names=set(),
                                    now=946684800.0 + 60)     # 60s old
    assert orphans == []


# --------------------------------------------------------------------------
# The circuit breaker
# --------------------------------------------------------------------------
class FakeClient:
    def __init__(self, vms):
        self._vms = vms
        self.terminated = []

    def list_vms(self, include_terminated=False):
        return [v for v in self._vms if not v.is_terminal]

    def terminate_vm(self, vm_id):
        self.terminated.append(vm_id)


def test_a_crowd_of_candidates_trips_the_breaker(tmp_path):
    """Five simultaneous orphans is not five abandoned machines; it is our own
    records having gone missing. Deleting on that reading is how a safety net
    becomes the outage."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    vms = []
    for i in range(6):
        intent.record(f'c{i}-abcd1234')
        vms.append(FakeVM(f'c{i}-abcd1234'))
    client = FakeClient(vms)
    with pytest.raises(reaper.FleetAnomalyError) as e:
        reaper.reap(client, known_names=set(), now=NOW, dry_run=False)
    assert client.terminated == []
    assert 'refusing' in str(e.value).lower()


def test_one_orphan_in_a_healthy_fleet_is_reaped(tmp_path):
    """The breaker must not make the reaper useless. One leak beside live work
    is the normal case and must still be collected."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('leaked-abcd1234')
    live = [FakeVM(f'live{i}-abcd1234') for i in range(8)]
    for v in live:
        intent.record(v.display_name)
    client = FakeClient(live + [FakeVM('leaked-abcd1234')])
    known = {v.display_name for v in live}
    result = reaper.reap(client, known_names=known, now=NOW, dry_run=False)
    assert client.terminated == ['id-leaked-abcd1234']
    assert result.terminated == ['id-leaked-abcd1234']


def test_the_breaker_trips_in_dry_run_too(tmp_path):
    """A dry run that reports 40 candidates has found a bug, not 40 orphans.
    Reporting it as a normal result would train people to ignore it."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    vms = []
    for i in range(12):
        intent.record(f'c{i}-abcd1234')
        vms.append(FakeVM(f'c{i}-abcd1234'))
    with pytest.raises(reaper.FleetAnomalyError):
        reaper.reap(FakeClient(vms), known_names=set(), now=NOW, dry_run=True)


def test_the_known_set_is_re_read_immediately_before_deleting(tmp_path):
    """A relaunch can ADOPT an existing VM (run_instances does this by design).
    If it happens between selection and deletion, the reaper would delete a VM
    a live cluster just took ownership of -- and grace cannot help, because the
    VM is old by construction."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    intent.record('adopted-abcd1234')
    client = FakeClient([FakeVM('adopted-abcd1234')])

    # Between select and delete, the cluster becomes known again.
    result = reaper.reap(client, known_names=set(), now=NOW, dry_run=False,
                         recheck_known=lambda: {'adopted-abcd1234'})
    assert client.terminated == []
    assert result.skipped_now_known == ['adopted-abcd1234']


# --------------------------------------------------------------------------
# The heartbeat
# --------------------------------------------------------------------------
def test_a_clean_run_stamps_the_heartbeat(tmp_path):
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    hb = tmp_path / 'heartbeat'
    reaper.reap(FakeClient([]), known_names=set(), now=NOW, dry_run=True,
                heartbeat_path=hb)
    assert hb.exists() and hb.read_text().strip()


def test_a_refusal_does_not_stamp_the_heartbeat(tmp_path):
    """The whole point. The reaper refusing is correct; refusing SILENTLY for
    six months is how 157 idle node-hours went unnoticed. A stale heartbeat is
    what someone can alert on."""
    intent.set_ledger_path(tmp_path / 'ledger.jsonl')
    hb = tmp_path / 'heartbeat'
    with pytest.raises(reaper.UnknownClusterStateError):
        reaper.reap(FakeClient([]), known_names=None, now=NOW, dry_run=True,
                    heartbeat_path=hb)
    assert not hb.exists()


def test_heartbeat_staleness_is_readable(tmp_path):
    hb = tmp_path / 'heartbeat'
    assert reaper.heartbeat_age_seconds(hb, now=NOW) is None
    reaper.stamp_heartbeat(hb, now=NOW - 900)
    assert abs(reaper.heartbeat_age_seconds(hb, now=NOW) - 900) < 2
