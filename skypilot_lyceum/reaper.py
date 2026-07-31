"""Orphan reaper: the only thing that ends billing when the control plane forgets.

Lyceum has no stop and no cloud-side TTL (C5). A VM bills from `ready` until
someone issues DELETE -- at b300x8 that is $63.92/h -- and
the only actor that ever issues it is our own control plane. SkyPilot's autodown
covers the normal case. This covers the abnormal one: an API server that died
mid-provision, an executor that crashed between create and record, a cluster DB
lost with its volume.

This is destructive automation pointed at production, so it is built to refuse:

  * dry run by default -- terminating requires opting in
  * a VM belonging to a live cluster is never touched
  * a VM younger than the grace window is never touched (it may be provisioning)
  * a VM whose name is not SkyPilot-shaped is never touched (a human made it)
  * an UNKNOWN cluster set raises rather than treating "I don't know" as
    "nothing is running" -- see `UnknownClusterStateError`

Identity is `display_name == cluster_name_on_cloud`: Lyceum has no tags and no
server-side filtering, so the name is the only handle a cluster has.
"""
from __future__ import annotations

import dataclasses
import datetime
import re
from typing import Any, Iterable, List, Optional, Sequence, Set

from skypilot_lyceum import api

#: How old a VM must be before it can be considered abandoned. Provisioning was
#: measured at 130-221s and `wait_instances` allows 900s, so anything inside
#: that window may legitimately not be in the cluster DB yet. An hour is far
#: past it: the cost of waiting is one extra hour of a VM that was leaked
#: anyway, and the cost of being wrong is a deleted training run.
DEFAULT_GRACE_SECONDS = 3600

#: Grace for a cluster stuck in AUTOSTOPPING. Much shorter, because the
#: situation is the opposite of the one above: the job ended, autostop fired on
#: schedule, and the teardown then failed. Nothing is pending and nothing else
#: will ever collect it -- on Lyceum the node CANNOT self-terminate, since the
#: skylet never calls `plugins.load_plugins()` and so has no registered
#: provisioner for an out-of-tree cloud. Ten minutes is enough to let a
#: genuinely-in-flight teardown finish; waiting the full hour is pure burn, and
#: at b300x8 that hour is $63.92.
STUCK_GRACE_SECONDS = 600

#: SkyPilot builds `cluster_name_on_cloud` as `<name>-<8 hex user hash>`. Only
#: names of that shape are candidates: anything else is a VM someone created by
#: hand in the Lyceum dashboard, which is none of this tool's business.
_SKYPILOT_NAME_RE = re.compile(r'^.+-[0-9a-f]{8}$')


class UnknownClusterStateError(RuntimeError):
    """We could not establish which clusters are live, so we refuse to act.

    The failure mode this prevents is total: if the cluster DB read fails and
    yields an empty set, EVERY live VM looks like an orphan and the reaper
    deletes the whole fleet mid-training. "I don't know" must never be
    silently equal to "nothing is running".
    """


@dataclasses.dataclass
class ReapResult:
    """What the reaper saw and did. `would_terminate` is populated in both
    modes; `terminated`/`failed` only when actually terminating."""
    scanned: int = 0
    would_terminate: List[str] = dataclasses.field(default_factory=list)
    terminated: List[str] = dataclasses.field(default_factory=list)
    failed: List[Any] = dataclasses.field(default_factory=list)
    dry_run: bool = True


def _created_at_epoch(vm: 'api.VM') -> Optional[float]:
    """Parse `created_at` to a UTC epoch, or None if absent/unparseable.

    A VM whose age cannot be established is treated as too young to reap by the
    caller -- unknown age must not license deletion.
    """
    raw = vm.created_at
    if not raw:
        return None
    text = raw.strip().replace('Z', '+00:00')
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def looks_like_a_skypilot_cluster(display_name: Optional[str]) -> bool:
    """True if `display_name` has the shape SkyPilot gives cluster_name_on_cloud."""
    return bool(display_name) and bool(_SKYPILOT_NAME_RE.match(display_name))


def select_orphans(vms: Iterable['api.VM'], known_names: Optional[Set[str]],
                   now: float,
                   grace_seconds: int = DEFAULT_GRACE_SECONDS,
                   stuck_names: Optional[Set[str]] = None
                   ) -> List['api.VM']:
    """VMs that are billing, SkyPilot-shaped, old enough, and unaccounted for.

    `known_names` is the set of `cluster_name_on_cloud` values the control plane
    believes are live. **None means "could not determine"** and raises; pass an
    empty set only when you have genuinely established there are no clusters.

    `stuck_names` are known clusters whose teardown has jammed -- in practice,
    stuck in AUTOSTOPPING. Being known is normally enough to be safe, but that
    stopped being true on Lyceum: the skylet fires autostop on schedule, the
    cluster moves to AUTOSTOPPING, and then the node tries
    `terminate_instances('lyceum', ...)`, which it cannot do unless this package
    is installed there. Observed live -- a VM billed for 30+ minutes after its
    job was killed, while remaining 'known' and therefore untouchable by the
    plain orphan rule. Membership here re-opens a VM to collection, still
    subject to the grace window.
    """
    if known_names is None:
        raise UnknownClusterStateError(
            'refusing to select orphans without a known cluster set: an '
            'unknown set would make every live VM look abandoned and delete '
            'the whole fleet. Fix the cluster-state read and retry.')

    orphans = []
    for vm in vms:
        if vm.is_terminal:
            continue                      # already dead, bills nothing (C4)
        if not looks_like_a_skypilot_cluster(vm.display_name):
            continue                      # someone's hand-made VM
        is_stuck = vm.display_name in (stuck_names or frozenset())
        if vm.display_name in known_names and not is_stuck:
            continue                      # live cluster -- real work
        # A stuck cluster is finished and nothing else will collect it, so it
        # waits minutes rather than the hour an in-flight provision needs.
        applicable_grace = STUCK_GRACE_SECONDS if is_stuck else grace_seconds
        created = _created_at_epoch(vm)
        if created is None or (now - created) < applicable_grace:
            continue                      # may still be provisioning
        orphans.append(vm)
    return orphans


def describe(vms: Sequence['api.VM'], now: float) -> List[str]:
    """One audit line per VM, so an operator can check before anything dies."""
    lines = []
    for vm in vms:
        created = _created_at_epoch(vm)
        age = f'{(now - created) / 3600:.1f}h' if created else 'unknown age'
        lines.append(
            f'{vm.display_name} vm_id={vm.vm_id} '
            f'{vm.hardware_profile}.{vm.gpu_count}x {vm.instance_type} '
            f'status={vm.status} age={age} ip={vm.ip}')
    return lines


def reap(client, known_names: Optional[Set[str]], now: float,
         grace_seconds: int = DEFAULT_GRACE_SECONDS,
         dry_run: bool = True,
         stuck_names: Optional[Set[str]] = None) -> ReapResult:
    """Find and (optionally) terminate orphaned Lyceum VMs.

    Dry run by DEFAULT: destroying GPUs is opt-in, so a misconfigured cron, a
    stale deployment or an accidental invocation reports instead of deleting.

    A failed listing propagates rather than being treated as an empty fleet --
    the Lyceum API demonstrably returns 500s, and acting on a partial view is
    how a safety net becomes an outage. A failed *termination* is recorded and
    the remaining VMs are still attempted: each one is money.
    """
    vms = client.list_vms(include_terminated=False)
    orphans = select_orphans(vms, known_names, now, grace_seconds,
                             stuck_names=stuck_names)
    result = ReapResult(scanned=len(vms),
                        would_terminate=[vm.vm_id for vm in orphans],
                        dry_run=dry_run)
    if dry_run:
        return result
    for vm in orphans:
        try:
            client.terminate_vm(vm.vm_id)
            result.terminated.append(vm.vm_id)
        except Exception as e:  # noqa: BLE001 - one failure must not strand the rest
            result.failed.append((vm.vm_id, str(e)))
    return result
