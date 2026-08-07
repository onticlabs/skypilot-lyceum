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
  * a VM this control plane has no RECORD of creating is never touched -- see
    `intent.py`. A name is not an ownership test: it cannot tell our cluster
    from a colleague's, or from another SkyPilot install sharing the org key
  * too many candidates at once raises rather than reaps: that pattern means we
    lost our own records, not that the fleet was abandoned
  * an UNKNOWN cluster set raises rather than treating "I don't know" as
    "nothing is running" -- see `UnknownClusterStateError`

Identity is `display_name == cluster_name_on_cloud`: Lyceum has no tags and no
server-side filtering, so the name is the only handle a cluster has.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import pathlib
import re
import time
from typing import Any, Iterable, List, Optional, Sequence, Set

from skypilot_lyceum import api, intent

logger = logging.getLogger(__name__)

#: How old a VM must be before it can be considered abandoned. Provisioning was
#: measured at 130-221s and `wait_instances` allows 900s, so anything inside
#: that window may legitimately not be in the cluster DB yet. An hour is far
#: past it: the cost of waiting is one extra hour of a VM that was leaked
#: anyway, and the cost of being wrong is a deleted training run.
DEFAULT_GRACE_SECONDS = 3600

#: Grace for a cluster stuck in AUTOSTOPPING. Much shorter, because the
#: situation is the opposite of the one above: the job ended, autostop fired on
#: schedule, and the teardown then failed. Nothing is pending.
#:
#: Node-side autodown now WORKS (see `node_autodown.py`, verified live
#: 2026-08-07), so reaching this state means that mechanism failed -- and every
#: way it can fail is silent: a `.pth` error is swallowed by `site.py`, and the
#: skylet swallows exceptions and retries every 60s. So this rule is not
#: obsolete now that autodown works; it is precisely the net beneath it.
#:
#: Ten minutes is enough to let a genuinely-in-flight teardown finish (the live
#: run completed in under three); waiting the full hour is pure burn, and at
#: b300x8 that hour is $63.92.
STUCK_GRACE_SECONDS = 600

#: SkyPilot builds `cluster_name_on_cloud` as `<name>-<8 hex user hash>`. Only
#: names of that shape are candidates: anything else is a VM someone created by
#: hand in the Lyceum dashboard, which is none of this tool's business.
_SKYPILOT_NAME_RE = re.compile(r'^.+-[0-9a-f]{8}$')


#: Refuse to act when candidates look like a fleet rather than a leak. Losing
#: our own records makes EVERY live VM look abandoned at once, and that reading
#: is far more likely than many machines being genuinely orphaned at the same
#: moment. Below the floor the absolute number is small enough to be real; above
#: it, a third of the fleet turning up as candidates is a bug in us.
BREAKER_FLOOR = 2
BREAKER_FRACTION = 1.0 / 3.0


class FleetAnomalyError(RuntimeError):
    """Too many candidates at once, so we refuse and ask for a human.

    Distinct from `UnknownClusterStateError`: there we know we cannot see the
    fleet, here we can see it and what we see does not look like a leak. Both
    end the same way -- nothing is deleted.
    """


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
    #: Candidates dropped at the last moment because a relaunch adopted them.
    skipped_now_known: List[str] = dataclasses.field(default_factory=list)
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

    ours = intent.recorded()
    orphans = []
    for vm in vms:
        if vm.is_terminal:
            continue                      # already dead, bills nothing (C4)
        if vm.display_name not in ours:
            # We hold no receipt for this VM, so it is not ours to delete: a
            # colleague's, another SkyPilot install's sharing the org key, or
            # one made by hand. The name shape cannot distinguish those -- see
            # `intent.py`. An empty ledger therefore selects nothing, which is
            # the safe direction when a volume has been recreated.
            continue
        if not looks_like_a_skypilot_cluster(vm.display_name):
            continue                      # belt and braces; the receipt leads
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
         stuck_names: Optional[Set[str]] = None,
         recheck_known=None,
         heartbeat_path=None) -> ReapResult:
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

    # Breaker BEFORE the dry-run return: a dry run reporting forty candidates
    # has found a bug, not forty orphans, and rendering that as an ordinary
    # result is how people learn to scroll past it.
    limit = max(BREAKER_FLOOR, int(len(vms) * BREAKER_FRACTION))
    if len(orphans) > limit:
        raise FleetAnomalyError(
            f'refusing to act: {len(orphans)} of {len(vms)} live VMs look '
            f'orphaned (limit {limit}). That pattern means this control plane '
            'lost its own records far more often than it means the fleet was '
            'abandoned. Nothing was terminated. Candidates: '
            f'{sorted(vm.display_name for vm in orphans)}')

    result = ReapResult(scanned=len(vms),
                        would_terminate=[vm.vm_id for vm in orphans],
                        dry_run=dry_run)
    if dry_run:
        _stamp(heartbeat_path, now)
        return result

    # Re-read the live set immediately before deleting. `run_instances` adopts
    # an existing VM with a matching name by design, so a relaunch between
    # selection and deletion would otherwise have its node deleted out from
    # under it -- and the grace window cannot help, because an adopted VM is
    # old by construction.
    if recheck_known is not None:
        try:
            fresh = recheck_known()
        except Exception:  # noqa: BLE001
            raise UnknownClusterStateError(
                'could not re-read the cluster set before terminating; '
                'refusing rather than acting on a stale view') from None
        if fresh is None:
            raise UnknownClusterStateError(
                'cluster set unavailable on re-read; nothing terminated')
        still = []
        for vm in orphans:
            if vm.display_name in fresh:
                result.skipped_now_known.append(vm.display_name)
            else:
                still.append(vm)
        orphans = still

    for vm in orphans:
        try:
            client.terminate_vm(vm.vm_id)
            result.terminated.append(vm.vm_id)
        except Exception as e:  # noqa: BLE001 - one failure must not strand the rest
            result.failed.append((vm.vm_id, str(e)))
    _stamp(heartbeat_path, now)
    return result


def stamp_heartbeat(path, now: Optional[float] = None) -> None:
    """Record that a reaping pass completed without refusing.

    The reaper's designed failure is to refuse, which is correct -- and until
    now it refused into a log line nobody reads, every thirty minutes,
    indefinitely. A file whose mtime stops advancing is something a check can
    see. Only a COMPLETED pass stamps it; every refusal path leaves it stale.
    """
    if path is None:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.time() if now is None else now
    path.write_text(f'{stamp:.0f}\n', encoding='utf-8')


def _stamp(path, now) -> None:
    try:
        stamp_heartbeat(path, now)
    except OSError as exc:      # evidence is a nicety; never break the reaper
        logger.warning('could not write reaper heartbeat (%s)', exc)


def heartbeat_age_seconds(path, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the last completed pass, or None if there has never been
    one. None is the loudest case, not the quietest: it means this reaper has
    not finished a single pass."""
    try:
        raw = pathlib.Path(path).read_text(encoding='utf-8').strip()
    except OSError:
        return None
    try:
        return (time.time() if now is None else now) - float(raw)
    except ValueError:
        return None
