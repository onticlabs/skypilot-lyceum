"""A receipt, written before we ask Lyceum to create a VM.

The reaper deletes machines. It therefore needs to answer "is this VM ours?"
correctly, and the honest answer cannot be derived from the VM itself: Lyceum
has no tags, no owner field, and no server-side filtering, so all the reaper
ever sees is a name. Names were the old test -- anything shaped like
`<something>-<8 hex>` was assumed to be a SkyPilot cluster of ours. That is not
an ownership test. It matches `run-20260807` (all digits are valid hex), and it
matches every cluster created by any OTHER SkyPilot installation sharing the
organisation's API key, whose clusters live in that installation's database and
are invisible here.

So we write it down instead. Before the create call, the name goes into an
append-only ledger on the API server's persistent volume; the reaper will only
ever delete a VM whose name it finds there.

The ordering matters more than the storage. A receipt written after the create
would be missing for precisely the VM worth finding -- one Lyceum made and whose
caller then died. Writing first means the only inconsistency possible is a
receipt with no VM, which is inert: the reaper intersects the ledger with what
Lyceum actually reports.

An absent or unreadable ledger yields the empty set, which selects nothing. That
is deliberate: a server whose volume was recreated should leak money until
someone looks, not delete a fleet it has simply forgotten.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Optional, Set

logger = logging.getLogger(__name__)

#: On the API server this is the Fly volume mounted at /root/.sky, so the ledger
#: outlives a machine restart. Overridable for tests and for anyone running the
#: server somewhere else.
LEDGER_ENV = 'SKYPILOT_LYCEUM_LEDGER'

_DEFAULT_LEDGER = pathlib.Path('~/.sky/lyceum_intent.jsonl').expanduser()

_ledger_path: Optional[pathlib.Path] = None

#: Rewrite the file once it passes this many lines. Cluster names repeat across
#: generations, so an append-per-launch grows without bound while the SET it
#: represents stays small.
_COMPACT_AT = 500


def set_ledger_path(path) -> None:
    global _ledger_path
    _ledger_path = pathlib.Path(path) if path is not None else None


def ledger_path() -> pathlib.Path:
    if _ledger_path is not None:
        return _ledger_path
    return pathlib.Path(os.environ.get(LEDGER_ENV, _DEFAULT_LEDGER))


def record(cluster_name_on_cloud: str) -> None:
    """Note that we are about to create this VM. Call BEFORE the create.

    Best-effort by design. A ledger write that fails must not fail a launch --
    the consequence is one VM the reaper will decline to collect, which is the
    situation we are already in for every VM today. Failing the launch instead
    would trade a cost risk for an availability one.
    """
    path = ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(
                json.dumps({'name': cluster_name_on_cloud,
                            'ts': int(time.time())}) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        _maybe_compact(path)
    except OSError as exc:
        logger.warning(
            'could not record launch intent for %s (%s) — the orphan reaper '
            'will not be able to collect this VM if it leaks',
            cluster_name_on_cloud, exc)


def recorded() -> Set[str]:
    """Every name we have ever recorded an intent to create.

    Tolerates a corrupt line rather than failing: this is appended to during
    provisioning, so a torn write is a real possibility and must not take the
    safety net down with it. A line we cannot parse is one name we will decline
    to reap, never a name we wrongly reap.
    """
    path = ledger_path()
    names: Set[str] = set()
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return names
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            name = json.loads(line).get('name')
        except (ValueError, AttributeError):
            logger.warning('skipping unparseable intent-ledger line: %.80s', line)
            continue
        if name:
            names.add(name)
    return names


def _maybe_compact(path: pathlib.Path) -> None:
    """Collapse the log to one line per distinct name once it gets long.

    Rewrite-and-replace so a crash mid-compaction leaves either the old file or
    the new one, never a truncated ledger -- losing the ledger means losing the
    ability to collect anything it covered.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            lines = handle.readlines()
        if len(lines) < _COMPACT_AT:
            return
        latest = {}
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get('name'):
                latest[entry['name']] = entry
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as handle:
            for entry in latest.values():
                handle.write(json.dumps(entry) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning('intent ledger compaction failed (%s); continuing', exc)
