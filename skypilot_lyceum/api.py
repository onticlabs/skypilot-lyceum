"""Sole owner of HTTP traffic to the Lyceum Cloud API.

Nothing else in this package may import `requests`. Every Lyceum API quirk
numbered C1-C12 (tabulated in README.md, each one observed against the live API
rather than read out of its documentation) is handled here or in
`provision/instance.py`; each has a fixture-backed regression test.
"""
from __future__ import annotations

import dataclasses
import datetime
import ipaddress
import os  # noqa: F401  -- used by read_api_key once implemented
import re
from typing import Any, Dict, List, Optional, Tuple

# Imported as a MODULE, and deliberately kept even while unused: `_request` must
# reach the transport as `requests.request(...)` so tests can swap it with
# `monkeypatch.setattr(api.requests, 'request', ...)`. A `from requests import
# request`, or a Session built once at import time, binds the callable early and
# defeats every mock in the suite. The noqa stops the linter deleting the import
# before the implementation lands -- it was silently stripped once already, which
# is exactly the failure tests/conftest.py's documented idiom depends on.
import requests  # noqa: F401

API_KEY_PATH = '~/.lyceum/api_key'
DEFAULT_BASE_URL = 'https://api.lyceum.technology'
API_PREFIX = '/api/v2/external'

#: Statuses that mean the VM is gone or unusable. A VM in one of these must
#: never be resolved as a live cluster member (C4: terminated VMs stay in
#: `/vms/list` forever and keep their display_name).
TERMINAL_STATUSES = frozenset({'terminated', 'failed', 'error'})

#: Statuses that mean provisioning finished. NOT sufficient on its own to treat
#: a node as usable -- see `VM.is_usable` (C10).
READY_STATUSES = frozenset({'ready', 'running'})

#: The full set of hardware profiles the API validates against, as enumerated by
#: its own 400 response. Kept as a constant so the catalog and the error path
#: agree on one list.
HARDWARE_PROFILES = ('a100', 'b200', 'b300', 'h100', 'h200', 'l40s')

#: GPU counts the API accepts. Anything else is rejected with an unhelpful 400
#: (C8), so validate client-side before spending a round trip.
ALLOWED_GPU_COUNTS = (1, 2, 4, 8)

#: Marks a 500 body as "no capacity" rather than a server fault (C7). Lyceum
#: signals capacity exhaustion with HTTP 500 and only this message distinguishes
#: it, so a generic retry-on-5xx client would back off against an exhausted SKU
#: instead of failing over to another cloud.
_CAPACITY_DETAIL_RE = re.compile(r'could not be provisioned', re.IGNORECASE)

#: Environment variable that overrides the key file. See README.md,
#: "Configuration".
API_KEY_ENV_VAR = 'LYCEUM_API_KEY'

#: Port used when `ip_address` carries no `:port` suffix (C2).
DEFAULT_SSH_PORT = 22

#: The two capacity/pricing pools. `instance_type` is always sent explicitly on
#: create so the bill never depends on a server-side default we did not choose.
ON_DEMAND = 'on-demand'
SPOT = 'spot'

#: `{instance_type}.{profile}.{count}x`, the key under `applies_to` (C3).
_PRICE_KEY_RE = re.compile(r'^(?P<instance_type>[^.]+)\.(?P<profile>[^.]+)\.'
                           r'(?P<gpu_count>\d+)x$')

#: The pricing meter that is compute rental. Storage/egress meters share the
#: endpoint and would otherwise be quoted to the optimizer as hourly VM prices.
_VM_METER_SLUG = 'vm_running'

#: Sorts before every real timestamp, so a VM whose `created_at` is missing or
#: unparseable is treated as the oldest generation rather than the newest.
_EPOCH = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)


class LyceumError(Exception):
    """Base class for every failure raised by this module."""


class LyceumAuthError(LyceumError):
    """Missing, malformed, or rejected credentials (401/403)."""


class LyceumNotFoundError(LyceumError):
    """The requested VM does not exist (404 `{"detail": "VM not found"}`)."""


class LyceumInvalidRequestError(LyceumError):
    """The request was well-formed but semantically rejected (400)."""


class LyceumCapacityError(LyceumError):
    """No capacity for the requested (profile, gpu_count, instance_type).

    Raised for the HTTP 500 documented in C7. Callers must treat this as
    "fail over now", never as a retryable transport error.
    """


class LyceumServerError(LyceumError):
    """A genuine 5xx that is NOT capacity exhaustion. Retryable."""


@dataclasses.dataclass(frozen=True)
class VM:
    """One Lyceum VM, normalised.

    `vm_id` is opaque: the API returns undashed hex for some VMs and dashed
    UUIDs for others (C12). Never parse or validate its shape.
    """
    vm_id: str
    status: str
    display_name: Optional[str]
    hardware_profile: Optional[str]
    gpu_count: Optional[int]
    instance_type: Optional[str]
    created_at: Optional[str]
    #: Host with the port already stripped off, or None while provisioning.
    ip: Optional[str]
    #: SSH port parsed out of `ip_address`, defaulting to 22 (C2).
    ssh_port: int
    raw: Dict[str, Any]

    @property
    def is_terminal(self) -> bool:
        """Dead beyond recovery.

        An UNRECOGNISED status is deliberately NOT terminal. The costs are
        asymmetric: believing a live VM is dead hides it from teardown and
        the provisioner launch a duplicate -- two machines billing at up to
        $63.92/h -- while believing a dead VM is alive costs one failed SSH.
        """
        return (self.status or '').strip().lower() in TERMINAL_STATUSES

    @property
    def is_usable(self) -> bool:
        """Ready AND reachable.

        A VM can report `status: "ready"` while `ip_address` is still null
        (C10, observed on h200 at 104 s). Provisioning must gate on this
        property, never on `status` alone.
        """
        return ((self.status or '').strip().lower() in READY_STATUSES and
                self.ip is not None)


def parse_ip_address(value: Optional[str]) -> Tuple[Optional[str], int]:
    """Split Lyceum's `ip_address` into (host, ssh_port).

    The field is polymorphic (C2): bare `"203.0.113.10"` on some VMs and
    `"198.51.100.20:22"` on others, from the same account minutes apart.

    Decided semantics (the tests pin all of these):
      * null/empty          -> (None, 22)
      * surrounding whitespace is stripped
      * bare IPv6 is a HOST, not host:port -- do not split on the last colon
      * `[v6]:port` parses as host + port
      * a malformed port (`host:`, `host:ssh`, `:0`, `:70000`) RAISES
        ValueError rather than falling back to 22. Silently producing a
        plausible-but-wrong host is exactly the intermittent failure C2 exists
        to prevent, and raising during provisioning is a path SkyPilot already
        knows how to tear down.
    """
    if value is None:
        return None, DEFAULT_SSH_PORT
    text = value.strip()
    if not text:
        return None, DEFAULT_SSH_PORT

    # `[v6]` / `[v6]:port` -- the one unambiguous IPv6 authority form.
    if text.startswith('['):
        end = text.find(']')
        if end < 0:
            raise ValueError(f'unparseable ip_address {value!r}: '
                             'missing closing bracket')
        host = text[1:end]
        rest = text[end + 1:]
        if not host:
            raise ValueError(f'unparseable ip_address {value!r}: empty host')
        if not rest:
            return host, DEFAULT_SSH_PORT
        if not rest.startswith(':'):
            raise ValueError(f'unparseable ip_address {value!r}: '
                             f'trailing {rest!r}')
        return host, _parse_port(rest[1:], value)

    colons = text.count(':')
    if colons == 0:
        return text, DEFAULT_SSH_PORT
    if colons == 1:
        host, _, port_text = text.partition(':')
        if not host:
            raise ValueError(f'unparseable ip_address {value!r}: empty host')
        return host, _parse_port(port_text, value)

    # More than one colon and no brackets: only a bare IPv6 literal is a
    # legitimate reading (RFC 3986 authority parsing). Splitting on the last
    # colon would turn `2a01:4f8:c17::1` into a broken host plus a port that
    # does not parse. Anything that is not valid IPv6 is a shape we do not
    # understand, and C2's lesson is to surface that on the first VM.
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        raise ValueError(f'unparseable ip_address {value!r}: '
                         'neither host, host:port, nor an IPv6 literal') from None
    return text, DEFAULT_SSH_PORT


def _parse_port(text: str, original: Optional[str]) -> int:
    """Strict TCP port parse. Never falls back to 22 -- see `parse_ip_address`."""
    if not text.isdigit():
        raise ValueError(f'unparseable ip_address {original!r}: '
                         f'port {text!r} is not a number')
    port = int(text)
    if not 1 <= port <= 65535:
        raise ValueError(f'unparseable ip_address {original!r}: '
                         f'port {port} out of range')
    return port


def read_api_key(path: str = API_KEY_PATH, env: Optional[Dict[str, str]] = None) -> str:
    """Return the API key from $LYCEUM_API_KEY, else the key file.

    Raises LyceumAuthError with an actionable message if neither is present.

    "Set but empty" is NOT a credential: container platforms commonly export an
    unset secret as `LYCEUM_API_KEY=` rather than omitting the variable, and
    honouring it would send `Bearer ` and turn a missing-secret deploy into a
    401 storm instead of a clear error. Same for a truncated/whitespace-only
    key file.
    """
    environ = os.environ if env is None else env
    from_env = environ.get(API_KEY_ENV_VAR)
    if from_env is not None and from_env.strip():
        return from_env.strip()

    expanded = os.path.expanduser(path)
    try:
        with open(expanded, 'r', encoding='utf-8') as handle:
            from_file = handle.read().strip()
    except OSError:
        from_file = ''
    if from_file:
        return from_file

    raise LyceumAuthError(
        f'No Lyceum API key found. Set ${API_KEY_ENV_VAR}, or write the key to '
        f'{expanded} (`mkdir -p ~/.lyceum && printf %s "$LYCEUM_API_KEY" > '
        f'{API_KEY_PATH}`). Keys are issued at https://lyceum.technology.')


def _decode_json(resp: Any) -> Any:
    """Body of a 2xx, or None if it is not JSON.

    A successful response with an undecodable body is not an error worth
    raising -- `DELETE /vms/{id}` is useful purely for its status code.
    """
    try:
        return resp.json()
    except Exception:  # ValueError, requests' JSONDecodeError, ...
        return None


def _error_detail(resp: Any) -> str:
    """The most useful human-readable text in an error response.

    Lyceum puts it in `detail`, and that text is often self-answering (the 400
    for an unknown profile enumerates every valid one), so it must reach the
    caller intact. A 500 from a proxy in front of the API is HTML rather than
    JSON; falling back to `.text` keeps the typed-error contract instead of
    raising a JSONDecodeError from inside the provisioner.
    """
    payload = _decode_json(resp)
    if isinstance(payload, dict):
        detail = payload.get('detail')
        if isinstance(detail, str) and detail.strip():
            return detail
        if detail is not None:
            return str(detail)
        if payload:
            return str(payload)
    elif payload is not None:
        return str(payload)
    text = getattr(resp, 'text', None)
    return text if isinstance(text, str) else ''


def _parse_timestamp(value: Optional[str]) -> datetime.datetime:
    """`created_at` as a comparable instant.

    Both shapes are real: `/status` returns `...067183Z` while `/vms/create`
    returns `2026-07-31T12:14:40.153832` with no zone, and `/user/status`
    proves offsets (`...+00:00`) occur too. `datetime.fromisoformat` rejects a
    trailing `Z` on Python 3.10 and this package supports >=3.10, so the `Z` is
    rewritten before parsing. Naive values are read as UTC so that a mixed list
    can be ordered at all -- comparing naive against aware raises TypeError.

    An unparseable or absent timestamp sorts oldest rather than raising: it
    must never promote an unknown row to "newest generation".
    """
    if not value or not isinstance(value, str):
        return _EPOCH
    text = value.strip()
    if text.endswith(('Z', 'z')):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return _EPOCH
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _parse_vm(payload: Dict[str, Any]) -> VM:
    """Normalise one VM record from create/status/list.

    Hardware identity comes from `instance_specs` first (C6): the create
    response has `hardware_profile` and `gpu_count` null at the top level while
    `instance_specs` carries the real values, and a VM record claiming to have
    no GPUs breaks every downstream accelerator match.
    """
    specs = payload.get('instance_specs') or {}

    profile = specs.get('gpu_type')
    if profile is None:
        profile = payload.get('hardware_profile')

    gpu_count = specs.get('gpu_count')
    if gpu_count is None:
        gpu_count = payload.get('gpu_count')
    if gpu_count is not None:
        try:
            gpu_count = int(gpu_count)
        except (TypeError, ValueError):
            gpu_count = None

    ip, ssh_port = parse_ip_address(payload.get('ip_address'))

    return VM(
        # C12: opaque. Never parsed, normalised, or validated.
        vm_id=payload.get('vm_id'),
        status=payload.get('status'),
        display_name=payload.get('display_name'),
        hardware_profile=profile,
        gpu_count=gpu_count,
        instance_type=payload.get('instance_type'),
        created_at=payload.get('created_at'),
        ip=ip,
        ssh_port=ssh_port,
        # The escape hatch for everything the dataclass does not model
        # (`uptime_seconds`, `billed`, `name`, `org_id`, and whatever the
        # vendor adds next).
        raw=payload,
    )


class LyceumClient:
    """Thin, typed wrapper over the five VM endpoints plus pricing.

    Every method raises a `LyceumError` subclass on failure; no method returns
    an error sentinel and none lets a `requests` exception escape.
    """

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._api_key = api_key

    @property
    def api_key(self) -> str:
        """The resolved credential.

        An explicit constructor argument always wins over ambient environment:
        autodown and the provisioner may share a process with other
        credentials, and picking up the wrong one deletes VMs in whichever org
        the env var happens to name.
        """
        if self._api_key:
            return self._api_key
        self._api_key = read_api_key()
        return self._api_key

    def _request(self, method: str, path: str,
                 body: Optional[Dict[str, Any]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 *, capacity_500: bool = True) -> Any:
        """Perform one HTTP call and map the response onto typed exceptions.

        Mapping (all verified against the live API):
          200/201 -> decoded JSON
          400     -> LyceumInvalidRequestError
          401/403 -> LyceumAuthError
          404     -> LyceumNotFoundError
          500 + `_CAPACITY_DETAIL_RE` in detail -> LyceumCapacityError  (C7)
                                                   ...but only if `capacity_500`
          other 5xx -> LyceumServerError

        `capacity_500` exists because the classification is per-call-site, not
        per-response. The body "The instance could not be provisioned right now"
        is meaningful on create and nonsense on delete, and `_request` cannot
        tell from the response alone. It matters because of what callers DO with
        the two types: LyceumCapacityError means "fail over now, stop trying
        here", and a teardown path that stops trying leaves a VM billing at up
        to $63.92/h with no cloud-side TTL behind it (C5).
        So `terminate_vm` passes `capacity_500=False` and every 5xx it sees is
        reported as retryable.
        """
        # Resolved first, and deliberately outside the try: a missing
        # credential must fail before the round trip, not as a 401 that reads
        # like a revoked key.
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        url = f'{self.base_url}{API_PREFIX}{path}'

        try:
            resp = requests.request(method, url, json=body, params=params,
                                    headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            # The module contract is that no `requests` exception escapes, so
            # callers can write one `except LyceumError` and be exhaustive.
            # A transport failure is by nature retryable.
            raise LyceumServerError(
                f'Lyceum API {method} {path} failed: {exc}') from exc

        status = resp.status_code
        if 200 <= status < 300:
            return _decode_json(resp)

        detail = _error_detail(resp)
        context = f'Lyceum API {method} {path} failed with HTTP {status}'
        message = f'{context}: {detail}' if detail else context

        if status == 400:
            raise LyceumInvalidRequestError(message)
        if status in (401, 403):
            raise LyceumAuthError(message)
        if status == 404:
            raise LyceumNotFoundError(message)
        if status >= 500:
            if capacity_500 and _CAPACITY_DETAIL_RE.search(detail or ''):
                raise LyceumCapacityError(message)
            raise LyceumServerError(message)
        if status >= 400:
            raise LyceumInvalidRequestError(message)
        raise LyceumError(message)

    # ---- VM lifecycle -----------------------------------------------------

    def create_vm(self, *, public_key: str, hardware_profile: str,
                  gpu_count: int = 1, display_name: Optional[str] = None,
                  use_spot: bool = False) -> VM:
        """POST /vms/create.

        Validates `hardware_profile` and `gpu_count` locally first, and RAISES
        `LyceumInvalidRequestError` on a bad value -- it does NOT clamp. The API
        silently coerces `gpu_count=0` to 1 and provisions a real billing VM
        (C8); silently repairing the caller's mistake here would reproduce that
        same surprise one layer up. The error message enumerates the valid
        values, mirroring the API's own helpful 400 for an unknown profile.
        No HTTP request is made when local validation fails.

        Note the response has `hardware_profile` and `gpu_count` as null at the
        top level; read them from `instance_specs` (C6).
        """
        if hardware_profile not in HARDWARE_PROFILES:
            raise LyceumInvalidRequestError(
                f'Unknown hardware_profile {hardware_profile!r}. '
                f'Valid options: {", ".join(HARDWARE_PROFILES)}')
        if isinstance(gpu_count, bool) or gpu_count not in ALLOWED_GPU_COUNTS:
            raise LyceumInvalidRequestError(
                f'Invalid gpu_count {gpu_count!r} for hardware_profile '
                f'{hardware_profile!r}. Valid options: '
                f'{", ".join(str(count) for count in ALLOWED_GPU_COUNTS)}. '
                '(Lyceum silently coerces gpu_count=0 to 1 and provisions a '
                'billing VM, so this is rejected here rather than on the wire.)')

        body: Dict[str, Any] = {
            'user_public_key': public_key,
            'hardware_profile': hardware_profile,
            # Nested, not top level: a top-level `gpu_count` is ignored by the
            # API, which is how C8's silent 0 -> 1 coercion goes unnoticed.
            'instance_specs': {'gpu_count': gpu_count},
            # Always explicit. The server default is the vendor's to change,
            # and it decides both the capacity pool and the price (C9).
            'instance_type': SPOT if use_spot else ON_DEMAND,
        }
        if display_name is not None:
            body['display_name'] = display_name

        return _parse_vm(self._request('POST', '/vms/create', body=body))

    def get_vm(self, vm_id: str) -> VM:
        """GET /vms/{id}/status. Raises LyceumNotFoundError on 404."""
        return _parse_vm(self._request('GET', f'/vms/{vm_id}/status'))

    def list_vms(self, *, include_terminated: bool = False) -> List[VM]:
        """GET /vms/list.

        Defaults to excluding terminated VMs. The API's own default for every
        include_* flag is `true`, so the naive call returns dead VMs (C4).

        Belt AND braces: the server-side flags are the cheap half of the
        defence, and the client-side status filter is the half whose semantics
        are ours rather than the vendor's.
        """
        flag = 'true' if include_terminated else 'false'
        params = {'include_terminated': flag, 'include_failed': flag}
        payload = self._request('GET', '/vms/list', params=params) or {}
        vms = [_parse_vm(row) for row in (payload.get('vms') or [])]
        if include_terminated:
            return vms
        return [vm for vm in vms if not vm.is_terminal]

    def terminate_vm(self, vm_id: str) -> None:
        """DELETE /vms/{id}. Idempotent: a 404 is success, not an error.

        Passes `capacity_500=False`: on a teardown path every 5xx must stay
        retryable. See `_request`.

        Idempotency covers 404 only. A 401 means we deleted nothing and must
        say so, or a caller reports a clean sweep it never made.
        """
        try:
            self._request('DELETE', f'/vms/{vm_id}', capacity_500=False)
        except LyceumNotFoundError:
            # Already gone is the outcome we wanted: teardown is retried and
            # races node-side autodown.
            return None
        return None

    def find_vms_by_display_name(self, display_name: str) -> List[VM]:
        """Live VMs whose display_name matches, newest first.

        This is the whole identity scheme: Lyceum has no tags and no
        server-side filtering, so `display_name == cluster_name_on_cloud` is
        the only handle. Terminal VMs are excluded and results are ordered by
        `created_at` descending, because names are reused across cluster
        generations and dead VMs keep theirs forever (C4).

        The match is EXACT, never a prefix: `sky-cluster` must not resolve
        `sky-cluster-abc`, or a teardown gets handed the wrong node. Ordering
        compares parsed instants, not raw strings -- `16:00+02:00` is the older
        instant while sorting later byte-wise.
        """
        matches = [
            vm for vm in self.list_vms(include_terminated=False)
            if vm.display_name == display_name
        ]
        return sorted(matches,
                      key=lambda vm: _parse_timestamp(vm.created_at),
                      reverse=True)

    # ---- catalog inputs ---------------------------------------------------

    def get_vm_prices(self) -> Dict[Tuple[str, str, int], float]:
        """GET /pricing -> {(instance_type, profile, gpu_count): usd_per_hour}.

        Reads rows with `meter_slug == "vm_running"`, whose key lives under
        `applies_to` (C3). Falls back to `group_by` for forward-compat.

        Values are plain `float`. `unit_price_per_hour` arrives as a string
        (`"2.790000"`); leaving it as one makes every comparison lexicographic
        ("10.0" < "2.79"), while `Decimal` and `numpy.float64` both break orjson
        in the SkyPilot API server. `unit_price` is per SECOND -- reading it
        instead would under-price Lyceum by 3600x.
        """
        payload = self._request('GET', '/pricing') or {}
        prices: Dict[Tuple[str, str, int], float] = {}

        for row in (payload.get('prices') or []):
            if not isinstance(row, dict):
                continue
            if row.get('meter_slug') != _VM_METER_SLUG:
                continue
            # `applies_to` FIRST. lyceum-cli 1.1.1's own bug is reading
            # `group_by`; the day the vendor emits it with stale content, a
            # group_by-first client quotes a catalog that does not exist.
            holder = row.get('applies_to')
            if not isinstance(holder, dict) or not holder:
                holder = row.get('group_by')
            if not isinstance(holder, dict):
                continue
            match = _PRICE_KEY_RE.match(str(holder.get('hardware_profile') or ''))
            if match is None:
                continue
            try:
                price = float(row['unit_price_per_hour'])
            except (KeyError, TypeError, ValueError):
                continue
            prices[(match.group('instance_type'), match.group('profile'),
                    int(match.group('gpu_count')))] = price

        return prices

    def get_availability(self) -> Dict[Tuple[str, str], List[int]]:
        """GET /vms/availability -> {(instance_type, profile): [gpu_counts]}.

        Must read `available_instance_variants`, NOT
        `available_hardware_profiles`: spot and on-demand are separate capacity
        axes and disagree, and l40s has no spot variant at all (C9).

        The result is advisory only and races hard (C11).

        "Variant exists, no capacity right now" is an EMPTY LIST; "variant does
        not exist" is an ABSENT KEY. The first is transient (retry after the
        cache TTL), the second permanent -- dropping empty keys conflates them.

        Counts are coerced to `int`: the catalog filters with
        `gpu_count in available`, and a str element makes every membership test
        quietly false, hiding the whole cloud from the optimizer.
        """
        payload = self._request('GET', '/vms/availability') or {}
        availability: Dict[Tuple[str, str], List[int]] = {}

        for variant in (payload.get('available_instance_variants') or []):
            if not isinstance(variant, dict):
                continue
            instance_type = variant.get('instance_type')
            profile = variant.get('hardware_profile')
            if instance_type is None or profile is None:
                continue
            counts: List[int] = []
            for count in (variant.get('available_gpus_per_instance') or []):
                try:
                    counts.append(int(count))
                except (TypeError, ValueError):
                    continue
            availability[(instance_type, profile)] = counts

        return availability

    def get_user_status(self) -> Dict[str, Any]:
        """GET /user/status. Used by `Lyceum.check_credentials`."""
        return self._request('GET', '/user/status')
