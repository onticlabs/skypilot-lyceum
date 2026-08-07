"""Regression tests for `skypilot_lyceum.api`.

Every test here pins a failure mode that was *observed* against the live Lyceum
API during an empirical review on 2026-07-31 -- the C1-C12 corrections tabulated
in README.md. The review cost real money on real GPUs; these tests exist so that
it never has to be repeated.

Docstrings cite the C-number and state the concrete production failure the test
prevents. Payloads come from `tests/fixtures/lyceum_api/*.json`, captured
verbatim from the live API. Where a scenario has no verbatim fixture (e.g. two
*live* VMs sharing a display_name) the payload is *derived* from a verbatim one
by copying real rows and changing only the field under test -- never invented
from scratch.

Mocking follows conftest: `unittest.mock`/`monkeypatch` + `RecordingTransport`.
Half of these bugs are in what we SEND, so most tests assert on
`RecordingTransport.calls` (method, URL, JSON body, query params, headers)
rather than only on the return value.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Sequence

import pytest
import requests

from conftest import FakeResponse
from conftest import RecordingTransport
from conftest import load_fixture
from conftest import response

from skypilot_lyceum import api

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def install(monkeypatch, responses: Sequence[Any]) -> RecordingTransport:
    """Route every outbound `requests` call into a RecordingTransport.

    conftest documents the idiom as `monkeypatch.setattr(api.requests,
    'request', t)`. We patch the `requests` module object itself (which *is*
    `api.requests` once the module imports it) plus the two other entry points a
    reasonable implementation might use -- `requests.api.request`, which backs
    `requests.post`/`requests.get`, and `Session.request`. That way a test that
    fails, fails because of the code under test and not because the
    implementation picked a different (equally valid) requests idiom.

    The one idiom this cannot intercept is `from requests import request` at
    import time; the implementation must call `requests.request(...)` through
    the module attribute.
    """
    transport = RecordingTransport(responses)
    monkeypatch.setattr(requests, 'request', transport)
    monkeypatch.setattr(requests.api, 'request', transport)
    monkeypatch.setattr(
        requests.sessions.Session, 'request',
        lambda self, method, url, **kwargs: transport(method, url, **kwargs))
    return transport


def client(**kwargs) -> api.LyceumClient:
    kwargs.setdefault('api_key', 'lk_test')
    return api.LyceumClient(**kwargs)


def url_path(call: Dict[str, Any]) -> str:
    """The endpoint path of a recorded call, with base URL + prefix removed.

    The prefix is not guesswork: the live create response carries
    `status_check_url == "/api/v2/external/vms/{vm_id}/status"`.
    """
    root = api.DEFAULT_BASE_URL + api.API_PREFIX
    url = call['url']
    assert url.startswith(root), f'expected {root!r} prefix, got {url!r}'
    return url[len(root):]


def make_vm(**overrides) -> api.VM:
    """A VM with every field defaulted, for testing pure properties."""
    fields: Dict[str, Any] = dict(
        vm_id='9918b72c1dce41a6875abaf5d5ab64e9',
        status='ready',
        display_name='sky-cluster-abc',
        hardware_profile='l40s',
        gpu_count=1,
        instance_type='on-demand',
        created_at='2026-07-31T12:14:40.067183Z',
        ip='203.0.113.10',
        ssh_port=22,
        raw={},
    )
    fields.update(overrides)
    return api.VM(**fields)


# ---------------------------------------------------------------------------
# C2 -- the polymorphic ip_address field
# ---------------------------------------------------------------------------


class TestParseIpAddress:
    """`ip_address` is `host` on some VMs and `host:port` on others (C2).

    Both shapes came out of the same org minutes apart: the on-demand l40s
    returned a bare address, the spot h100 returned `<address>:22`. (The
    addresses below are the real captured shapes with the octets replaced by
    documentation-range ones.) A client that assumes a bare IP builds an
    unreachable SSH host string on an arbitrary subset of nodes -- an
    intermittent launch failure that looks like a network flake.
    """

    def test_bare_ip_from_live_fixture(self):
        """C2: the on-demand l40s shape must parse to the default SSH port."""
        raw = load_fixture('vm_status_ready_bare_ip')['ip_address']
        assert raw == '203.0.113.10'  # guard against fixture drift
        assert api.parse_ip_address(raw) == ('203.0.113.10', 22)

    def test_host_port_from_live_fixture(self):
        """C2: the spot h100 shape must have its `:22` stripped off the host.

        Leaving it on yields `ssh lyceum@198.51.100.20:22`, which does not
        resolve.
        """
        raw = load_fixture('vm_status_ready_host_port')['ip_address']
        assert raw == '198.51.100.20:22'
        assert api.parse_ip_address(raw) == ('198.51.100.20', 22)

    def test_non_default_port_round_trips(self):
        """C2: a non-22 port must survive into `InstanceInfo.ssh_port`.

        SkyPilot supports non-22 SSH ports natively (RunPod, Vast). Hard-coding
        22 while stripping the suffix would silently point SSH at the wrong
        port the day Lyceum moves it.
        """
        assert api.parse_ip_address('198.51.100.20:2222') == ('198.51.100.20',
                                                              2222)

    @pytest.mark.parametrize('value', [None, '', '   '])
    def test_missing_ip_yields_none_and_default_port(self, value):
        """C2/C10: a VM without an IP yet must parse, not explode.

        `ip_address` is null for the whole provisioning window and, per C10,
        can still be null after `status == "ready"`. Raising here would crash
        the poll loop instead of polling again.
        """
        assert api.parse_ip_address(value) == (None, 22)

    def test_whitespace_is_stripped(self):
        """Defensive: a padded value must not become an unreachable host."""
        assert api.parse_ip_address('  203.0.113.10 ') == ('203.0.113.10', 22)

    def test_bare_ipv6_is_not_split_on_its_last_colon(self):
        """DECISION: a bare IPv6 literal is a host, not host:port.

        Lyceum has only ever returned IPv4, so this is a judgement call, and we
        record it here: splitting on the *last* colon would turn
        `2a01:4f8:c17::1` into host `2a01:4f8:c17:` port-parse-failure. An
        address with >1 colon and no brackets is therefore treated as a bare
        host with the default port -- the same rule RFC 3986 authority parsing
        uses.
        """
        assert api.parse_ip_address('2a01:4f8:c17::1') == ('2a01:4f8:c17::1',
                                                           22)

    def test_bracketed_ipv6_with_port(self):
        """DECISION: `[v6]:port` is the one unambiguous IPv6 host:port form."""
        assert api.parse_ip_address('[2a01:4f8:c17::1]:2222') == (
            '2a01:4f8:c17::1', 2222)

    @pytest.mark.parametrize(
        'value', ['203.0.113.10:', '203.0.113.10:ssh', '203.0.113.10:0',
                  '203.0.113.10:70000', '203.0.113.10:22:22'])
    def test_malformed_port_raises_rather_than_defaulting(self, value):
        """DECISION: an unparseable port fails loud instead of assuming 22.

        C2's whole lesson is that quietly producing a plausible-but-wrong host
        string yields intermittent, hard-to-attribute launch failures. If the
        vendor ever emits a shape we do not understand we want it to surface on
        the first VM, not on one node in twenty. Raising here is safe: it
        happens inside provisioning, which SkyPilot already treats as a
        recoverable failure and tears down -- no VM is left billing.
        """
        with pytest.raises(ValueError):
            api.parse_ip_address(value)


# ---------------------------------------------------------------------------
# C10 -- "ready" is not the same as usable
# ---------------------------------------------------------------------------


class TestVMStatusProperties:
    """`status: "ready"` with `ip_address: null` is a real response (C10)."""

    def test_ready_with_null_ip_is_not_usable(self, api_key, monkeypatch):
        """C10: the exact h200 response that crashed the measurement script.

        `wait_instances` gating on `status` alone hands SkyPilot a null host and
        fails the launch on an arbitrary subset of nodes. This fixture is that
        response, verbatim.
        """
        transport = install(monkeypatch,
                            [response('vm_status_ready_null_ip')])
        vm = client().get_vm('3044903794294bd19dd457ea6784ad91')

        assert vm.status == 'ready'
        assert vm.ip is None
        assert vm.is_usable is False
        assert vm.is_terminal is False
        assert transport.exhausted

    @pytest.mark.parametrize(
        'fixture_name,expected_ip,expected_port',
        [('vm_status_ready_bare_ip', '203.0.113.10', 22),
         ('vm_status_ready_host_port', '198.51.100.20', 22)])
    def test_ready_with_ip_is_usable(self, api_key, monkeypatch, fixture_name,
                                     expected_ip, expected_port):
        """C10 + C2: ready AND addressable is the only usable state."""
        install(monkeypatch, [response(fixture_name)])
        vm = client().get_vm('any-id')

        assert vm.is_usable is True
        assert vm.ip == expected_ip
        assert vm.ssh_port == expected_port

    @pytest.mark.parametrize('status', ['terminated', 'failed', 'error'])
    def test_is_terminal_true_for_dead_statuses(self, status):
        """C4: a terminal VM must never be resolved as a live cluster member.

        Terminated VMs keep their display_name forever, so this predicate is
        the only thing standing between a name lookup and a dead machine.
        """
        assert make_vm(status=status).is_terminal is True

    @pytest.mark.parametrize('status',
                             ['pending', 'provisioning', 'ready', 'running'])
    def test_is_terminal_false_for_live_statuses(self, status):
        """C4: a provisioning VM is not dead -- reaping it would kill a launch."""
        assert make_vm(status=status).is_terminal is False

    def test_unknown_status_is_not_terminal(self):
        """DECISION: an unrecognised status counts as alive.

        Asymmetric costs. Believing a live VM is dead makes autodown skip it
        and the provisioner launch a duplicate -- two machines billing at up
        to $63.92/h. Believing a dead VM is alive costs one failed SSH. So an
        unknown status is never terminal.
        """
        assert make_vm(status='stopping').is_terminal is False

    def test_terminal_vm_is_never_usable(self):
        """C4: a terminated VM keeps its last IP in `/vms/list` -- and it is
        the second-most-dangerous thing in the API after the name reuse."""
        assert make_vm(status='terminated',
                       ip='198.51.100.20').is_usable is False


# ---------------------------------------------------------------------------
# C6 / C8 -- create
# ---------------------------------------------------------------------------


class TestCreateVM:

    def test_hardware_identity_comes_from_instance_specs(
            self, api_key, monkeypatch):
        """C6: the create response has null `hardware_profile`/`gpu_count`.

        Both are null at the top level while `instance_specs` carries the real
        values. Reading the top level yields a VM record that claims to have no
        GPUs, which then breaks every downstream accelerator match.
        """
        payload = load_fixture('vm_create_pending')
        assert payload['hardware_profile'] is None  # guard against drift
        assert payload['gpu_count'] is None
        assert payload['instance_specs'] == {'gpu_type': 'l40s',
                                             'gpu_count': 1}

        install(monkeypatch, [response('vm_create_pending')])
        vm = client().create_vm(public_key='ssh-ed25519 AAAAC3Nz test',
                                hardware_profile='l40s',
                                gpu_count=1,
                                display_name='sky-spike-l40s-0')

        assert vm.hardware_profile == 'l40s'
        assert vm.gpu_count == 1
        assert vm.vm_id == '9918b72c1dce41a6875abaf5d5ab64e9'
        assert vm.status == 'pending'
        assert vm.display_name == 'sky-spike-l40s-0'
        assert vm.ip is None

    def test_vm_id_is_kept_verbatim(self, api_key, monkeypatch):
        """C12: `vm_id` is undashed hex here and a dashed UUID elsewhere.

        Normalising or validating its shape would reject half the fleet.
        """
        install(monkeypatch, [response('vm_create_pending')])
        vm = client().create_vm(public_key='ssh-ed25519 AAAAC3Nz test',
                                hardware_profile='l40s')
        assert vm.vm_id == load_fixture('vm_create_pending')['vm_id']

    def test_request_body_shape(self, api_key, monkeypatch):
        """The payload shape confirmed against lyceum-cli 1.1.1 and a live call.

        `gpu_count` is nested under `instance_specs`; a top-level `gpu_count` is
        ignored by the API, which is how C8's silent 0->1 coercion goes
        unnoticed.
        """
        transport = install(monkeypatch, [response('vm_create_pending')])
        client().create_vm(public_key='ssh-ed25519 AAAAC3Nz test',
                           hardware_profile='l40s',
                           gpu_count=2,
                           display_name='sky-cluster-abc')

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call['method'].upper() == 'POST'
        assert url_path(call) == '/vms/create'

        body = call['json']
        assert body['user_public_key'] == 'ssh-ed25519 AAAAC3Nz test'
        assert body['hardware_profile'] == 'l40s'
        assert body['instance_specs']['gpu_count'] == 2
        assert body['display_name'] == 'sky-cluster-abc'
        assert body['instance_type'] == 'on-demand'

    def test_authorization_header_is_bearer(self, api_key, monkeypatch):
        """A missing/misformatted header is a 401 in production and a silent
        pass in a test that only checks the response body."""
        transport = install(monkeypatch, [response('vm_create_pending')])
        client(api_key='lk_secret').create_vm(public_key='k',
                                              hardware_profile='l40s')
        assert transport.calls[0]['headers']['Authorization'] == \
            'Bearer lk_secret'

    def test_timeout_is_always_passed(self, api_key, monkeypatch):
        """A control-plane HTTP call with no timeout wedges the provisioner
        thread forever; the VM keeps billing while nothing progresses."""
        transport = install(monkeypatch, [response('vm_create_pending')])
        client(timeout=17.0).create_vm(public_key='k', hardware_profile='l40s')
        assert transport.calls[0]['timeout'] == 17.0

    def test_instance_type_is_spot_when_requested(self, api_key, monkeypatch):
        """`instance_type` selects a different capacity pool AND price (C9).

        h100 spot is $1.10/h against $2.79 on-demand; sending the wrong one
        either overpays 2.5x or provisions from a pool that does not exist for
        that profile.
        """
        transport = install(monkeypatch, [response('vm_status_ready_host_port')])
        client().create_vm(public_key='k', hardware_profile='h100',
                           use_spot=True)
        assert transport.calls[0]['json']['instance_type'] == 'spot'

    def test_instance_type_is_sent_explicitly_when_on_demand(
            self, api_key, monkeypatch):
        """`instance_type` must never be omitted and left to a server default.

        The default is the vendor's to change. Omitting the field makes the
        bill depend on a value we never chose, and makes the catalog price we
        quoted the optimizer a fiction.
        """
        transport = install(monkeypatch, [response('vm_create_pending')])
        client().create_vm(public_key='k', hardware_profile='l40s',
                           use_spot=False)
        body = transport.calls[0]['json']
        assert 'instance_type' in body
        assert body['instance_type'] == 'on-demand'

    @pytest.mark.parametrize('gpu_count', [1, 2, 4, 8])
    def test_allowed_gpu_counts_are_sent_through(self, api_key, monkeypatch,
                                                 gpu_count):
        """C8: the four counts the API actually accepts must not be blocked."""
        transport = install(monkeypatch, [response('vm_create_pending')])
        client().create_vm(public_key='k', hardware_profile='h100',
                           gpu_count=gpu_count)
        assert transport.calls[0]['json']['instance_specs']['gpu_count'] == \
            gpu_count
        assert tuple(api.ALLOWED_GPU_COUNTS) == (1, 2, 4, 8)

    def test_gpu_count_zero_is_rejected_without_any_http_call(
            self, api_key, monkeypatch):
        """C8: the live API answers `gpu_count: 0` with 200 and a billing VM.

        Observed: the response came back with `instance_specs: {gpu_count: 1}`
        and a real machine started charging. The request must therefore die on
        this side of the wire -- an exception raised after the POST is too
        late, the VM already exists.
        """
        transport = install(monkeypatch, [])  # any HTTP call = test failure
        with pytest.raises(api.LyceumInvalidRequestError):
            client().create_vm(public_key='k', hardware_profile='l40s',
                               gpu_count=0)
        assert transport.calls == []

    @pytest.mark.parametrize('gpu_count', [3, -1, 16, 5])
    def test_out_of_set_gpu_counts_are_rejected_locally(
            self, api_key, monkeypatch, gpu_count):
        """C8: `gpu_count: 3` earns a generic 400 that names nothing.

        `{"detail": "The instance request was invalid..."}` -- no field, no
        allowed set. Validating locally turns an unactionable round trip into a
        message that says which counts are legal.
        """
        transport = install(monkeypatch, [])
        with pytest.raises(api.LyceumInvalidRequestError):
            client().create_vm(public_key='k', hardware_profile='h100',
                               gpu_count=gpu_count)
        assert transport.calls == []

    def test_unknown_hardware_profile_is_rejected_locally(
            self, api_key, monkeypatch):
        """A typo'd profile must not cost a round trip to discover.

        The catalog and the validator read the same `HARDWARE_PROFILES`
        constant, so a profile the catalog can offer is always a profile create
        will accept.
        """
        transport = install(monkeypatch, [])
        with pytest.raises(api.LyceumInvalidRequestError) as exc:
            client().create_vm(public_key='k',
                               hardware_profile='nonexistent-xyz')
        assert transport.calls == []
        assert 'nonexistent-xyz' in str(exc.value)

    def test_local_validation_message_enumerates_valid_profiles(
            self, api_key, monkeypatch):
        """The server's 400 is helpfully explicit; ours must not be worse.

        `Unknown hardware_profile 'x'. Valid options: a100, b200, b300, h100,
        h200, l40s` is a self-answering error. A local "invalid profile" that
        does not list the alternatives is a regression on the API we wrap.
        """
        install(monkeypatch, [])
        with pytest.raises(api.LyceumInvalidRequestError) as exc:
            client().create_vm(public_key='k', hardware_profile='gh200')
        message = str(exc.value)
        assert all(profile in message for profile in api.HARDWARE_PROFILES)


# ---------------------------------------------------------------------------
# C7 -- error mapping, and why capacity must be distinguishable
# ---------------------------------------------------------------------------


class TestErrorMapping:

    def test_500_capacity_body_raises_capacity_error(self, api_key,
                                                     monkeypatch):
        """C7: capacity exhaustion is signalled by HTTP 500, not 4xx.

        Three probes (b200 x1, b300 x1, h200 x2) were all refused in 2.7-9.1 s
        with this exact body. A conventional retry-on-5xx client sits in
        backoff against an SKU that has no capacity instead of failing over to
        another cloud.
        """
        body = load_fixture('error_500_capacity')
        assert body['detail'].startswith('The instance could not be '
                                         'provisioned')  # drift guard
        transport = install(monkeypatch,
                            [response('error_500_capacity', status=500)])

        with pytest.raises(api.LyceumCapacityError) as exc:
            client().create_vm(public_key='k', hardware_profile='b200')
        assert 'could not be provisioned' in str(exc.value)
        # Exactly one call: a capacity 500 must NOT be retried. Only one
        # response is scripted, so an internal retry would surface as
        # RecordingTransport's "unscripted request" assertion.
        assert len(transport.calls) == 1

    def test_500_with_any_other_body_raises_server_error(self, api_key,
                                                         monkeypatch):
        """C7, the other half: a genuine fault must stay retryable.

        Classifying every 500 as capacity would make a transient gateway blip
        look like a permanently exhausted SKU and abandon the launch.

        Several identical responses are scripted deliberately: unlike the
        capacity case this error IS retryable, so the client is free to back
        off internally and the test must not pin it to a single call.
        """
        install(monkeypatch,
                [response({'detail': 'Internal Server Error'}, status=500)] * 4)
        with pytest.raises(api.LyceumServerError) as exc:
            client().create_vm(public_key='k', hardware_profile='h100')
        assert not isinstance(exc.value, api.LyceumCapacityError)
        assert 'Internal Server Error' in str(exc.value)

    def test_capacity_error_is_not_a_subclass_of_server_error(self):
        """C7: the two must be distinguishable by `except`, in both directions.

        They mean opposite things -- capacity means "fail over now", server
        error means "retry here". If either were a subclass of the other, an
        `except LyceumServerError: retry` block would swallow capacity errors
        and burn the retry budget on an SKU that cannot be provisioned, or
        vice versa. Both remain catchable as `LyceumError`.
        """
        assert not issubclass(api.LyceumCapacityError, api.LyceumServerError)
        assert not issubclass(api.LyceumServerError, api.LyceumCapacityError)
        assert issubclass(api.LyceumCapacityError, api.LyceumError)
        assert issubclass(api.LyceumServerError, api.LyceumError)

    def test_capacity_detection_is_case_insensitive(self, api_key,
                                                    monkeypatch):
        """C7: the classifier hangs off one English sentence -- fragile by
        construction, per the design. A capitalisation change on the vendor
        side must not silently turn every failover into a retry storm; this
        test pins the tolerance that `_CAPACITY_DETAIL_RE` already declares."""
        install(monkeypatch, [
            response({'detail': 'The instance COULD NOT BE PROVISIONED right '
                                'now.'}, status=500)
        ])
        with pytest.raises(api.LyceumCapacityError):
            client().create_vm(public_key='k', hardware_profile='b300')

    def test_500_with_unparseable_body_raises_server_error(self, api_key,
                                                           monkeypatch):
        """A 500 from a proxy is HTML, not JSON. Blowing up with a
        JSONDecodeError inside the provisioner loses the typed-error contract
        (`no method lets a requests exception escape`) and skips the retry."""
        # A response whose .json() raises and whose .text is HTML -- what a
        # proxy in front of the API returns when the app is down. Repeated,
        # since a non-capacity 5xx may legitimately be retried internally.
        install(monkeypatch,
                [FakeResponse(status_code=500, payload=None,
                              text='<html>502 Bad Gateway</html>')] * 4)
        with pytest.raises(api.LyceumServerError):
            client().get_vm('9918b72c1dce41a6875abaf5d5ab64e9')

    def test_400_unknown_profile_preserves_the_detail_text(self, api_key,
                                                           monkeypatch):
        """The 400 body enumerates the valid profiles; that must reach the user.

        `Unknown hardware_profile 'nonexistent-xyz'. Valid options: a100, b200,
        b300, h100, h200, l40s`. Swallowing it and raising a bare "bad request"
        turns a self-answering error into a support ticket.
        """
        detail = load_fixture('error_400_unknown_profile')['detail']
        install(monkeypatch,
                [response('error_400_unknown_profile', status=400)])
        with pytest.raises(api.LyceumInvalidRequestError) as exc:
            client().get_vm('some-id')
        assert detail in str(exc.value)
        assert 'a100, b200, b300, h100, h200, l40s' in str(exc.value)

    def test_400_generic_invalid_request(self, api_key, monkeypatch):
        """C8's unhelpful sibling: the body that names nothing.

        Still must map to LyceumInvalidRequestError -- 400 is never retryable,
        and never a capacity signal.
        """
        install(monkeypatch,
                [response('error_400_invalid_request', status=400)])
        with pytest.raises(api.LyceumInvalidRequestError) as exc:
            client().get_vm('some-id')
        assert 'The instance request was invalid' in str(exc.value)
        assert not isinstance(exc.value, api.LyceumCapacityError)
        assert not isinstance(exc.value, api.LyceumServerError)

    def test_404_raises_not_found(self, api_key, monkeypatch):
        """A vanished VM must be distinguishable from a broken API.

        `query_instances` turns "not found" into "the node is gone" and
        anything else into "status unknown, do not touch the cluster".
        """
        transport = install(monkeypatch,
                            [response('error_404_vm_not_found', status=404)])
        with pytest.raises(api.LyceumNotFoundError) as exc:
            client().get_vm('9918b72c1dce41a6875abaf5d5ab64e9')
        assert 'VM not found' in str(exc.value)
        # No retry on a 4xx: it will never become a 200, and the poll loop
        # that calls this runs once a second for up to ten minutes.
        assert len(transport.calls) == 1

    @pytest.mark.parametrize('status', [401, 403])
    def test_auth_failures_map_to_auth_error(self, api_key, monkeypatch,
                                             status):
        """A rotated key must fail `sky check` loudly, not look like an outage.

        401 and 403 both mean the credential is the problem; retrying either is
        pure latency.
        """
        install(monkeypatch,
                [response({'detail': 'Invalid API key'}, status=status)])
        with pytest.raises(api.LyceumAuthError) as exc:
            client().get_user_status()
        assert 'Invalid API key' in str(exc.value)

    def test_every_error_is_a_lyceum_error(self, api_key, monkeypatch):
        """The module contract: no `requests` exception escapes, so callers can
        write one `except LyceumError` and be exhaustive."""
        for status, fixture_name in [(400, 'error_400_invalid_request'),
                                     (404, 'error_404_vm_not_found'),
                                     (500, 'error_500_capacity')]:
            install(monkeypatch, [response(fixture_name, status=status)])
            with pytest.raises(api.LyceumError):
                client().get_vm('some-id')


# ---------------------------------------------------------------------------
# C4 -- listing, and the terminated VMs that never go away
# ---------------------------------------------------------------------------


class TestListVMs:

    def test_include_terminated_false_is_sent_by_default(self, api_key,
                                                         monkeypatch):
        """C4: every `include_*` flag defaults to `true` server-side.

        The naive call returns terminated VMs. Since identity is display-name
        matching, that resolves a dead VM as live -- "the single most dangerous
        finding in the review".
        """
        transport = install(monkeypatch, [response('vm_list_mixed')])
        client().list_vms()

        call = transport.calls[0]
        assert call['method'].upper() == 'GET'
        assert url_path(call) == '/vms/list'
        params = call['params']
        assert params['include_terminated'] in (False, 'false')

    def test_all_include_flags_default_to_false(self, api_key, monkeypatch):
        """C4 generalised: if we send an `include_*` flag at all, it is false.

        The vendor's default is `true` for every one of them, so any flag we
        pass through must be explicitly narrowing, never widening.
        """
        transport = install(monkeypatch, [response('vm_list_mixed')])
        client().list_vms()
        params = transport.calls[0]['params']
        for key, value in params.items():
            if key.startswith('include_'):
                assert value in (False, 'false'), f'{key}={value!r}'

    def test_include_terminated_true_sends_true(self, api_key, monkeypatch):
        """Teardown needs the opposite view: everything, including corpses.

        Terminating an orphan requires first seeing it.
        """
        transport = install(monkeypatch, [response('vm_list_mixed')])
        client().list_vms(include_terminated=True)
        assert transport.calls[0]['params']['include_terminated'] in (True,
                                                                     'true')

    def test_terminal_vms_are_dropped_client_side(self, api_key, monkeypatch):
        """C4, belt and braces: filter locally even though we asked the server.

        The fixture is what `/vms/list` really returns -- a terminated h100 and
        a ready l40s that *share* `display_name: "sky-cluster-abc"`. The flag's
        semantics are the vendor's to change; the filter is ours.
        """
        install(monkeypatch, [response('vm_list_mixed')])
        vms = client().list_vms()

        assert [vm.vm_id for vm in vms] == [
            '9918b72c1dce41a6875abaf5d5ab64e9',
            '6f3bd1ec9f524d6c83a24078b94e214c',
        ]
        assert all(not vm.is_terminal for vm in vms)

    def test_include_terminated_true_keeps_terminal_vms(self, api_key,
                                                        monkeypatch):
        """Asking for terminated VMs and getting them filtered out would make
        node-side autodown structurally unable to see its targets."""
        install(monkeypatch, [response('vm_list_mixed')])
        vms = client().list_vms(include_terminated=True)

        assert len(vms) == 3
        assert 'fa292eba-16ba-4dd2-8c31-f58ac11d87bf' in {
            vm.vm_id for vm in vms
        }

    def test_empty_listing(self, api_key, monkeypatch):
        """A fresh org returns `{"vms": [], "total": 0}`, not `null`."""
        install(monkeypatch, [response('vm_list_empty')])
        assert client().list_vms() == []

    def test_listed_vms_are_fully_parsed(self, api_key, monkeypatch):
        """C2/C6 apply to the list endpoint too, not just to status."""
        install(monkeypatch, [response('vm_list_mixed')])
        vms = {vm.vm_id: vm for vm in client().list_vms()}

        ready = vms['9918b72c1dce41a6875abaf5d5ab64e9']
        assert ready.display_name == 'sky-cluster-abc'
        assert (ready.ip, ready.ssh_port) == ('203.0.113.10', 22)
        assert ready.hardware_profile == 'l40s'
        assert ready.gpu_count == 1
        assert ready.instance_type == 'on-demand'
        assert ready.is_usable is True

        provisioning = vms['6f3bd1ec9f524d6c83a24078b94e214c']
        assert provisioning.gpu_count == 2
        assert provisioning.ip is None
        assert provisioning.is_usable is False


# ---------------------------------------------------------------------------
# C4 -- name resolution, the whole identity scheme
# ---------------------------------------------------------------------------


class TestFindVMsByDisplayName:
    """Lyceum has no tags and no server-side filtering, so
    `display_name == cluster_name_on_cloud` resolved by listing is the only
    handle we have on a cluster."""

    def test_terminated_namesake_is_excluded(self, api_key, monkeypatch):
        """C4: the fixture's two `sky-cluster-abc` rows are the trap.

        One terminated, one ready, same name -- exactly what you get after
        relaunching a cluster. Returning the terminated one points SkyPilot at
        a machine that no longer exists.
        """
        install(monkeypatch, [response('vm_list_mixed')])
        vms = client().find_vms_by_display_name('sky-cluster-abc')

        assert [vm.vm_id for vm in vms] == [
            '9918b72c1dce41a6875abaf5d5ab64e9'
        ]
        assert vms[0].status == 'ready'

    def test_prefix_does_not_match(self, api_key, monkeypatch):
        """A prefix/substring match would resolve `sky-cluster-abc` *and*
        `sky-cluster-other` for the name `sky-cluster`, and hand a teardown the
        wrong node -- the fixture holds two differently-named clusters for
        exactly this check."""
        install(monkeypatch, [response('vm_list_mixed')])
        assert client().find_vms_by_display_name('sky-cluster') == []

    def test_exact_match_only(self, api_key, monkeypatch):
        """The unrelated cluster resolves to itself and nothing else."""
        install(monkeypatch, [response('vm_list_mixed')])
        vms = client().find_vms_by_display_name('sky-cluster-other')
        assert [vm.vm_id for vm in vms] == [
            '6f3bd1ec9f524d6c83a24078b94e214c'
        ]

    def test_no_match_returns_empty_list_not_an_exception(
            self, api_key, monkeypatch):
        """`run_instances` asks "does this cluster exist yet?" on every launch.

        The common answer is "no". Raising would make the normal path an
        exception path.
        """
        install(monkeypatch, [response('vm_list_mixed')])
        assert client().find_vms_by_display_name('sky-cluster-nope') == []

    def test_newest_generation_wins(self, api_key, monkeypatch):
        """C4: names are reused across cluster generations.

        Derived from `vm_list_mixed` by reviving its terminated row (status and
        created_at changed, nothing else) so that two *live* VMs share a name.
        Uniqueness is not enforced server-side, so ordering by `created_at`
        descending is what keeps us on the current generation.
        """
        payload = copy.deepcopy(load_fixture('vm_list_mixed'))
        payload['vms'][0]['status'] = 'ready'
        payload['vms'][0]['created_at'] = '2026-07-31T16:00:00.000000Z'

        install(monkeypatch, [response(payload)])
        vms = client().find_vms_by_display_name('sky-cluster-abc')

        assert [vm.vm_id for vm in vms] == [
            'fa292eba-16ba-4dd2-8c31-f58ac11d87bf',
            '9918b72c1dce41a6875abaf5d5ab64e9',
        ]

    def test_ordering_survives_mixed_timestamp_formats(self, api_key,
                                                       monkeypatch):
        """Both timestamp shapes are real: `/vms/create` returns
        `2026-07-31T12:14:40.153832` (no zone) and `/status` returns
        `...067183Z`.

        `datetime.fromisoformat` rejects the trailing `Z` on Python 3.10, and
        this package supports >=3.10, so a naive parser raises on exactly the
        rows the create path produces. Ordering must be correct across the mix.
        """
        payload = copy.deepcopy(load_fixture('vm_list_mixed'))
        payload['vms'][0]['status'] = 'ready'
        payload['vms'][0]['created_at'] = '2026-07-31T16:00:00.000000Z'
        payload['vms'][1]['created_at'] = '2026-07-31T15:00:00.000000'

        install(monkeypatch, [response(payload)])
        vms = client().find_vms_by_display_name('sky-cluster-abc')
        assert [vm.vm_id for vm in vms] == [
            'fa292eba-16ba-4dd2-8c31-f58ac11d87bf',
            '9918b72c1dce41a6875abaf5d5ab64e9',
        ]

    def test_lookup_asks_the_server_to_exclude_terminated(self, api_key,
                                                          monkeypatch):
        """The identity rule, step 1: the server-side flag is the cheap half
        of the defence; without it every lookup pages through every VM the org
        has ever created."""
        transport = install(monkeypatch, [response('vm_list_mixed')])
        client().find_vms_by_display_name('sky-cluster-abc')
        assert transport.calls[0]['params']['include_terminated'] in (False,
                                                                     'false')


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------


class TestTerminateVM:

    def test_delete_hits_the_vm_path(self, api_key, monkeypatch):
        """This DELETE is the *only* thing that stops a Lyceum VM billing --
        there is no cloud-side TTL (C5). Getting the method or path wrong leaks
        money silently."""
        transport = install(monkeypatch, [response('vm_terminate_ok')])
        assert client().terminate_vm('9918b72c1dce41a6875abaf5d5ab64e9') is None

        call = transport.calls[0]
        assert call['method'].upper() == 'DELETE'
        assert url_path(call) == '/vms/9918b72c1dce41a6875abaf5d5ab64e9'

    def test_opaque_vm_id_is_not_normalised(self, api_key, monkeypatch):
        """C12: dashed UUIDs and undashed hex both occur; stripping or
        re-formatting either one deletes nothing (or, worse, something else)."""
        transport = install(monkeypatch, [response('vm_terminate_ok')])
        client().terminate_vm('fa292eba-16ba-4dd2-8c31-f58ac11d87bf')
        assert url_path(transport.calls[0]).endswith(
            'fa292eba-16ba-4dd2-8c31-f58ac11d87bf')

    def test_404_is_success(self, api_key, monkeypatch):
        """Teardown is retried and races with autodown.

        Already-gone is the outcome we wanted. Raising here would abort
        SkyPilot's `down` path and leave the cluster marked as live.
        """
        install(monkeypatch, [response('error_404_vm_not_found', status=404)])
        assert client().terminate_vm('9918b72c1dce41a6875abaf5d5ab64e9') is None

    def test_capacity_shaped_500_on_delete_is_a_server_error(
            self, api_key, monkeypatch):
        """DECISION: C7's capacity classification applies to create, not delete.

        "The instance could not be provisioned right now" is nonsense on a
        DELETE, but the string match in `_request` cannot know that. The
        classification matters because of what callers do with it:
        LyceumCapacityError means "fail over now, stop trying here", and a
        teardown path that stops trying leaves a VM billing at up to $63.92/h
        with no cloud-side TTL to catch it -- the most expensive failure mode
        in the design. So terminate_vm reports any 5xx as LyceumServerError
        (retryable), and never as capacity.
        """
        install(monkeypatch, [response('error_500_capacity', status=500)])
        with pytest.raises(api.LyceumServerError) as exc:
            client().terminate_vm('9918b72c1dce41a6875abaf5d5ab64e9')
        assert not isinstance(exc.value, api.LyceumCapacityError)

    def test_auth_failure_on_delete_still_raises(self, api_key, monkeypatch):
        """Idempotency covers 404 only. A 401 means we deleted nothing and must
        say so, or autodown will report a clean sweep it never made."""
        install(monkeypatch,
                [response({'detail': 'Invalid API key'}, status=401)])
        with pytest.raises(api.LyceumAuthError):
            client().terminate_vm('9918b72c1dce41a6875abaf5d5ab64e9')


# ---------------------------------------------------------------------------
# C3 -- pricing, which IS the catalog
# ---------------------------------------------------------------------------


class TestGetVMPrices:

    def test_exactly_48_rows(self, api_key, monkeypatch):
        """C3: 6 profiles x 4 GPU counts x {on-demand, spot} = 48.

        This is the whole catalog. A parser that drops rows makes SKUs
        invisible to the optimizer; one that duplicates them makes the
        optimizer pick a price that does not exist.
        """
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()
        assert len(prices) == 48

    def test_key_is_instance_type_profile_gpu_count(self, api_key,
                                                    monkeypatch):
        """C3: the key is `{instance_type}.{profile}.{count}x` under
        `applies_to`, split into a typed tuple.

        Leaving it as the raw dotted string forces every consumer to re-parse
        it, and `4x` sorts before `8x` but after `2x` only by accident.
        """
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()

        assert ('on-demand', 'h100', 1) in prices
        assert ('spot', 'h100', 8) in prices
        assert set(prices) == {
            (instance_type, profile, count)
            for instance_type in ('on-demand', 'spot')
            for profile in api.HARDWARE_PROFILES
            for count in api.ALLOWED_GPU_COUNTS
        }

    def test_values_are_plain_python_floats(self, api_key, monkeypatch):
        """`unit_price_per_hour` arrives as a *string* (`"2.790000"`).

        The obvious conversions leak: `Decimal` and `numpy.float64` both break
        orjson serialisation in the SkyPilot API server, and `numpy.float64` is
        a `float` subclass so `isinstance` would not catch it -- hence the
        exact `type(...) is float` check. A str value silently makes every
        price comparison lexicographic ("10.0" < "2.79").
        """
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()
        for key, value in prices.items():
            assert type(value) is float, f'{key} -> {value!r} ({type(value)})'

    def test_hourly_prices_match_the_live_numbers(self, api_key, monkeypatch):
        """Reading `unit_price` (per *second*) instead of `unit_price_per_hour`
        would under-price everything by 3600x and make Lyceum win every
        optimizer decision."""
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()

        assert prices[('on-demand', 'h100', 1)] == pytest.approx(2.79, abs=1e-4)
        assert prices[('spot', 'h100', 1)] == pytest.approx(1.10, abs=1e-4)
        assert prices[('on-demand', 'l40s', 1)] == pytest.approx(1.19,
                                                                 abs=1e-4)
        assert prices[('on-demand', 'b300', 8)] == pytest.approx(63.92,
                                                                 abs=1e-4)

    def test_price_is_linear_in_gpu_count(self, api_key, monkeypatch):
        """C3: h100 goes 2.79 / 5.58 / 11.16 / 22.32.

        If the `{count}x` suffix were mis-parsed, every count would collapse
        onto the same key and the last row would win -- an 8-GPU node quoted at
        the 1-GPU price.
        """
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()
        for instance_type in ('on-demand', 'spot'):
            for profile in api.HARDWARE_PROFILES:
                unit = prices[(instance_type, profile, 1)]
                for count in (2, 4, 8):
                    assert prices[(instance_type, profile,
                                   count)] == pytest.approx(unit * count,
                                                            rel=1e-3)

    def test_spot_is_cheaper_than_on_demand_everywhere(self, api_key,
                                                       monkeypatch):
        """Sanity check on the `on-demand.`/`spot.` split in the key.

        If the two prefixes were confused, spot rows would carry on-demand
        prices and the 2.5x saving that justifies phase 6 would vanish into a
        rounding error nobody notices.
        """
        install(monkeypatch, [response('pricing_vm_running')])
        prices = client().get_vm_prices()
        for profile in api.HARDWARE_PROFILES:
            for count in api.ALLOWED_GPU_COUNTS:
                spot = prices[('spot', profile, count)]
                on_demand = prices[('on-demand', profile, count)]
                assert spot < on_demand, f'{profile} x{count}'

    def test_key_is_read_from_applies_to(self, api_key, monkeypatch):
        """C3: lyceum-cli 1.1.1 reads this key from `group_by`, which the API
        does not return -- their rate display silently no-ops.

        Derived from the live fixture by deleting `applies_to` entirely: if the
        implementation reads `group_by` first, the result is 48 rows either
        way and the bug goes unnoticed. Here it must come out empty.
        """
        payload = copy.deepcopy(load_fixture('pricing_vm_running'))
        for row in payload['prices']:
            row.pop('applies_to')

        install(monkeypatch, [response(payload)])
        assert client().get_vm_prices() == {}

    def test_group_by_is_accepted_as_a_fallback(self, api_key, monkeypatch):
        """C3 forward-compat: if the vendor ever fixes their own client by
        emitting `group_by`, we must not go blind.

        Derived from the live fixture by moving the key from `applies_to` to
        `group_by`, nothing else changed.
        """
        payload = copy.deepcopy(load_fixture('pricing_vm_running'))
        for row in payload['prices']:
            row['group_by'] = row.pop('applies_to')

        install(monkeypatch, [response(payload)])
        prices = client().get_vm_prices()
        assert len(prices) == 48
        assert prices[('on-demand', 'h100', 1)] == pytest.approx(2.79,
                                                                 abs=1e-4)

    def test_other_meter_slugs_are_ignored(self, api_key, monkeypatch):
        """C3: only `meter_slug == "vm_running"` is compute rental.

        Derived by appending storage and egress meters to the live payload.
        Folding those into the catalog would quote the optimizer a per-GB rate
        as if it were an hourly VM price.
        """
        payload = copy.deepcopy(load_fixture('pricing_vm_running'))
        payload['prices'].append({
            'meter_slug': 'storage_gb_month',
            'resource': 'storage',
            'applies_to': {'hardware_profile': 'on-demand.h100.1x'},
            'unit': 'gb_month',
            'unit_price': '0.10000000',
            'unit_price_per_hour': '0.00013699',
        })
        payload['prices'].append({
            'meter_slug': 'egress_gb',
            'resource': 'network',
            'applies_to': {'hardware_profile': 'on-demand.l40s.8x'},
            'unit': 'gb',
            'unit_price': '0.05000000',
            'unit_price_per_hour': '0.05000000',
        })

        install(monkeypatch, [response(payload)])
        prices = client().get_vm_prices()
        assert len(prices) == 48
        assert prices[('on-demand', 'h100', 1)] == pytest.approx(2.79,
                                                                 abs=1e-4)
        assert prices[('on-demand', 'l40s', 8)] == pytest.approx(9.52,
                                                                 abs=1e-2)

    def test_pricing_endpoint_and_method(self, api_key, monkeypatch):
        """C3: the price source is `/pricing`, not `/vms/availability`.

        `/vms/availability` also carries a `price_per_hour`, but only for x1 --
        reading it there gives the optimizer a 1-GPU price for an 8-GPU node.
        """
        transport = install(monkeypatch, [response('pricing_vm_running')])
        client().get_vm_prices()
        assert len(transport.calls) == 1
        assert transport.calls[0]['method'].upper() == 'GET'
        assert url_path(transport.calls[0]) == '/pricing'


# ---------------------------------------------------------------------------
# C9 -- availability, which has two disagreeing sources
# ---------------------------------------------------------------------------


class TestGetAvailability:

    def test_keyed_by_instance_type_and_profile(self, api_key, monkeypatch):
        """C9: spot and on-demand are separate capacity axes that disagree.

        Keying by profile alone collapses them and offers spot rows that cannot
        provision (fast 500, C7) while hiding on-demand rows that can.
        """
        install(monkeypatch, [response('vms_availability')])
        availability = client().get_availability()

        assert sorted(availability[('on-demand', 'h100')]) == [1, 2, 4, 8]
        assert sorted(availability[('spot', 'h100')]) == [1, 2, 8]
        assert sorted(availability[('on-demand', 'h200')]) == [1, 2]
        assert sorted(availability[('spot', 'h200')]) == [2]

    def test_built_from_instance_variants_not_hardware_profiles(
            self, api_key, monkeypatch):
        """C9: `available_hardware_profiles` is the on-demand view only.

        This is the test that must fail if someone switches the source.
        `l40s` appears in `available_hardware_profiles` and has NO spot
        variant at all, so a `('spot', 'l40s')` key can only come from reading
        the wrong list -- and it would advertise a SKU that can never
        provision. The row counts also differ (6 profiles vs 11 variants).
        """
        source = load_fixture('vms_availability')
        assert len(source['available_hardware_profiles']) == 6  # drift guard
        assert len(source['available_instance_variants']) == 11

        install(monkeypatch, [response('vms_availability')])
        availability = client().get_availability()

        assert ('spot', 'l40s') not in availability
        assert ('on-demand', 'l40s') in availability
        assert len(availability) == 11
        assert {instance_type for instance_type, _ in availability} == {
            'on-demand', 'spot'
        }

    def test_zero_capacity_is_an_empty_list_not_a_missing_key(
            self, api_key, monkeypatch):
        """C9/C11: "variant exists, no capacity right now" and "variant does not
        exist" are different facts and must not be conflated.

        b200 spot is a real variant that was empty during the review; l40s spot
        is not a variant at all. The first is transient (retry in 120 s, per the
        cache TTL); the second is permanent. Dropping empty keys makes them
        indistinguishable.
        """
        install(monkeypatch, [response('vms_availability')])
        availability = client().get_availability()

        assert availability[('on-demand', 'b200')] == []
        assert availability[('spot', 'b300')] == []
        assert ('spot', 'l40s') not in availability

    def test_gpu_counts_are_plain_ints(self, api_key, monkeypatch):
        """The catalog filters rows by `gpu_count in available`; a str/float
        element makes every membership test quietly false and hides the whole
        cloud from the optimizer."""
        install(monkeypatch, [response('vms_availability')])
        for counts in client().get_availability().values():
            for count in counts:
                assert type(count) is int

    def test_availability_endpoint_and_method(self, api_key, monkeypatch):
        """C11: this is cached for only 120 s because it races hard, so it is
        fetched often -- a wrong path here shows up as a permanently empty
        catalog, i.e. Lyceum silently invisible to the optimizer."""
        transport = install(monkeypatch, [response('vms_availability')])
        client().get_availability()
        assert transport.calls[0]['method'].upper() == 'GET'
        assert url_path(transport.calls[0]) == '/vms/availability'


# ---------------------------------------------------------------------------
# user status
# ---------------------------------------------------------------------------


class TestGetUserStatus:

    def test_returns_decoded_body(self, api_key, monkeypatch):
        """`Lyceum.check_credentials` calls this; it must return the payload,
        not a bool, so the failure message can name the account."""
        transport = install(monkeypatch, [response('user_status')])
        status = client().get_user_status()

        assert status['email'] == 'user@example.com'
        assert status['status'] == 'authenticated'
        assert url_path(transport.calls[0]) == '/user/status'


# ---------------------------------------------------------------------------
# credential resolution
# ---------------------------------------------------------------------------


class TestReadAPIKey:
    """A deployed API server seeds `~/.lyceum/api_key` from its own secret store
    while developers use the env var. Both paths must work, and neither may ever
    silently fall through to an empty string."""

    def test_env_var_wins_over_file(self, monkeypatch, tmp_path):
        """A deliberate override must not be shadowed by a stale key file left
        behind in the image."""
        key_file = tmp_path / 'api_key'
        key_file.write_text('lk_from_file')
        monkeypatch.setenv('LYCEUM_API_KEY', 'lk_from_env')

        assert api.read_api_key(str(key_file)) == 'lk_from_env'

    def test_file_is_read_and_stripped(self, monkeypatch, tmp_path):
        """`echo $LYCEUM_API_KEY > ~/.lyceum/api_key` appends a newline.

        Sending `Bearer lk_...\\n` produces a 401 that looks exactly like a
        revoked key, and costs an hour of debugging the wrong thing.
        """
        monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
        key_file = tmp_path / 'api_key'
        key_file.write_text('  lk_from_file\n')

        assert api.read_api_key(str(key_file)) == 'lk_from_file'

    def test_default_path_expands_under_home(self, monkeypatch, tmp_path):
        """The default is `~/.lyceum/api_key`; no test may touch the real one.

        Also pins that `~` is expanded -- a literal `~/.lyceum/api_key` path
        never exists, so the failure would be "credentials missing" on a
        machine where they are present.
        """
        monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
        monkeypatch.setenv('HOME', str(tmp_path))
        key_dir = tmp_path / '.lyceum'
        key_dir.mkdir()
        (key_dir / 'api_key').write_text('lk_from_home\n')

        assert api.API_KEY_PATH == '~/.lyceum/api_key'
        assert api.read_api_key() == 'lk_from_home'

    def test_empty_env_var_falls_back_to_the_file(self, monkeypatch, tmp_path):
        """An unset deployment secret often exports `LYCEUM_API_KEY=` --
        present but empty.

        Treating "set but empty" as a credential sends `Bearer ` and turns a
        missing-secret deploy into a 401 storm instead of a clear error.
        """
        monkeypatch.setenv('LYCEUM_API_KEY', '')
        key_file = tmp_path / 'api_key'
        key_file.write_text('lk_from_file\n')

        assert api.read_api_key(str(key_file)) == 'lk_from_file'

    def test_missing_everywhere_raises_actionable_auth_error(
            self, monkeypatch, tmp_path):
        """`sky check` must tell the operator what to do, not just say no."""
        monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
        monkeypatch.setenv('HOME', str(tmp_path))
        missing = tmp_path / 'nope' / 'api_key'

        with pytest.raises(api.LyceumAuthError) as exc:
            api.read_api_key(str(missing))

        message = str(exc.value)
        assert 'LYCEUM_API_KEY' in message
        assert str(missing) in message or '.lyceum' in message

    def test_whitespace_only_file_raises(self, monkeypatch, tmp_path):
        """A truncated/zero-byte seed must fail here, not as a 401 later."""
        monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
        key_file = tmp_path / 'api_key'
        key_file.write_text('\n  \n')

        with pytest.raises(api.LyceumAuthError):
            api.read_api_key(str(key_file))

    def test_explicit_env_mapping_is_honoured(self, monkeypatch, tmp_path):
        """`read_api_key(env=...)` exists so callers can resolve a credential
        without mutating the process environment; if it silently fell through
        to `os.environ` the isolation would be a fiction."""
        monkeypatch.setenv('LYCEUM_API_KEY', 'lk_from_process_env')
        missing = tmp_path / 'nope'

        assert api.read_api_key(str(missing),
                                env={'LYCEUM_API_KEY':
                                     'lk_injected'}) == 'lk_injected'


class TestClientCredentialResolution:

    def test_explicit_key_wins(self, monkeypatch, tmp_path):
        """A key passed to the constructor must not be overridden by ambient
        environment: autodown and the provisioner may run in the same process
        as other credentials, and picking up the wrong one deletes VMs in
        whichever org the env var happens to name."""
        monkeypatch.setenv('LYCEUM_API_KEY', 'lk_from_env')
        monkeypatch.setenv('HOME', str(tmp_path))
        assert api.LyceumClient(api_key='lk_explicit').api_key == 'lk_explicit'

    def test_falls_back_to_environment(self, api_key, monkeypatch):
        """A deployed container sets the env var; the client must find it
        without being handed the key explicitly at every call site."""
        assert api.LyceumClient().api_key == api_key

    def test_missing_credentials_fail_before_any_http_call(
            self, monkeypatch, tmp_path):
        """An unauthenticated request wastes a round trip and returns a 401
        that reads like a revoked key rather than an absent one."""
        monkeypatch.delenv('LYCEUM_API_KEY', raising=False)
        monkeypatch.setenv('HOME', str(tmp_path))
        transport = install(monkeypatch, [])

        with pytest.raises(api.LyceumAuthError):
            api.LyceumClient().get_user_status()
        assert transport.calls == []

    def test_base_url_is_configurable_and_normalised(self):
        """A trailing slash in an override would produce `//api/v2/...` paths,
        which some gateways 404."""
        assert api.LyceumClient(api_key='k',
                                base_url='https://example.test/').base_url == \
            'https://example.test'


# ===========================================================================
# Added by MUTATION-TESTING REVIEW, 2026-07-31.
#
# Method: an honest reference implementation of `skypilot_lyceum.api` was
# written against the contract above, reached 101/101 on the suite as it then
# stood, and was then mutated one bug at a time (95 mutations
# derived from C1-C12 plus the transport and credential paths). Fifteen
# mutations SURVIVED -- the injected bug was invisible to every existing test.
# Each test below kills one of those survivors and names it in its docstring.
# ===========================================================================


class TestMutationReviewC2Wiring:

    def test_non_default_ssh_port_reaches_the_parsed_vm(self, api_key,
                                                        monkeypatch):
        """SURVIVOR: VM parsing splits the host but pins `ssh_port = 22`.

        `test_non_default_port_round_trips` exercises `parse_ip_address` in
        isolation, but every fixture carries `:22` or a bare IP, so nothing
        pinned the wiring from the parser into `VM.ssh_port`. An implementation
        that strips the suffix and then hardcodes 22 passes the entire suite,
        and produces exactly the C2 failure the parser test exists to prevent:
        an unreachable SSH target on an arbitrary subset of nodes. Derived from
        the live `host:port` fixture; only the port digits are changed.
        """
        payload = copy.deepcopy(load_fixture('vm_status_ready_host_port'))
        payload['ip_address'] = '198.51.100.20:2222'

        install(monkeypatch, [response(payload)])
        vm = client().get_vm('fa292eba-16ba-4dd2-8c31-f58ac11d87bf')

        assert (vm.ip, vm.ssh_port) == ('198.51.100.20', 2222)
        assert vm.is_usable is True


class TestMutationReviewC4Ordering:
    """Both existing ordering tests are satisfied by the server's own order."""

    def test_newest_wins_when_the_server_lists_it_last(self, api_key,
                                                       monkeypatch):
        """SURVIVOR: `find_vms_by_display_name` does not sort at all.

        `test_newest_generation_wins` revives the fixture's *first* row and
        gives it the latest timestamp, so the payload order already equals the
        expected order and an implementation that returns the list untouched
        passes. Here the newest generation is listed LAST, so only a real
        `created_at`-descending sort can produce the expected result. The
        identity rule's step 3 -- "among survivors matching the name, take the
        newest by created_at" -- was otherwise untested.
        """
        payload = copy.deepcopy(load_fixture('vm_list_mixed'))
        payload['vms'][0]['status'] = 'ready'
        payload['vms'][0]['created_at'] = '2026-07-31T09:00:00.000000Z'
        payload['vms'][1]['created_at'] = '2026-07-31T18:00:00.000000Z'

        install(monkeypatch, [response(payload)])
        vms = client().find_vms_by_display_name('sky-cluster-abc')

        assert [vm.vm_id for vm in vms] == [
            '9918b72c1dce41a6875abaf5d5ab64e9',
            'fa292eba-16ba-4dd2-8c31-f58ac11d87bf',
        ]

    def test_ordering_compares_instants_not_strings(self, api_key,
                                                    monkeypatch):
        """SURVIVOR: ordering sorts the raw `created_at` strings byte-wise.

        Every existing ordering payload happens to agree with a lexicographic
        compare, so an implementation that never parses the timestamp passes
        them all -- including `test_ordering_survives_mixed_timestamp_formats`,
        whose entire stated point is that the value must be *parsed*
        (`fromisoformat`, the trailing `Z`, Python 3.10). The API does emit
        offsets: `/user/status` returns `...783206+00:00`. Here
        `16:00+02:00` (= 14:00Z) is the OLDER instant while sorting later as a
        string. Also pins that `created_at` survives parsing at all -- dropping
        it made every existing ordering test pass too.
        """
        payload = copy.deepcopy(load_fixture('vm_list_mixed'))
        payload['vms'][0]['status'] = 'ready'
        payload['vms'][0]['created_at'] = '2026-07-31T16:00:00.000000+02:00'
        payload['vms'][1]['created_at'] = '2026-07-31T15:00:00.000000Z'

        install(monkeypatch, [response(payload)])
        vms = client().find_vms_by_display_name('sky-cluster-abc')

        assert [vm.vm_id for vm in vms] == [
            '9918b72c1dce41a6875abaf5d5ab64e9',
            'fa292eba-16ba-4dd2-8c31-f58ac11d87bf',
        ]
        assert vms[0].created_at == '2026-07-31T15:00:00.000000Z'


class TestMutationReviewC3Precedence:

    def test_applies_to_wins_when_group_by_is_also_present(self, api_key,
                                                           monkeypatch):
        """SURVIVOR: `get_vm_prices` reads `group_by` first, `applies_to` second.

        `test_key_is_read_from_applies_to` deletes `applies_to` and
        `test_group_by_is_accepted_as_a_fallback` moves the key into
        `group_by`; neither presents both, so the precedence the docstring
        states ("Reads ... `applies_to` (C3). Falls back to `group_by`") is
        unpinned and a group_by-first client passes both. lyceum-cli 1.1.1's
        own bug is reading `group_by`; the day the vendor starts emitting it
        with stale content, a group_by-first client quotes the optimizer a
        catalog that does not exist. Derived from the live fixture by adding a
        deliberately wrong `group_by` to every row.
        """
        payload = copy.deepcopy(load_fixture('pricing_vm_running'))
        for row in payload['prices']:
            row['group_by'] = {'hardware_profile': 'spot.b300.8x'}

        install(monkeypatch, [response(payload)])
        prices = client().get_vm_prices()

        assert len(prices) == 48
        assert prices[('on-demand', 'h100', 1)] == pytest.approx(2.79,
                                                                 abs=1e-4)


class TestMutationReviewTransport:
    """The endpoint/verb/auth/timeout checks existed only for `create_vm`."""

    #: (client method, args, fixture, HTTP verb, path)
    ENDPOINTS = [
        ('get_vm', ('9918b72c1dce41a6875abaf5d5ab64e9',),
         'vm_status_ready_bare_ip', 'GET',
         '/vms/9918b72c1dce41a6875abaf5d5ab64e9/status'),
        ('list_vms', (), 'vm_list_mixed', 'GET', '/vms/list'),
        ('terminate_vm', ('9918b72c1dce41a6875abaf5d5ab64e9',),
         'vm_terminate_ok', 'DELETE',
         '/vms/9918b72c1dce41a6875abaf5d5ab64e9'),
        ('get_vm_prices', (), 'pricing_vm_running', 'GET', '/pricing'),
        ('get_availability', (), 'vms_availability', 'GET',
         '/vms/availability'),
        ('get_user_status', (), 'user_status', 'GET', '/user/status'),
    ]

    @pytest.mark.parametrize('method_name,args,fixture_name,verb,path',
                             ENDPOINTS)
    def test_endpoint_verb_and_path(self, api_key, monkeypatch, method_name,
                                    args, fixture_name, verb, path):
        """SURVIVORS: `get_vm` may use any path or verb; `get_user_status` any verb.

        The suite pins method+path for create, list, delete, /pricing and
        /vms/availability, but never for `GET /vms/{id}/status` -- the poll
        endpoint the whole provisioning loop runs on -- and pins only the path,
        not the verb, for `/user/status`. `status_check_url` in the live create
        response ("/api/v2/external/vms/{vm_id}/status") is the authority for
        the shape asserted here.
        """
        transport = install(monkeypatch, [response(fixture_name)])
        getattr(client(), method_name)(*args)

        call = transport.calls[0]
        assert call['method'].upper() == verb
        assert url_path(call) == path

    @pytest.mark.parametrize('method_name,args,fixture_name,verb,path',
                             ENDPOINTS)
    def test_every_request_is_authenticated_and_bounded(
            self, api_key, monkeypatch, method_name, args, fixture_name, verb,
            path):
        """SURVIVORS: Authorization header, and timeout, attached only to POST.

        `test_authorization_header_is_bearer` and `test_timeout_is_always_
        passed` both go through `create_vm`, so a client that authenticates and
        bounds only its POST passes. That client leaves the `/status` poll --
        which runs once a second for up to ten minutes while a GPU bills -- both
        unauthenticated (401 on every poll) and unbounded (a wedged provisioner
        thread with a VM billing behind it, the most expensive failure mode
        there is on a cloud with no TTL), and leaves the DELETE that is the only
        thing that stops the billing in the same state.
        """
        transport = install(monkeypatch, [response(fixture_name)])
        getattr(client(api_key='lk_secret', timeout=17.0), method_name)(*args)

        call = transport.calls[0]
        assert call['headers']['Authorization'] == 'Bearer lk_secret'
        assert call['timeout'] == 17.0


class TestMutationReviewParsing:

    def test_gpu_counts_are_coerced_to_int(self, api_key, monkeypatch):
        """SURVIVOR: `get_availability` passes `available_gpus_per_instance` through.

        `test_gpu_counts_are_plain_ints` asserts the property against a fixture
        whose values are already ints, so it cannot fail and pins nothing: an
        implementation with no conversion at all survives it. This payload is
        derived from the live one with the counts as JSON strings -- the shape
        a vendor serialisation change produces -- and requires normalisation,
        because the catalog filters with `gpu_count in available` and a str
        element makes every membership test quietly false, hiding the whole
        cloud from the optimizer.
        """
        payload = copy.deepcopy(load_fixture('vms_availability'))
        for variant in payload['available_instance_variants']:
            variant['available_gpus_per_instance'] = [
                str(count)
                for count in variant['available_gpus_per_instance']
            ]

        install(monkeypatch, [response(payload)])
        availability = client().get_availability()

        assert sorted(availability[('on-demand', 'h100')]) == [1, 2, 4, 8]
        for counts in availability.values():
            for count in counts:
                assert type(count) is int

    def test_raw_payload_is_retained_on_the_vm(self, api_key, monkeypatch):
        """SURVIVOR: `VM.raw` is populated with an empty dict.

        `raw` is the declared escape hatch for everything the dataclass does
        not model (`uptime_seconds`, `billed`, `name`, `org_id`, and whatever
        the vendor adds next). No test read it, so discarding the payload was
        invisible.
        """
        install(monkeypatch, [response('vm_status_ready_bare_ip')])
        vm = client().get_vm('9918b72c1dce41a6875abaf5d5ab64e9')

        assert vm.raw == load_fixture('vm_status_ready_bare_ip')


class TestMutationReviewListFlags:

    def test_include_failed_is_sent_as_well(self, api_key, monkeypatch):
        """SURVIVOR: `list_vms` omits `include_failed` entirely.

        The identity rule, step 1, specifies
        `?include_terminated=false&include_failed=false`, and
        `test_all_include_flags_default_to_false` only constrains flags that
        are actually sent -- so sending just `include_terminated` passes it
        vacuously. `failed` is one of TERMINAL_STATUSES, so the missing flag
        makes the server hand back dead VMs on every lookup and leaves the
        client-side filter as the sole defence, exactly the belt-and-braces
        C4 asks not to rely on.
        """
        transport = install(monkeypatch, [response('vm_list_mixed')])
        client().list_vms()

        params = transport.calls[0]['params']
        assert params['include_failed'] in (False, 'false')
