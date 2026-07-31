"""Shared test fixtures.

Mocking follows SkyPilot's own idiom (verified across tests/unit_tests/ in
skypilot v0.13.0): `unittest.mock` + pytest `monkeypatch` only. No `responses`,
no `requests-mock`, no cassettes -- that keeps adapted upstream tests
copy-pasteable.

We deviate from upstream in one place: response bodies live in
`tests/fixtures/lyceum_api/*.json`, captured verbatim from the live Lyceum API
on 2026-07-31, rather than being inlined as dicts. API drift then shows up as a
diff in a fixture file instead of a puzzling assertion failure.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest

FIXTURE_DIR = pathlib.Path(__file__).parent / 'fixtures' / 'lyceum_api'


def load_fixture(name: str) -> Any:
    """Load `tests/fixtures/lyceum_api/<name>.json`."""
    path = FIXTURE_DIR / (name if name.endswith('.json') else f'{name}.json')
    if not path.is_file():
        available = sorted(p.stem for p in FIXTURE_DIR.glob('*.json'))
        raise AssertionError(f'no fixture {name!r}; have: {available}')
    return json.loads(path.read_text())


@pytest.fixture
def fixture():
    """Function-scoped accessor so tests can do `fixture('vm_list_mixed')`."""
    return load_fixture


class FakeResponse:
    """Minimal stand-in for `requests.Response`.

    Only the surface `LyceumClient._request` is allowed to touch: `status_code`,
    `json()`, `text`, and `ok`. Keeping it this narrow means the test suite
    fails loudly if the client starts depending on more of requests' API.
    """

    def __init__(self, status_code: int = 200,
                 payload: Optional[Union[Dict, List]] = None,
                 text: Optional[str] = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError('no JSON body')
        return self._payload


def response(name_or_payload: Union[str, Dict, List], status: int = 200) -> FakeResponse:
    """Build a FakeResponse from a fixture name or a literal payload."""
    payload = (load_fixture(name_or_payload)
               if isinstance(name_or_payload, str) else name_or_payload)
    return FakeResponse(status_code=status, payload=payload)


class RecordingTransport:
    """Scripts a sequence of responses and records every call.

    Usage:
        t = RecordingTransport([response('vm_create_pending', 200)])
        monkeypatch.setattr(api.requests, 'request', t)
        ...
        assert t.calls[0]['json']['hardware_profile'] == 'l40s'

    Raises AssertionError if the code under test makes more calls than were
    scripted -- an unscripted call is a bug, not something to paper over with a
    default response.
    """

    def __init__(self, responses: Sequence[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, method: str = None, url: str = None, **kwargs) -> FakeResponse:
        self.calls.append({'method': method, 'url': url, **kwargs})
        if not self._responses:
            raise AssertionError(
                f'unscripted request #{len(self.calls)}: {method} {url}')
        return self._responses.pop(0)

    @property
    def exhausted(self) -> bool:
        return not self._responses


@pytest.fixture
def transport():
    """Factory for RecordingTransport, so tests read `transport([...])`."""
    return RecordingTransport


@pytest.fixture
def api_key(monkeypatch, tmp_path):
    """Provide a credential via env var so no test touches the real ~/.lyceum."""
    monkeypatch.setenv('LYCEUM_API_KEY', 'lk_' + 'f' * 64)
    monkeypatch.setenv('HOME', str(tmp_path))
    return 'lk_' + 'f' * 64


def reset_catalog_caches() -> None:
    """Drop `skypilot_lyceum.catalog`'s module-level TTL caches.

    Lives here, not in `test_catalog.py`, because the catalog's caches are
    process-global while the tests that populate them are not: `test_catalog.py`
    drives the real catalog, and so does `test_cloud_class.py`. Keeping the
    reset module-local meant the second file was only ever clean because
    alphabetical collection order happened to run `test_catalog.py` first and
    its teardown happened to sweep up. That is a genuine cross-file ordering
    hazard -- `-p no:randomly` off, a rename, or running one file alone changes
    the answer -- so the sweep is autouse for the whole suite.
    """
    from skypilot_lyceum import catalog

    reset = getattr(catalog, '_reset_caches', None)
    if callable(reset):
        try:
            reset()
            return
        except NotImplementedError:
            # The hook is declared but not yet built. Fall through to the manual
            # sweep so tests fail on their own assertions rather than every test
            # collapsing into an identical setup error, which destroys the
            # signal this suite exists to provide.
            pass
    for name in list(vars(catalog)):
        if name.startswith('_') and ('CACHE' in name.upper() or
                                     name in ('_df', '_DF')):
            setattr(catalog, name, None)


@pytest.fixture(autouse=True)
def _clean_catalog_state():
    """Reset the catalog's TTL caches around every test in the suite."""
    reset_catalog_caches()
    yield
    reset_catalog_caches()


@pytest.fixture
def lyceum_enabled():
    """Run `enable()` and put every global it touches back afterwards.

    `enable()` mutates process-global state by design: `CLOUD_REGISTRY`,
    `skylet.constants.ALL_CLOUDS`, `sky.provision._registered_provisioners`, and
    a `sys.modules` entry. A test that calls it without restoring leaves the
    suite in a different state than it found it -- which is not hypothetical:
    `test_plugin.py`'s module-scoped leak detector caught exactly that when
    `test_template_override.py` ran before it under a reversed file order.

    `test_registration.py` and `test_plugin.py` keep their own stricter,
    autouse versions of this (they also start from a deliberately cleared
    slate, because they are testing registration itself). This one is for the
    files that merely need the cloud registered.
    """
    import sys

    import sky.provision as sky_provision
    from sky.skylet import constants
    from sky.utils import registry

    import skypilot_lyceum

    catalog_module = 'sky.catalog.lyceum_catalog'
    missing = object()
    reg = registry.CLOUD_REGISTRY
    # pylint: disable=protected-access
    saved = {
        'registry': dict(reg),
        'aliases': dict(getattr(reg, '_aliases', {}) or {}),
        'all_clouds': constants.ALL_CLOUDS,
        'provisioners': dict(sky_provision._registered_provisioners),
        'catalog_module': sys.modules.get(catalog_module, missing),
        # The package modules themselves. `Lyceum` registers via an import-time
        # decorator, so restoring the registry WITHOUT restoring these leaves a
        # state no `enable()` can recover from: the name is absent from the
        # registry, but `skypilot_lyceum.cloud` is still cached, so re-importing
        # it never re-fires the decorator and `enable()` fails loud (correctly).
        'lyceum_modules': {name: module
                           for name, module in sys.modules.items()
                           if name == 'skypilot_lyceum' or
                           name.startswith('skypilot_lyceum.')},
    }
    try:
        skypilot_lyceum.enable()
        yield skypilot_lyceum
    finally:
        reg.clear()
        reg.update(saved['registry'])
        aliases = getattr(reg, '_aliases', None)
        if isinstance(aliases, dict):
            aliases.clear()
            aliases.update(saved['aliases'])
        constants.ALL_CLOUDS = saved['all_clouds']
        sky_provision._registered_provisioners.clear()
        sky_provision._registered_provisioners.update(saved['provisioners'])
        if saved['catalog_module'] is missing:
            sys.modules.pop(catalog_module, None)
        else:
            sys.modules[catalog_module] = saved['catalog_module']
        for name in [n for n in sys.modules
                     if n == 'skypilot_lyceum' or
                     n.startswith('skypilot_lyceum.')]:
            del sys.modules[name]
        sys.modules.update(saved['lyceum_modules'])


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Hard stop: fail any test that opens a real socket.

    The review that produced this suite cost real money on real GPUs. A unit
    test must never be able to repeat that by accident.
    """
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError(
            'a test attempted a real network connection -- mock the transport')

    monkeypatch.setattr(socket.socket, 'connect', _blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', _blocked)
