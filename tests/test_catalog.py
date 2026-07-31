"""Tests for `skypilot_lyceum.catalog` -- the table SkyPilot's optimizer reads.

Adapted from skypilot 0.13.0 `tests/unit_tests/test_catalog.py` (notably
`test_get_hourly_cost_returns_python_float` and
`test_catalog_prices_are_json_serializable`, the numpy.float64 regression
guards for upstream issue #7969) and from the two shipped out-of-tree-shaped
catalogs, `sky/catalog/shadeform_catalog.py` and
`sky/catalog/hyperbolic_catalog.py`.

Everything here mocks at the `LyceumClient` boundary, never at `requests`:
these tests are about building a correct table out of two API payloads, not
about HTTP. The payloads themselves (`pricing_vm_running.json`,
`vms_availability.json`) were captured verbatim from the live API on
2026-07-31, so every price and every capacity list below is a real number.

C-numbers refer to the empirical Lyceum API corrections tabulated in
README.md.
"""
from __future__ import annotations

import ast
import inspect
import json
import pathlib
import time

import numpy as np
import pandas as pd
import pytest

# `tests/` is on sys.path under pytest's default (prepend) import mode, so the
# conftest helpers are importable as a module. Reusing `load_fixture` keeps
# fixture-path knowledge in exactly one place.
from conftest import load_fixture
from conftest import reset_catalog_caches

from skypilot_lyceum import api
from skypilot_lyceum import catalog

# --------------------------------------------------------------------------
# Fixture-derived expectations
# --------------------------------------------------------------------------

#: The nine columns SkyPilot's `sky/catalog/common.py` impl helpers index into.
#: A missing one is not a soft failure: `get_hourly_cost_impl` KeyErrors on
#: 'Price'/'SpotPrice', `list_accelerators_impl` on 'GpuInfo', and
#: `_get_instance_type` on 'Region'.
REQUIRED_COLUMNS = (
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'Price',
    'SpotPrice',
    'Region',
    'GpuInfo',
)

PROFILES = api.HARDWARE_PROFILES
GPU_COUNTS = api.ALLOWED_GPU_COUNTS

#: The vendor bills per second and reports `unit_price_per_hour` as that
#: per-second figure multiplied back up, so most rows carry rounding noise
#: (a100 x1 is '1.59001200', spot h100 x1 is '1.10001600'). A cent-level
#: tolerance pins the price; anything finer would pin the noise. h100
#: on-demand is the exception -- its strings are exact ('2.790000') -- and is
#: asserted exactly below.
PRICE_TOL = 1e-4


def prices_from_fixture() -> dict:
    """Parse the /pricing fixture the way `LyceumClient.get_vm_prices` must.

    Key is `{instance_type}.{profile}.{count}x` under `applies_to` (C3).
    """
    out = {}
    for row in load_fixture('pricing_vm_running')['prices']:
        if row['meter_slug'] != 'vm_running':
            continue
        key = (row.get('applies_to') or row.get('group_by'))['hardware_profile']
        instance_type, profile, count = key.split('.')
        out[(instance_type, profile,
             int(count.rstrip('x')))] = float(row['unit_price_per_hour'])
    return out


def availability_from_fixture() -> dict:
    """Parse /vms/availability from `available_instance_variants` only (C9)."""
    payload = load_fixture('vms_availability')
    return {(v['instance_type'], v['hardware_profile']):
            list(v['available_gpus_per_instance'])
            for v in payload['available_instance_variants']}


def hardware_profile_availability_from_fixture() -> dict:
    """Parse the *wrong* list, so a test can prove we did not use it (C9)."""
    payload = load_fixture('vms_availability')
    return {v['hardware_profile']: list(v['available_gpus_per_instance'])
            for v in payload['available_hardware_profiles']}


def available_pairs(instance_type: str) -> set:
    """{(profile, gpu_count)} with live capacity for one pricing variant."""
    return {(profile, count)
            for (variant, profile), counts in availability_from_fixture().items()
            if variant == instance_type
            for count in counts}


ON_DEMAND_PAIRS = available_pairs('on-demand')
SPOT_PAIRS = available_pairs('spot')
#: Every (profile, count) that can be provisioned at all right now.
OFFERED_PAIRS = ON_DEMAND_PAIRS | SPOT_PAIRS


def row_of(df: 'pd.DataFrame', instance_type: str) -> 'pd.Series':
    matches = df[df['InstanceType'] == instance_type]
    assert len(matches) == 1, (
        f'expected exactly one row for {instance_type!r}, got {len(matches)}:'
        f'\n{matches}')
    return matches.iloc[0]


# --------------------------------------------------------------------------
# The one mocking seam: LyceumClient
# --------------------------------------------------------------------------


class FakeLyceumClient:
    """Records calls and replays the two catalog payloads.

    Deliberately implements only `get_vm_prices` and `get_availability`: if
    the catalog reaches for any other client method (or for `requests`), the
    AttributeError says so instead of a network call quietly happening.
    """

    def __init__(self, prices=None, availability=None, price_error=None,
                 availability_error=None):
        self.prices = dict(prices or {})
        self.availability = dict(availability or {})
        self.price_error = price_error
        self.availability_error = availability_error
        self.price_calls = 0
        self.availability_calls = 0

    def get_vm_prices(self):
        self.price_calls += 1
        if self.price_error is not None:
            raise self.price_error
        return dict(self.prices)

    def get_availability(self):
        self.availability_calls += 1
        if self.availability_error is not None:
            raise self.availability_error
        return {k: list(v) for k, v in self.availability.items()}


def install_client(monkeypatch, fake: FakeLyceumClient) -> FakeLyceumClient:
    """Patch every plausible binding through which the catalog gets a client.

    The catalog may hold the class (`api.LyceumClient`, or re-exported as
    `catalog.LyceumClient`) or an accessor (`catalog._get_client`). All three
    are patched; a fourth path would be a seam this suite cannot mock, i.e. a
    bug worth failing on.
    """
    monkeypatch.setattr(api, 'LyceumClient', lambda *a, **k: fake)
    monkeypatch.setattr(catalog, 'LyceumClient', lambda *a, **k: fake,
                        raising=False)
    monkeypatch.setattr(catalog, '_get_client', lambda *a, **k: fake,
                        raising=False)
    return fake


# `reset_catalog_caches` and the autouse `_clean_catalog_state` fixture that
# drives it live in `conftest.py`: the catalog's caches are process-global and
# `test_cloud_class.py` exercises the real catalog too, so the reset has to
# apply suite-wide rather than depending on this file's collection order.


@pytest.fixture
def lyceum_client(monkeypatch, api_key):
    """A live-API client serving both captured payloads."""
    del api_key  # Ensures no test can read a real ~/.lyceum/api_key.
    return install_client(
        monkeypatch,
        FakeLyceumClient(prices_from_fixture(), availability_from_fixture()))


@pytest.fixture
def offline_client(monkeypatch, api_key):
    """A client where both catalog endpoints fail, forcing the CSV fallback."""
    del api_key
    return install_client(
        monkeypatch,
        FakeLyceumClient(
            price_error=api.LyceumServerError('502 Bad Gateway'),
            availability_error=api.LyceumServerError('502 Bad Gateway')))


@pytest.fixture
def price_only_client(monkeypatch, api_key):
    """/pricing answers, /vms/availability is down. The half-outage (M2).

    `offline_client` fails both endpoints, so nothing until now has exercised
    the case where the catalog has real prices but no capacity signal -- which
    is the more likely outage of the two, since availability is the volatile
    endpoint.
    """
    del api_key
    return install_client(
        monkeypatch,
        FakeLyceumClient(
            prices_from_fixture(),
            availability_error=api.LyceumServerError('502 Bad Gateway')))


class FakeClock:
    """Monotonic time under test control -- never `time.sleep` in a unit test."""

    def __init__(self, now: float = 1_600_000_000.0):
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, 'time', fake)
    monkeypatch.setattr(time, 'monotonic', fake)
    return fake


# --------------------------------------------------------------------------
# Instance-type naming
# --------------------------------------------------------------------------


def test_instance_type_name_uses_the_vendor_key_format():
    """Names must match Lyceum's own `{profile}.{count}x` pricing key (C3).

    Prevents: a catalog whose InstanceType strings cannot be mapped back onto
    a `hardware_profile` + `gpu_count` create payload, which would make every
    launch a 400.
    """
    assert catalog.instance_type_name('h100', 8) == 'h100.8x'
    assert catalog.instance_type_name('l40s', 1) == 'l40s.1x'


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('gpu_count', GPU_COUNTS)
def test_instance_type_name_round_trips(profile, gpu_count):
    """Property: parse(name(p, n)) == (p, n) for all 6 x 4 valid combinations.

    Prevents: the provisioner deriving the wrong profile or GPU count from an
    optimizer-chosen instance type -- i.e. silently launching (and billing)
    hardware nobody asked for.
    """
    name = catalog.instance_type_name(profile, gpu_count)
    assert catalog.parse_instance_type(name) == (profile, gpu_count)


@pytest.mark.parametrize('bad', [
    '',
    'h100',
    'h100.8',
    '8x',
    '.8x',
    'h100.',
    'h100.8x.1',
    'h100.xx',
    'h100.-1x',
    'h100 8x',
])
def test_parse_instance_type_rejects_malformed_input(bad):
    """Structural garbage must raise ValueError, not return a guess.

    Prevents: a typo'd instance type silently resolving to a real profile.
    """
    with pytest.raises(ValueError):
        catalog.parse_instance_type(bad)


@pytest.mark.parametrize('bad', ['h100.0x', 'h100.3x', 'h100.16x', 'v100.1x',
                                 'a10.1x'])
def test_parse_instance_type_rejects_values_the_api_would_reject(bad):
    """Counts outside {1,2,4,8} and unknown profiles must raise (C8).

    `gpu_count: 0` is the dangerous one: the API silently coerces it to 1 and
    provisions a *billing* VM. The catalog is the last place to catch that
    before money is spent, so its parser must refuse the value outright.
    """
    with pytest.raises(ValueError):
        catalog.parse_instance_type(bad)


def test_instance_specs_cover_every_hardware_profile():
    """INSTANCE_SPECS and the API's own profile list must not drift apart.

    Prevents: a profile the API accepts having no vCPU/RAM row, which makes
    `get_vcpus_mem_from_instance_type` return NaN and `--cpus` matching bogus.
    """
    assert set(catalog.INSTANCE_SPECS) == set(PROFILES)
    assert set(catalog.ACCELERATOR_NAMES) == set(PROFILES)


# --------------------------------------------------------------------------
# DataFrame shape
# --------------------------------------------------------------------------


def test_df_has_every_column_the_sky_catalog_helpers_index(lyceum_client):
    """The impl helpers in `sky/catalog/common.py` index these by name.

    Prevents: a KeyError deep inside the optimizer instead of at catalog build.
    """
    df = catalog._get_df()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f'missing catalog columns: {missing}'


def test_df_has_one_row_per_profile_and_gpu_count(lyceum_client):
    """Rows are keyed by (profile x gpu_count); spot is a column, not a row.

    Prevents: duplicate InstanceType rows, which make
    `get_vcpus_mem_from_instance_type_impl` trip its `len(set(...)) == 1`
    assertion and take down every optimizer call.
    """
    df = catalog._get_df()
    instance_types = list(df['InstanceType'])
    assert len(instance_types) == len(set(instance_types)), (
        f'duplicate InstanceType rows: {instance_types}')
    for instance_type in instance_types:
        profile, count = catalog.parse_instance_type(instance_type)
        assert (profile, count) in OFFERED_PAIRS


def test_accelerator_names_are_skypilots_uppercase_names(lyceum_client):
    """`--gpus H100` matches on AcceleratorName; 'h100' silently matches nothing.

    `get_instance_type_for_accelerator_impl` does a case-insensitive
    fullmatch, but `list_accelerators` groups by the raw string and every
    other cloud's catalog is uppercase -- a lowercase Lyceum row would show up
    as a separate, duplicate GPU entry in `sky show-gpus`.
    """
    df = catalog._get_df()
    names = set(df['AcceleratorName'])
    assert names <= {'H100', 'A100', 'L40S', 'H200', 'B200', 'B300'}
    assert names, 'catalog offered no accelerators at all'
    for name in names:
        assert name == name.upper()
    for _, row in df.iterrows():
        profile, _ = catalog.parse_instance_type(row['InstanceType'])
        assert row['AcceleratorName'] == catalog.ACCELERATOR_NAMES[profile]


def test_accelerator_count_matches_the_instance_type(lyceum_client):
    """AcceleratorCount is what the optimizer matches `H100:8` against.

    Prevents: an 8-GPU node being offered as a 1-GPU node (or vice versa),
    which the user only discovers after paying for the wrong machine.
    """
    df = catalog._get_df()
    for _, row in df.iterrows():
        _, count = catalog.parse_instance_type(row['InstanceType'])
        assert int(row['AcceleratorCount']) == count


def test_vcpus_and_memory_scale_with_gpu_count(lyceum_client):
    """Measured per-GPU specs multiplied by the GPU count.

    Prevents: reporting the x1 host size for an x8 node, which makes
    `--cpus 64+` reject a machine that has 256 vCPUs.
    """
    df = catalog._get_df()
    for _, row in df.iterrows():
        profile, count = catalog.parse_instance_type(row['InstanceType'])
        vcpus_per_gpu, mem_per_gpu, _, _ = catalog.INSTANCE_SPECS[profile]
        assert float(row['vCPUs']) == float(vcpus_per_gpu * count)
        assert float(row['MemoryGiB']) == float(mem_per_gpu * count)


def test_every_row_carries_a_region_and_gpuinfo(lyceum_client):
    """Region is the single synthetic 'lyceum'; GpuInfo must never be null.

    `list_accelerators_impl` drops every row whose GpuInfo is NaN when
    `gpus_only=True` -- a null column would make Lyceum invisible in
    `sky show-gpus` while still appearing to work elsewhere.
    """
    df = catalog._get_df()
    assert set(df['Region']) == {catalog.DEFAULT_REGION}
    assert df['GpuInfo'].notna().all()


@pytest.mark.parametrize('client_fixture', ['lyceum_client', 'offline_client'])
def test_gpuinfo_is_the_dict_literal_upstream_parses(request, client_fixture):
    """`sky show-gpus` loses GPU memory unless GpuInfo is `ast.literal_eval`-able.

    `sky/catalog/common.py:756` does `df['GpuInfo'].apply(ast.literal_eval)`
    and reads `['Gpus'][0]['MemoryInfo']['SizeInMiB']` to derive
    DeviceMemoryGiB. It wraps that in `except (ValueError, SyntaxError): df[
    'DeviceMemoryGiB'] = None`, so a malformed value is not an error -- it is
    a silently blank column, for the WHOLE cloud, since `.apply` runs over the
    entire column and one bad row poisons all of them. Verified against the
    installed skypilot 0.13.0: with a bare `L40S` in the column,
    `list_accelerators_impl` returns `device_memory=None`; with the dict
    literal below it returns 44.99 GiB.

    The canonical shape is set by the in-tree fetchers (`fetch_runpod.py:469`,
    `fetch_vast.py:116`): `json.dumps({...}).replace('"', "'")`, i.e. a Python
    dict literal. `catalog.INSTANCE_SPECS` already carries `gpu_mem_mib` per
    profile for exactly this purpose.

    Checked on the offline frame too: the packaged `data/vms.csv` currently
    writes a bare accelerator name in this column, so it needs regenerating in
    this format.
    """
    import sky.catalog.common as sky_catalog_common

    request.getfixturevalue(client_fixture)
    df = catalog._get_df()
    assert not df.empty

    for _, row in df.iterrows():
        profile, count = catalog.parse_instance_type(row['InstanceType'])
        try:
            gpu_info = ast.literal_eval(row['GpuInfo'])
        except (ValueError, SyntaxError) as exc:
            raise AssertionError(
                f'GpuInfo for {row["InstanceType"]} is {row["GpuInfo"]!r}, '
                f'which ast.literal_eval rejects ({exc}); upstream will '
                'silently blank DeviceMemoryGiB for every Lyceum row') from exc
        assert isinstance(gpu_info, dict), gpu_info
        gpus = gpu_info['Gpus']
        assert len(gpus) == 1, gpu_info
        gpu_mem_mib = catalog.INSTANCE_SPECS[profile][2]
        assert gpus[0]['Name'] == catalog.ACCELERATOR_NAMES[profile]
        assert gpus[0]['Count'] == count
        assert gpus[0]['MemoryInfo']['SizeInMiB'] == gpu_mem_mib
        assert gpu_info['TotalGpuMemoryInMiB'] == gpu_mem_mib * count

    # End to end through the upstream helper that consumes the column: this is
    # the assertion that actually protects `sky show-gpus`.
    listed = sky_catalog_common.list_accelerators_impl('Lyceum', df, True,
                                                       None, None, None)
    assert listed, 'upstream list_accelerators_impl dropped every row'
    for name, infos in listed.items():
        for info in infos:
            profile, _ = catalog.parse_instance_type(info.instance_type)
            assert info.device_memory is not None, (
                f'{info.instance_type}: upstream derived no DeviceMemoryGiB '
                'from GpuInfo')
            assert float(info.device_memory) == pytest.approx(
                catalog.INSTANCE_SPECS[profile][2] / 1024.0, rel=1e-6), name


# --------------------------------------------------------------------------
# Price correctness (C3)
# --------------------------------------------------------------------------


def test_on_demand_h100_prices_are_the_exact_metered_values(lyceum_client):
    """h100 is the one profile whose /pricing strings are exact.

    2.790000 / 5.58000 / 11.1600 / 22.3200 in the captured payload. Pins that
    the price came from the `vm_running` meter and was not silently rescaled.
    """
    df = catalog._get_df().set_index('InstanceType')
    assert float(df.loc['h100.1x', 'Price']) == 2.79
    assert float(df.loc['h100.2x', 'Price']) == 5.58
    assert float(df.loc['h100.4x', 'Price']) == 11.16
    assert float(df.loc['h100.8x', 'Price']) == 22.32


def test_prices_match_the_pricing_payload_row_for_row(lyceum_client):
    """Every offered row's Price/SpotPrice equals its /pricing meter row.

    Tolerance is cent-level because the vendor's per-hour figure is a
    per-second rate multiplied back up (a100 x1 reports 1.59001200).

    Prevents: sourcing price from /vms/availability, which only publishes a
    single per-profile `price_per_hour` and would price l40s.8x at 1.19.
    """
    prices = prices_from_fixture()
    df = catalog._get_df()
    for _, row in df.iterrows():
        profile, count = catalog.parse_instance_type(row['InstanceType'])
        assert float(row['Price']) == pytest.approx(
            prices[('on-demand', profile, count)], abs=PRICE_TOL)
        if not pd.isna(row['SpotPrice']):
            assert float(row['SpotPrice']) == pytest.approx(
                prices[('spot', profile, count)], abs=PRICE_TOL)


def test_price_is_linear_in_gpu_count(lyceum_client):
    """price(profile, N) / price(profile, 1) == N, for both pricing variants.

    Asserted as a ratio, not as hardcoded per-count numbers, so it survives a
    vendor repricing while still catching the failure it exists for: a catalog
    that copies the per-profile rate from /vms/availability onto every GPU
    count would make an x8 node look 8x cheaper than it is and win every
    optimizer comparison.
    """
    df = catalog._get_df().set_index('InstanceType')
    for profile in PROFILES:
        base_name = catalog.instance_type_name(profile, 1)
        if base_name not in df.index:
            continue
        for column in ('Price', 'SpotPrice'):
            base = df.loc[base_name, column]
            if pd.isna(base):
                continue
            for count in GPU_COUNTS:
                name = catalog.instance_type_name(profile, count)
                if name not in df.index or pd.isna(df.loc[name, column]):
                    continue
                assert float(df.loc[name, column]) / float(base) == (
                    pytest.approx(count, rel=1e-3)), (
                        f'{column} for {name} is not linear in gpu_count')


def test_spot_is_cheaper_than_on_demand_on_every_row(lyceum_client):
    """A spot row priced at or above on-demand means the columns got swapped.

    Prevents: the optimizer picking a preemptible machine that costs more
    than the reliable one.
    """
    df = catalog._get_df()
    checked = 0
    for _, row in df.iterrows():
        if pd.isna(row['SpotPrice']):
            continue
        assert float(row['SpotPrice']) < float(row['Price']), row
        checked += 1
    assert checked, 'no spot rows were offered at all -- C9 filter too broad?'


def test_get_hourly_cost_selects_the_requested_pricing_variant(lyceum_client):
    """use_spot picks SpotPrice, not-use_spot picks Price.

    Prevents: cost reports and optimizer rankings quoting the wrong variant,
    the 2.5x error between h100 on-demand (2.79) and spot (1.10).
    """
    prices = prices_from_fixture()
    on_demand = catalog.get_hourly_cost('h100.1x', use_spot=False)
    spot = catalog.get_hourly_cost('h100.1x', use_spot=True)
    assert on_demand == pytest.approx(prices[('on-demand', 'h100', 1)],
                                      abs=PRICE_TOL)
    assert spot == pytest.approx(prices[('spot', 'h100', 1)], abs=PRICE_TOL)
    assert spot < on_demand


# --------------------------------------------------------------------------
# No numpy / Decimal leaks (upstream issue #7969)
# --------------------------------------------------------------------------


def test_get_hourly_cost_returns_python_float(lyceum_client):
    """`type(cost) is float` exactly -- numpy.float64 breaks orjson.

    Adapted from upstream `test_get_hourly_cost_returns_python_float`. Note
    `isinstance(np.float64(1), float)` is True, so isinstance is not a
    sufficient assertion; identity of the type is.

    Prevents: the API server 500ing on `sky cost-report` / any response that
    embeds a catalog price (upstream issue #7969). The prices arrive from
    /pricing as JSON *strings*, so a Decimal leak is the other live risk here.
    """
    for instance_type in catalog._get_df()['InstanceType']:
        for use_spot in (False, True):
            row = row_of(catalog._get_df(), instance_type)
            if use_spot and pd.isna(row['SpotPrice']):
                continue
            cost = catalog.get_hourly_cost(instance_type, use_spot=use_spot)
            assert type(cost) is float, (
                f'{instance_type} use_spot={use_spot} returned '
                f'{type(cost)!r}')
            assert not isinstance(cost, np.floating)


def test_vcpus_and_memory_are_python_numbers(lyceum_client):
    """Same #7969 guard for the vCPU/memory pair.

    `get_vcpus_mem_from_instance_type` feeds resource records that the API
    server serializes; a numpy scalar there fails the same way a price does.
    """
    for instance_type in catalog._get_df()['InstanceType']:
        vcpus, mem = catalog.get_vcpus_mem_from_instance_type(instance_type)
        for value in (vcpus, mem):
            assert type(value) in (float, int), (
                f'{instance_type}: {value!r} is {type(value)!r}')
            assert not isinstance(value, np.generic)


def test_catalog_return_values_are_json_serializable(lyceum_client):
    """json.dumps over everything the catalog hands back.

    Adapted from upstream `test_catalog_prices_are_json_serializable`.
    `json.dumps` raises TypeError on numpy scalars and on Decimal, which is
    exactly the class of leak that took down the API server in #7969.
    """
    df = catalog._get_df()
    payload = {
        'instance_types': list(df['InstanceType']),
        'exists': catalog.instance_type_exists('h100.1x'),
        'default': catalog.get_default_instance_type(),
        'regions': [region.name for region in catalog.regions()],
        'validated': list(catalog.validate_region_zone(None, None)),
        'costs': {},
        'accelerators': {},
        'vcpus_mem': {},
        'for_accelerator': list(
            catalog.get_instance_type_for_accelerator('H100', 1)),
    }
    for instance_type in df['InstanceType']:
        payload['costs'][instance_type] = catalog.get_hourly_cost(
            instance_type, use_spot=False)
        payload['accelerators'][instance_type] = (
            catalog.get_accelerators_from_instance_type(instance_type))
        payload['vcpus_mem'][instance_type] = list(
            catalog.get_vcpus_mem_from_instance_type(instance_type))
    encoded = json.dumps(payload)
    assert json.loads(encoded)['exists'] is True


def test_list_accelerators_returns_json_serializable_values(lyceum_client):
    """`sky show-gpus` output must survive orjson too.

    Upstream's `list_accelerators_impl` builds InstanceTypeInfo straight out
    of DataFrame rows, so it hands back numpy scalars unless the cloud's
    catalog coerces them. That is precisely the #7969 leak, so the Lyceum
    catalog must coerce before returning.
    """
    result = catalog.list_accelerators(gpus_only=True)
    assert result, 'list_accelerators returned nothing'
    json.dumps({name: [list(info) for info in infos]
                for name, infos in result.items()})
    for infos in result.values():
        for info in infos:
            for value in info:
                assert not isinstance(value, np.generic), info


# --------------------------------------------------------------------------
# Availability filtering (C9)
# --------------------------------------------------------------------------


def test_offered_rows_match_available_instance_variants(lyceum_client):
    """Rows are exactly the (profile, count) pairs with live capacity (C9/C11).

    Derived from the fixture rather than hardcoded, plus three named cases:
    a100.4x, h200.4x and h200.8x are priced but have no capacity in either
    variant, so offering them would hand the optimizer a plan that always
    ends in a 500 (C7).
    """
    df = catalog._get_df()
    offered = {catalog.parse_instance_type(name)
               for name in df['InstanceType']}
    assert offered == OFFERED_PAIRS
    for absent in ('a100.4x', 'h200.4x', 'h200.8x'):
        assert absent not in set(df['InstanceType'])


def test_spot_l40s_is_not_offered_although_it_is_priced(lyceum_client):
    """The C9 trap: /pricing has spot l40s, /vms/availability does not.

    `available_hardware_profiles` reports l40s available at [8,1,2,4] with no
    mention of pricing variant; only `available_instance_variants` reveals
    that l40s has *no spot variant at all*. Sourcing the filter from the
    former offers a spot l40s that can never provision -- and hides
    combinations that can.

    This test fails if the filter is keyed on profile instead of
    (profile, instance_type).
    """
    # The trap must still be live in the fixture, or this test proves nothing.
    priced = prices_from_fixture()
    assert ('spot', 'l40s', 1) in priced
    assert 'l40s' in hardware_profile_availability_from_fixture()
    assert not any(profile == 'l40s' for profile, _ in SPOT_PAIRS)

    df = catalog._get_df()
    l40s_rows = df[df['AcceleratorName'] == 'L40S']
    assert not l40s_rows.empty, 'on-demand l40s has capacity and must be offered'
    assert l40s_rows['Price'].notna().all()
    assert l40s_rows['SpotPrice'].isna().all(), (
        'a spot l40s row was offered; the availability filter read '
        'available_hardware_profiles instead of available_instance_variants')

    on_demand_types, _ = catalog.get_instance_type_for_accelerator(
        'L40S', 1, use_spot=False)
    assert on_demand_types == ['l40s.1x']
    spot_types, _ = catalog.get_instance_type_for_accelerator(
        'L40S', 1, use_spot=True)
    assert not spot_types, (
        f'optimizer was offered spot L40S: {spot_types}')


def test_profile_without_capacity_yields_no_rows(lyceum_client):
    """b200/b300 report `available_gpus_per_instance: []` in both variants.

    Prevents: the optimizer ranking a $63.92/h b300 x8 first (it is the
    biggest machine on offer) and then failing every launch with a C7 500.
    """
    for profile in ('b200', 'b300'):
        assert not any(p == profile for p, _ in OFFERED_PAIRS)
    df = catalog._get_df()
    assert df[df['AcceleratorName'].isin(['B200', 'B300'])].empty
    for count in GPU_COUNTS:
        assert not catalog.instance_type_exists(f'b200.{count}x')
        assert not catalog.instance_type_exists(f'b300.{count}x')


@pytest.mark.parametrize('instance_type', ['h100.4x', 'a100.8x', 'h200.1x'])
def test_rows_without_spot_capacity_have_no_spot_price(lyceum_client,
                                                       instance_type):
    """Spot and on-demand are separate capacity axes on the same row (C9).

    In the captured payload spot h100 is [1,8,2] (no 4), spot a100 is [1,2]
    (no 8) and spot h200 is [2] (no 1) -- each of those rows is on-demand-only
    even though /pricing quotes a spot rate for it.
    """
    profile, count = catalog.parse_instance_type(instance_type)
    assert (profile, count) in ON_DEMAND_PAIRS
    assert (profile, count) not in SPOT_PAIRS

    row = row_of(catalog._get_df(), instance_type)
    assert not pd.isna(row['Price'])
    assert pd.isna(row['SpotPrice'])


def test_capacity_disappearing_removes_the_row(lyceum_client, clock):
    """Capacity vanished within minutes during the live review (C11).

    After the availability TTL the catalog must reflect the new capacity, not
    a frozen snapshot.
    """
    assert catalog.instance_type_exists('h200.2x')
    lyceum_client.availability[('on-demand', 'h200')] = []
    lyceum_client.availability[('spot', 'h200')] = []
    clock.advance(catalog.AVAILABILITY_TTL_S + 1)
    assert not catalog.instance_type_exists('h200.2x')


# --------------------------------------------------------------------------
# Caching and TTLs (C11)
# --------------------------------------------------------------------------


def test_availability_is_cached_far_more_briefly_than_price():
    """The *relationship* is the requirement; the exact seconds are a tunable.

    C11 is what makes this an invariant rather than a preference: during the
    live review capacity went from advertised to gone -- and stayed gone --
    inside a single 7-minute window, while price only ever moves on a vendor
    pricing event. So availability must expire far sooner than price, or the
    catalog spends that window handing the optimizer plans that end in a C7
    500. Both values must also stay inside a defensible band: an availability
    TTL short enough to sit well inside that observed window but long enough
    that caching is worth doing at all, and a price TTL that still picks up a
    repricing the same day.

    Pinning either constant to the minute (`== 3600`) would turn a legitimate
    retune into a red suite while catching nothing the band below misses.
    """
    assert 30 <= catalog.AVAILABILITY_TTL_S <= 300, (
        f'availability TTL {catalog.AVAILABILITY_TTL_S}s is outside the band '
        'the C11 observations support')
    assert 300 <= catalog.PRICE_TTL_S <= 86_400, (
        f'price TTL {catalog.PRICE_TTL_S}s is outside the sane band')
    assert catalog.AVAILABILITY_TTL_S * 4 <= catalog.PRICE_TTL_S, (
        'availability must expire much sooner than price, not merely sooner')


def test_second_call_within_the_ttl_does_not_hit_the_client(lyceum_client,
                                                            clock):
    """Two _get_df() calls a second apart make one round trip each way.

    Prevents: an uncached catalog hammering /pricing and /vms/availability on
    every optimizer iteration -- the optimizer calls it per candidate.
    """
    catalog._get_df()
    assert lyceum_client.price_calls == 1
    assert lyceum_client.availability_calls == 1
    clock.advance(1)
    catalog._get_df()
    assert lyceum_client.price_calls == 1
    assert lyceum_client.availability_calls == 1


def test_availability_expires_before_price(lyceum_client, clock):
    """At t+121 s availability is refetched and price is not.

    This is the whole point of two TTLs: 48 static price rows must not force a
    120 s refresh, and volatile capacity must not inherit a 1 h lifetime.
    """
    catalog._get_df()
    clock.advance(catalog.AVAILABILITY_TTL_S + 1)
    catalog._get_df()
    assert lyceum_client.availability_calls == 2, (
        'availability was served stale past its TTL')
    assert lyceum_client.price_calls == 1, (
        'price was refetched though its TTL had not expired')


def test_price_is_refetched_after_its_ttl(lyceum_client, clock):
    """A vendor repricing must land within the hour.

    Prevents: a price cache with no expiry at all, which would keep quoting
    yesterday's rates until the API server restarts.
    """
    catalog._get_df()
    clock.advance(catalog.PRICE_TTL_S + 1)
    catalog._get_df()
    assert lyceum_client.price_calls == 2


# --------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------


def test_falls_back_to_the_baked_csv_when_the_api_fails(offline_client):
    """An unreachable Lyceum API must not empty the catalog.

    Prevents: the failure mode the upstream Shadeform stub demonstrates --
    degrade to an empty DataFrame with a warning, at which point the optimizer
    simply never proposes Lyceum and nobody notices until the bill for the
    other cloud arrives.
    """
    df = catalog._get_df()
    assert not df.empty
    assert offline_client.price_calls >= 1
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f'fallback frame missing columns: {missing}'
    assert set(df['InstanceType']) == {
        catalog.instance_type_name(profile, count)
        for profile in PROFILES for count in GPU_COUNTS}
    assert float(row_of(df, 'h100.1x')['Price']) == 2.79


def test_fallback_frame_is_never_silently_empty(offline_client):
    """Both endpoints down is still a usable catalog, or a loud failure.

    An empty frame here is the silent-invisibility bug; if the baked CSV is
    also unreadable the catalog must raise rather than return nothing.
    """
    df = catalog._get_df()
    assert len(df) > 0


def test_availability_outage_still_offers_every_priced_row(price_only_client):
    """/pricing up, /vms/availability down: serve everything, filter nothing.

    The availability filter is an optimizer-quality measure, not a cost guard,
    and C7/C11 are why: a doomed combination is refused at create in
    2.7-9.1 s, costing nothing, and SkyPilot fails over. So losing
    the capacity signal must degrade to "offer the priced rows" and never to
    "offer nothing".

    The failure this exists for is the naive one: treat the availability
    exception as an empty availability map, inner-join the price rows against
    it, and every row disappears. The catalog then returns an empty frame, the
    optimizer stops proposing Lyceum entirely, and nobody finds out until the
    bill for the cloud it picked instead arrives. That must never happen.
    `offline_client` cannot catch it, because with prices gone too the CSV
    fallback masks the join.
    """
    prices = prices_from_fixture()
    df = catalog._get_df()

    assert not df.empty, (
        'an availability outage emptied the catalog; Lyceum just vanished '
        'from the optimizer')
    assert price_only_client.price_calls >= 1
    assert price_only_client.availability_calls >= 1, (
        'availability was never even attempted')

    assert set(df['InstanceType']) == {
        catalog.instance_type_name(profile, count)
        for (_, profile, count) in prices}, (
            'rows were dropped even though the only thing that failed was the '
            'advisory capacity signal')

    # The rows live availability would have suppressed must be present here:
    # b200/b300 have no capacity at any count, and l40s has no spot variant.
    for absent_when_live in ('b200.1x', 'b300.8x', 'h200.4x', 'a100.4x'):
        assert absent_when_live in set(df['InstanceType'])
    assert not pd.isna(row_of(df, 'l40s.1x')['SpotPrice'])

    for _, row in df.iterrows():
        profile, count = catalog.parse_instance_type(row['InstanceType'])
        assert float(row['Price']) == pytest.approx(
            prices[('on-demand', profile, count)], abs=PRICE_TOL)
        assert float(row['SpotPrice']) == pytest.approx(
            prices[('spot', profile, count)], abs=PRICE_TOL)


def test_offline_fallback_never_reaches_skypilots_hosted_catalog(
        monkeypatch, offline_client):
    """The fallback is the packaged CSV, not catalog.skypilot.co.

    `sky.catalog.common.read_catalog` downloads from HOSTED_CATALOG_DIR_URL
    and, when that fails, logs a warning and yields an empty frame. Lyceum is
    not published there, so any use of it is both a pointless network round
    trip inside the optimizer and a silent path to zero rows.

    Asserted behaviourally only: the monkeypatched `read_catalog` explodes if
    it is ever called. An earlier version of this test also grepped
    `inspect.getsource(catalog)` for the string 'read_catalog', which banned
    even a comment explaining why the hosted catalog is avoided -- exactly the
    comment that belongs in that file.
    """
    import requests
    import sky.catalog.common as sky_catalog_common

    def _boom(*args, **kwargs):
        raise AssertionError(
            'the Lyceum catalog attempted to use SkyPilot\'s hosted catalog')

    monkeypatch.setattr(sky_catalog_common, 'read_catalog', _boom)
    monkeypatch.setattr(requests, 'get', _boom)
    monkeypatch.setattr(requests, 'request', _boom)

    df = catalog._get_df()
    assert not df.empty


def test_offline_fallback_offers_spot_l40s_and_that_is_deliberate(
        offline_client, monkeypatch, api_key):
    """Accepted degradation, asserted so it cannot happen by accident.

    The baked CSV is generated from /pricing alone, which *does* quote a spot
    l40s rate; only the live /vms/availability payload knows that no spot l40s
    variant exists (C9). With the API down we cannot know, so the offline
    catalog offers the row. A launch that picks it fails fast and free at
    create with the C7 capacity 500 (2.7-9.1 s, no VM, no charge) and SkyPilot
    fails over. The alternative -- hardcoding today's capacity into the CSV --
    would bake a snapshot that the review showed goes stale in minutes.

    The live path must still filter it out; that asymmetry is the point.
    """
    del api_key
    offline_row = row_of(catalog._get_df(), 'l40s.1x')
    assert not pd.isna(offline_row['SpotPrice']), (
        'the baked CSV lost its spot l40s price; if that was intentional, '
        'update this test and the fallback rationale together')

    reset_catalog_caches()
    install_client(
        monkeypatch,
        FakeLyceumClient(prices_from_fixture(), availability_from_fixture()))
    live_row = row_of(catalog._get_df(), 'l40s.1x')
    assert pd.isna(live_row['SpotPrice']), (
        'live availability must suppress the spot l40s row even though the '
        'offline fallback offers it')


# --------------------------------------------------------------------------
# Measured vs extrapolated specs
# --------------------------------------------------------------------------


def test_unmeasured_profiles_are_flagged_in_instance_specs():
    """b200/b300 never had capacity during the review; their specs are guesses.

    The 4th tuple element is the measured? flag. Prevents: extrapolated vCPU
    and RAM numbers quietly acquiring the authority of measured ones.
    """
    unmeasured = {profile for profile, spec in catalog.INSTANCE_SPECS.items()
                  if not spec[3]}
    assert unmeasured == {'b200', 'b300'}
    for profile in ('l40s', 'h100', 'a100', 'h200'):
        assert catalog.INSTANCE_SPECS[profile][3] is True


def test_measured_flag_is_carried_onto_every_catalog_row(offline_client):
    """The flag must reach a surface, not just live in a dict.

    Checked on the offline frame because that is the only one containing
    b200/b300 rows -- live availability filters them out entirely. The baked
    CSV carries the flag as a `SpecsMeasured` column.
    """
    df = catalog._get_df()
    assert 'SpecsMeasured' in df.columns, (
        'no surface exposes whether a row\'s vCPU/RAM were measured')
    for _, row in df.iterrows():
        profile, _ = catalog.parse_instance_type(row['InstanceType'])
        assert bool(row['SpecsMeasured']) is catalog.INSTANCE_SPECS[profile][3]
    unmeasured_rows = df[~df['SpecsMeasured'].astype(bool)]
    assert set(unmeasured_rows['AcceleratorName']) == {'B200', 'B300'}


@pytest.mark.xfail(
    strict=True,
    reason='b200/b300 vCPU/RAM are still extrapolated -- neither had capacity '
    'during the 2026-07-31 review. This xfail is a deliberate reminder: the '
    'first job that lands on a Blackwell node should measure it, set the '
    'INSTANCE_SPECS flag to True, and delete this marker. strict=True means '
    'the suite goes red the moment the flag flips, so the reminder cannot '
    'rot silently.')
def test_all_profiles_have_measured_specs():
    """Every INSTANCE_SPECS row should eventually be measured, not guessed."""
    assert all(spec[3] for spec in catalog.INSTANCE_SPECS.values())


# --------------------------------------------------------------------------
# Catalog interface completeness
# --------------------------------------------------------------------------


def _dispatched_method_names() -> set:
    """Every method name `sky.catalog.__init__` may getattr off our module.

    Derived by parsing the dispatcher's source for the string literal in
    `_map_clouds_catalog(clouds, '<name>', ...)`, rather than guessed.
    """
    import sky.catalog as sky_catalog
    tree = ast.parse(pathlib.Path(sky_catalog.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, 'id', None) != '_map_clouds_catalog':
            continue
        for arg in node.args[1:2]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
    assert names, 'could not parse the dispatcher; upstream layout changed'
    return names


def _reference_catalog_functions() -> set:
    """Public functions implemented by the two shipped reference catalogs."""
    import importlib
    names = set()
    for module_name in ('sky.catalog.shadeform_catalog',
                        'sky.catalog.hyperbolic_catalog'):
        module = importlib.import_module(module_name)
        names |= {name for name, obj in vars(module).items()
                  if inspect.isfunction(obj) and
                  obj.__module__ == module_name and
                  not name.startswith('_')}
    return names


def test_catalog_implements_every_function_the_dispatcher_may_call():
    """Missing one is an AttributeError raised inside a thread pool.

    `_map_clouds_catalog` getattrs the method off `sky.catalog.lyceum_catalog`
    and runs it via `run_in_parallel` across all clouds, so one missing
    function does not just break Lyceum -- it breaks `sky show-gpus` and the
    optimizer for every cloud in the same call.

    The required set is derived: names the dispatcher dispatches, intersected
    with what the two shipped reference catalogs (shadeform, hyperbolic)
    actually implement.
    """
    required = _dispatched_method_names() & _reference_catalog_functions()
    assert 'get_hourly_cost' in required, 'derivation produced nonsense'
    missing = sorted(name for name in required
                     if not callable(getattr(catalog, name, None)))
    assert not missing, f'lyceum_catalog is missing: {missing}'


#: Every observed call shape for each method in `sky/catalog/__init__.py`, as
#: `_map_clouds_catalog(clouds, name, *args, **kwargs)` forwards them. A list
#: per method because `list_accelerators` is dispatched two different ways.
DISPATCHER_CALL_CONVENTION = {
    'instance_type_exists': [(('h100.1x',), {})],
    'validate_region_zone': [((None, None), {})],
    'regions': [((), {})],
    'get_region_zones_for_instance_type': [(('h100.1x', False), {})],
    'get_hourly_cost': [(('h100.1x', False, None, None), {})],
    'get_vcpus_mem_from_instance_type': [(('h100.1x',), {})],
    'get_default_instance_type': [((None, None, None, None, None, None, False,
                                    None), {})],
    'get_accelerators_from_instance_type': [(('h100.1x',), {})],
    'get_instance_type_for_accelerator': [
        (('H100', 1, None, None, False, None, None, None, None), {})],
    'list_accelerators': [
        # sky/catalog/__init__.py:75 -- fully positional.
        ((True, None, None, None, True, False, True), {}),
        # sky/catalog/__init__.py:100 -- the SAME method, with two arguments
        # passed BY KEYWORD. Parameter *names* are therefore part of this
        # contract, not merely their order.
        ((True, None, None, None), {'all_regions': False,
                                    'require_price': False}),
    ],
}


def _reference_parameter_names(name: str):
    """Parameter names both shipped reference catalogs agree on, or None.

    Derived rather than hardcoded: if `shadeform_catalog` and
    `hyperbolic_catalog` declare the same parameter names for a function, that
    is the in-tree convention and the dispatcher is free to start passing any
    of them by keyword (it already does for `list_accelerators`).
    """
    import importlib
    signatures = []
    for module_name in ('sky.catalog.shadeform_catalog',
                        'sky.catalog.hyperbolic_catalog'):
        function = getattr(importlib.import_module(module_name), name, None)
        if function is not None and inspect.isfunction(function):
            signatures.append(
                tuple(inspect.signature(function).parameters))
    if len(signatures) == 2 and signatures[0] == signatures[1]:
        return signatures[0]
    return None


@pytest.mark.parametrize('name', sorted(DISPATCHER_CALL_CONVENTION))
def test_catalog_functions_accept_the_dispatchers_call_convention(name):
    """Signatures must bind every shape the dispatcher forwards -- and only those.

    `_map_clouds_catalog` passes everything through, so an extra or missing
    parameter is a TypeError at optimizer time, not an import error. Note
    `get_instance_type_for_accelerator` is forwarded nine positional arguments
    -- including `local_disk` and `max_hourly_cost`, which both reference
    catalogs declare.

    `.bind()` alone is far too weak a check, and was measured to be: replacing
    a catalog function with `lambda *a, **k: True` left all 29 interface tests
    green, because `*args, **kwargs` binds anything. So variadic parameters are
    rejected outright, and the parameter names are compared against the two
    shipped reference catalogs wherever they agree -- `list_accelerators` is
    already dispatched with `all_regions=` and `require_price=` as keywords, so
    a renamed parameter is a live TypeError, and a `**kwargs` would swallow
    them into nothing.
    """
    function = getattr(catalog, name, None)
    assert callable(function), f'lyceum_catalog has no {name}()'
    signature = inspect.signature(function)

    for args, kwargs in DISPATCHER_CALL_CONVENTION[name]:
        signature.bind(*args, **kwargs)

    variadic = [
        f'{parameter.name} ({parameter.kind.name})'
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD)
    ]
    assert not variadic, (
        f'{name}{signature} declares {variadic}; a catch-all signature binds '
        'every dispatcher call while implementing none of the contract, which '
        'is precisely what these tests must not accept')

    reference_names = _reference_parameter_names(name)
    if reference_names is not None:
        assert tuple(signature.parameters) == reference_names, (
            f'{name} declares {tuple(signature.parameters)}; both shipped '
            f'reference catalogs declare {reference_names}, and the dispatcher '
            'passes some of these by keyword')


# --------------------------------------------------------------------------
# validate_region_zone
# --------------------------------------------------------------------------


@pytest.mark.parametrize('zone', ['lyceum-a', 'a', 'us-east-1a'])
def test_validate_region_zone_rejects_any_zone(lyceum_client, zone):
    """Lyceum has no zones; accepting one would silently ignore the request.

    Mirrors `hyperbolic_catalog.validate_region_zone`, which raises rather
    than dropping the argument.
    """
    with pytest.raises(ValueError):
        catalog.validate_region_zone(None, zone)
    with pytest.raises(ValueError):
        catalog.validate_region_zone(catalog.DEFAULT_REGION, zone)


@pytest.mark.parametrize('region', [None, 'lyceum'])
def test_validate_region_zone_accepts_the_single_region(lyceum_client, region):
    """`infra: lyceum` and `infra: lyceum/lyceum` must both validate."""
    validated_region, validated_zone = catalog.validate_region_zone(region, None)
    assert validated_zone is None
    if region is not None:
        assert validated_region == catalog.DEFAULT_REGION


@pytest.mark.parametrize('region', ['us-east-1', 'lyceum-west', 'LYCEUM2'])
def test_validate_region_zone_rejects_unknown_regions(lyceum_client, region):
    """A typo'd region must fail loudly, not fall back to the only one."""
    with pytest.raises(ValueError):
        catalog.validate_region_zone(region, None)


def test_regions_reports_exactly_one_zoneless_region(lyceum_client):
    """`regions()` feeds the provisioner's failover loop.

    Prevents: an empty region list, which makes the optimizer treat Lyceum as
    having nowhere to launch.
    """
    regions = catalog.regions()
    assert [region.name for region in regions] == [catalog.DEFAULT_REGION]
    for region in regions:
        assert not region.zones
