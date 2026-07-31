"""Lyceum catalog: implements the `sky.catalog.<cloud>_catalog` interface.

Injected into `sys.modules` as `sky.catalog.lyceum_catalog` by `patches.py`,
because `sky.catalog.__init__` dispatches via a hardcoded
`importlib.import_module(f'sky.catalog.{cloud}_catalog')` with no registry.

Rows are (hardware_profile x gpu_count x instance_type). Price comes from
`/pricing` rows with meter_slug == 'vm_running' -- 48 of them, which IS the
catalog (C3). Availability comes from `available_instance_variants` keyed by
(profile, instance_type), since spot and on-demand are separate axes (C9).

vCPU/RAM are exposed by no API endpoint and are carried from measurement; see
INSTANCE_SPECS.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import typing
from typing import Dict, List, Optional, Tuple, Union

from sky import sky_logging
from sky.adaptors import common as adaptors_common
from sky.catalog import common
from sky.clouds import cloud
from sky.utils import ux_utils

from skypilot_lyceum import api

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = adaptors_common.LazyImport('pandas')

logger = sky_logging.init_logger(__name__)

#: Lyceum has no regions or zones. One synthetic region keeps the catalog
#: helpers in `sky.catalog.common` happy.
DEFAULT_REGION = 'lyceum'

#: Measured per-GPU-unit specs. l40s/h100/a100/h200 were provisioned and
#: inspected directly; b200/b300 never had capacity and are EXTRAPOLATED --
#: they must stay flagged until a real job lands on one.
#: profile -> (vcpus_per_gpu, mem_gib_per_gpu, gpu_mem_mib, measured?)
INSTANCE_SPECS: Dict[str, Tuple[int, int, int, bool]] = {
    'l40s': (12, 70, 46068, True),
    'h100': (32, 181, 81559, True),
    'a100': (16, 94, 81920, True),
    'h200': (16, 196, 143771, True),
    'b200': (32, 181, 183359, False),
    'b300': (32, 181, 288358, False),
}

#: hardware_profile -> SkyPilot accelerator name.
ACCELERATOR_NAMES: Dict[str, str] = {
    'l40s': 'L40S', 'h100': 'H100', 'a100': 'A100',
    'h200': 'H200', 'b200': 'B200', 'b300': 'B300',
}

#: Price cache TTL: 48 static rows that change only on vendor pricing events.
PRICE_TTL_S = 3600
#: Availability cache TTL: volatile and races hard (C11); staleness costs a
#: doomed launch, so keep it short.
AVAILABILITY_TTL_S = 120

#: How long a *failed* fetch is remembered. Deliberately short and deliberately
#: NOT `PRICE_TTL_S`: a one-second blip must not pin the catalog to the baked
#: CSV for an hour, but the optimizer calls `_get_df()` once per candidate and
#: must not hammer a dead endpoint either.
FALLBACK_TTL_S = AVAILABILITY_TTL_S

#: The two pricing variants Lyceum exposes; they are the first element of both
#: the `/pricing` key and the `/vms/availability` key.
ON_DEMAND = 'on-demand'
SPOT = 'spot'

#: The offline catalog, generated from `/pricing` alone. Shipped in the wheel
#: (see `[tool.setuptools.package-data]`).
#:
#: NOTE: this file, and *only* this file, is the fallback. We must never reach
#: for `sky.catalog.common.read_catalog`: it downloads from
#: HOSTED_CATALOG_DIR_URL, Lyceum is not published there, and its failure mode
#: is a warning plus an empty DataFrame -- which silently removes Lyceum from
#: the optimizer instead of failing.
DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                        'vms.csv')

#: `{profile}.{count}x`, with no leading zeros so that parse(name(...)) is a
#: true inverse rather than merely a lenient one.
_INSTANCE_TYPE_RE = re.compile(r'^([a-z0-9]+)\.([1-9][0-9]*|0)x$')

#: Columns in the order SkyPilot's own CSVs use, plus `SpecsMeasured` (see
#: INSTANCE_SPECS): the measured/extrapolated distinction has to reach a
#: surface, not merely live in a dict.
_COLUMNS = ('InstanceType', 'AcceleratorName', 'AcceleratorCount', 'vCPUs',
            'MemoryGiB', 'Price', 'SpotPrice', 'Region', 'GpuInfo',
            'SpecsMeasured')

# (fetched_at, {(variant, profile, count): usd_per_hour}, came_from_api)
_PRICE_CACHE: Optional[Tuple[float, Dict[Tuple[str, str, int], float],
                             bool]] = None
# (fetched_at, {(variant, profile): [gpu_counts]} or None when the signal is
# unavailable -- None means "do not filter", NOT "nothing is available")
_AVAILABILITY_CACHE: Optional[Tuple[float,
                                    Optional[Dict[Tuple[str, str],
                                                  List[int]]]]] = None
# (price_fetched_at, availability_fetched_at, df)
_DF_CACHE: Optional[Tuple[float, float, 'pd.DataFrame']] = None


def _get_client():
    """The single seam through which this module reaches the Lyceum API.

    Declared explicitly so tests have one thing to patch, and so the caching in
    `_get_df` has one place to invalidate.
    """
    return api.LyceumClient()


def _reset_caches() -> None:
    """Drop the price and availability caches. For tests and for the reaper."""
    global _PRICE_CACHE, _AVAILABILITY_CACHE, _DF_CACHE
    _PRICE_CACHE = None
    _AVAILABILITY_CACHE = None
    _DF_CACHE = None


def instance_type_name(profile: str, gpu_count: int) -> str:
    """Canonical SkyPilot instance type, e.g. ('h100', 8) -> 'h100.8x'."""
    return f'{profile}.{gpu_count}x'


def parse_instance_type(instance_type: str) -> Tuple[str, int]:
    """Inverse of `instance_type_name`. Raises ValueError if unparseable."""
    if not isinstance(instance_type, str):
        raise ValueError(f'Invalid Lyceum instance type: {instance_type!r}')
    match = _INSTANCE_TYPE_RE.match(instance_type)
    if match is None:
        raise ValueError(
            f'Invalid Lyceum instance type {instance_type!r}; expected '
            '"{hardware_profile}.{gpu_count}x", e.g. "h100.8x".')
    profile, count_str = match.group(1), match.group(2)
    if profile not in INSTANCE_SPECS:
        raise ValueError(
            f'Unknown Lyceum hardware profile {profile!r}. Valid options: '
            f'{", ".join(sorted(INSTANCE_SPECS))}.')
    gpu_count = int(count_str)
    if gpu_count not in api.ALLOWED_GPU_COUNTS:
        # C8: the API silently coerces `gpu_count: 0` to 1 and provisions a
        # real billing VM, and rejects other bad counts with a 400 that names
        # nothing. This is the last place to catch it before money is spent.
        raise ValueError(
            f'Invalid Lyceum gpu_count {gpu_count} in {instance_type!r}. '
            f'Valid options: {", ".join(str(c) for c in api.ALLOWED_GPU_COUNTS)}.')
    return profile, gpu_count


def _gpu_info(profile: str, gpu_count: int) -> str:
    """The `GpuInfo` cell, as a Python dict literal.

    `sky/catalog/common.py:756` runs `df['GpuInfo'].apply(ast.literal_eval)`
    over the whole column and falls back to `DeviceMemoryGiB = None` for the
    ENTIRE cloud on ValueError/SyntaxError, so one malformed cell silently
    blanks device memory everywhere. The in-tree fetchers
    (`fetch_runpod.py:469`, `fetch_vast.py:116`) build it as
    `json.dumps({...}).replace('"', "'")`; do the same.
    """
    gpu_mem_mib = INSTANCE_SPECS[profile][2]
    return json.dumps({
        'Gpus': [{
            'Name': ACCELERATOR_NAMES[profile],
            'Count': gpu_count,
            'MemoryInfo': {
                'SizeInMiB': gpu_mem_mib
            },
        }],
        'TotalGpuMemoryInMiB': gpu_mem_mib * gpu_count,
    }).replace('"', "'")


def _load_csv_prices() -> Dict[Tuple[str, str, int], float]:
    """Price map parsed out of the baked CSV.

    Raises rather than returning an empty map: an unreadable fallback must be
    loud, because the alternative -- an empty DataFrame -- makes Lyceum
    invisible to the optimizer with no error anywhere.
    """
    prices: Dict[Tuple[str, str, int], float] = {}
    with open(DATA_CSV, 'r', newline='', encoding='utf-8') as csv_file:
        for row in csv.DictReader(csv_file):
            instance_type = (row.get('InstanceType') or '').strip()
            if not instance_type:
                continue
            profile, gpu_count = parse_instance_type(instance_type)
            for variant, column in ((ON_DEMAND, 'Price'), (SPOT, 'SpotPrice')):
                raw = (row.get(column) or '').strip()
                if raw:
                    prices[(variant, profile, gpu_count)] = float(raw)
    if not prices:
        raise RuntimeError(
            f'The baked Lyceum catalog {DATA_CSV} yielded no priced rows; '
            'refusing to serve an empty catalog.')
    return prices


def _get_prices() -> Tuple[float, Dict[Tuple[str, str, int], float], bool]:
    """(fetched_at, price map, came_from_api), cached for `PRICE_TTL_S`."""
    global _PRICE_CACHE
    now = time.monotonic()
    if _PRICE_CACHE is not None:
        stamp, _, from_api = _PRICE_CACHE
        if now - stamp < (PRICE_TTL_S if from_api else FALLBACK_TTL_S):
            return _PRICE_CACHE

    prices: Optional[Dict[Tuple[str, str, int], float]] = None
    try:
        prices = dict(_get_client().get_vm_prices())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f'Lyceum /pricing is unavailable ({exc!r}); falling '
                       f'back to the packaged catalog {DATA_CSV}.')
    if not prices:
        # Includes the "API answered with nothing" case: an empty price map is
        # indistinguishable from an outage and must not empty the catalog.
        _PRICE_CACHE = (now, _load_csv_prices(), False)
    else:
        _PRICE_CACHE = (now, {key: float(value)
                              for key, value in prices.items()}, True)
    return _PRICE_CACHE


def _get_availability(
) -> Tuple[float, Optional[Dict[Tuple[str, str], List[int]]]]:
    """(fetched_at, availability map or None), cached for `AVAILABILITY_TTL_S`.

    `None` means "we have no capacity signal", which is emphatically not the
    same as "nothing is available": see `_build_df`.
    """
    global _AVAILABILITY_CACHE
    now = time.monotonic()
    if (_AVAILABILITY_CACHE is not None and
            now - _AVAILABILITY_CACHE[0] < AVAILABILITY_TTL_S):
        return _AVAILABILITY_CACHE

    availability: Optional[Dict[Tuple[str, str], List[int]]]
    try:
        availability = {
            key: [int(count) for count in counts]
            for key, counts in _get_client().get_availability().items()
        }
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            f'Lyceum /vms/availability is unavailable ({exc!r}); serving '
            'every priced row unfiltered. The filter is an optimizer-quality '
            'measure, not a cost guard: a doomed combination is refused at '
            'create in seconds, for free (C7).')
        availability = None
    _AVAILABILITY_CACHE = (now, availability)
    return _AVAILABILITY_CACHE


def _rows(prices: Dict[Tuple[str, str, int], float],
          availability: Optional[Dict[Tuple[str, str], List[int]]]) -> List[Dict]:
    """One row per (profile, gpu_count) that is priced and (if known) offered."""
    rows: List[Dict] = []
    for profile in sorted(INSTANCE_SPECS):
        vcpus_per_gpu, mem_per_gpu, _, measured = INSTANCE_SPECS[profile]
        for gpu_count in api.ALLOWED_GPU_COUNTS:
            on_demand_price = prices.get((ON_DEMAND, profile, gpu_count))
            spot_price = prices.get((SPOT, profile, gpu_count))
            if on_demand_price is None and spot_price is None:
                continue
            if availability is not None:
                # C9: spot and on-demand are SEPARATE capacity axes keyed by
                # (instance_type, profile). Keying on the profile alone offers
                # spot rows that can never provision (l40s has no spot variant
                # at all) and hides combinations that can.
                on_demand_ok = gpu_count in availability.get(
                    (ON_DEMAND, profile), ())
                spot_ok = gpu_count in availability.get((SPOT, profile), ())
                if not on_demand_ok and not spot_ok:
                    continue
                if not spot_ok:
                    spot_price = None
            rows.append({
                'InstanceType': instance_type_name(profile, gpu_count),
                'AcceleratorName': ACCELERATOR_NAMES[profile],
                'AcceleratorCount': int(gpu_count),
                'vCPUs': float(vcpus_per_gpu * gpu_count),
                'MemoryGiB': float(mem_per_gpu * gpu_count),
                'Price': (float('nan') if on_demand_price is None else
                          float(on_demand_price)),
                'SpotPrice': (float('nan') if spot_price is None else
                              float(spot_price)),
                'Region': DEFAULT_REGION,
                'GpuInfo': _gpu_info(profile, gpu_count),
                'SpecsMeasured': bool(measured),
            })
    return rows


def _build_df(prices: Dict[Tuple[str, str, int], float],
              availability: Optional[Dict[Tuple[str, str], List[int]]]
              ) -> 'pd.DataFrame':
    rows = _rows(prices, availability)
    if not rows and availability is not None:
        # Every combination lost capacity at once. Serving nothing would remove
        # Lyceum from the optimizer entirely; serving the priced rows costs at
        # most a fast, free C7 refusal that SkyPilot fails over from.
        logger.warning('Lyceum reports no capacity for any priced row; '
                       'serving the priced rows unfiltered.')
        rows = _rows(prices, None)
    if not rows:
        raise RuntimeError(
            'The Lyceum catalog resolved to zero rows, which would make the '
            'cloud invisible to the optimizer.')
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def _get_df() -> 'pd.DataFrame':
    """Build (or return cached) the catalog DataFrame.

    Columns follow SkyPilot's CSV schema: InstanceType, AcceleratorName,
    AcceleratorCount, vCPUs, MemoryGiB, Price, SpotPrice, Region, GpuInfo.

    MUST NOT depend on SkyPilot's hosted catalog server, and MUST fall back to
    the baked CSV if the Lyceum API is unreachable -- a silent empty frame
    makes Lyceum invisible to the optimizer.

    All numeric cells must be plain Python floats/ints, never numpy scalars:
    numpy.float64 leaks break orjson in the API server.
    """
    global _DF_CACHE
    price_stamp, prices, _ = _get_prices()
    availability_stamp, availability = _get_availability()
    if (_DF_CACHE is not None and _DF_CACHE[0] == price_stamp and
            _DF_CACHE[1] == availability_stamp):
        return _DF_CACHE[2]
    df = _build_df(prices, availability)
    _DF_CACHE = (price_stamp, availability_stamp, df)
    return df


def _python_scalar(value):
    """numpy.float64 -> float, numpy.int64 -> int, numpy.bool_ -> bool.

    `isinstance(np.float64(1), float)` is True, so callers cannot detect the
    leak; orjson can, and 500s on it (upstream issue #7969).
    """
    item = getattr(value, 'item', None)
    if item is not None and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_get_df(), instance_type)


def validate_region_zone(region: Optional[str], zone: Optional[str]
                         ) -> Tuple[Optional[str], Optional[str]]:
    """Lyceum has no zones; a non-None zone must raise."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Lyceum does not support zones.')
    return common.validate_region_zone_impl('lyceum', _get_df(), region, zone)


def get_hourly_cost(instance_type: str, use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Lyceum does not support zones.')
    return common.get_hourly_cost_impl(_get_df(), instance_type, use_spot,
                                       region, zone)


def get_vcpus_mem_from_instance_type(instance_type: str
                                     ) -> Tuple[Optional[float], Optional[float]]:
    return common.get_vcpus_mem_from_instance_type_impl(_get_df(),
                                                        instance_type)


def get_default_instance_type(cpus: Optional[str] = None,
                              memory: Optional[str] = None,
                              disk_tier: Optional[str] = None,
                              local_disk: Optional[str] = None,
                              region: Optional[str] = None,
                              zone: Optional[str] = None,
                              use_spot: bool = False,
                              max_hourly_cost: Optional[float] = None
                              ) -> Optional[str]:
    del disk_tier, local_disk  # Lyceum exposes neither.
    return common.get_instance_type_for_cpus_mem_impl(_get_df(), cpus, memory,
                                                      region, zone, use_spot,
                                                      max_hourly_cost)


def get_accelerators_from_instance_type(instance_type: str
                                        ) -> Optional[Dict[str, Union[int, float]]]:
    return common.get_accelerators_from_instance_type_impl(_get_df(),
                                                           instance_type)


def get_instance_type_for_accelerator(
    acc_name: str,
    acc_count: Union[int, float],
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    use_spot: bool = False,
    local_disk: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    max_hourly_cost: Optional[float] = None,
) -> Tuple[Optional[List[str]], List[str]]:
    # NINE positional args, in this order. `sky/catalog/__init__.py:283`
    # forwards them positionally; dropping `local_disk`/`max_hourly_cost` (as an
    # earlier draft of this stub did) is a TypeError in the middle of optimize.
    del local_disk  # Lyceum exposes no local-disk choice.
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Lyceum does not support zones.')
    return common.get_instance_type_for_accelerator_impl(
        df=_get_df(),
        acc_name=acc_name,
        acc_count=acc_count,
        cpus=cpus,
        memory=memory,
        use_spot=use_spot,
        region=region,
        zone=zone,
        max_hourly_cost=max_hourly_cost)


def get_region_zones_for_instance_type(instance_type: str, use_spot: bool
                                       ) -> List['cloud.Region']:
    df = _get_df()
    return common.get_region_zones(df[df['InstanceType'] == instance_type],
                                   use_spot)


def regions() -> List['cloud.Region']:
    return [cloud.Region(DEFAULT_REGION)]


def list_accelerators(gpus_only: bool = True, name_filter: Optional[str] = None,
                      region_filter: Optional[str] = None,
                      quantity_filter: Optional[int] = None,
                      case_sensitive: bool = True, all_regions: bool = False,
                      require_price: bool = True) -> Dict[str, List]:
    del require_price  # Unused, matching every in-tree catalog.
    listed = common.list_accelerators_impl('Lyceum', _get_df(), gpus_only,
                                           name_filter, region_filter,
                                           quantity_filter, case_sensitive,
                                           all_regions)
    # `list_accelerators_impl` builds InstanceTypeInfo straight out of
    # DataFrame rows, so every numeric field is a numpy scalar unless we coerce
    # it here -- and a numpy scalar anywhere in an API-server response is
    # upstream issue #7969.
    return {
        name: [
            common.InstanceTypeInfo(*[_python_scalar(field) for field in info])
            for info in infos
        ] for name, infos in listed.items()
    }
