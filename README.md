# skypilot-lyceum

An **out-of-tree cloud provider plugin** that teaches
[SkyPilot](https://github.com/skypilot-org/skypilot) 0.13.0 how to run on
[Lyceum Cloud](https://lyceum.technology).

This is not a fork of SkyPilot and it does not patch a SkyPilot checkout. It is
an ordinary Python package: install it next to SkyPilot, enable it, and
`infra: lyceum` starts working in task YAML, in the optimizer, and in the
provisioner.

```yaml
# task.yaml
resources:
  infra: lyceum
  accelerators: H100:8
run: nvidia-smi
```

## What Lyceum is

Lyceum is a European GPU cloud that rents single VMs with directly attached
NVIDIA accelerators (L40S, A100, H100, H200, and Blackwell B200/B300), on-demand
or spot. Its public surface is a small REST API — create a VM with an inline SSH
public key, poll it for status, list VMs, delete a VM, plus pricing and
availability — and SSH into the box once it is up. There are no regions or
zones, no tags, no object store, no firewall API, and (importantly, see below)
no stop/start and no server-side idle timeout.

That maps cleanly onto SkyPilot's "VM with an SSH key" provisioner model, which
is what this package implements.

## What this package does

* Registers a `Lyceum` `sky.clouds.Cloud` subclass, so `infra: lyceum` parses
  and the optimizer can price it.
* Serves a **catalog** (instance types, vCPU/RAM, GPU info, on-demand and spot
  prices, live capacity) built from Lyceum's own `/pricing` and
  `/vms/availability` endpoints, with a packaged CSV as an offline fallback.
* Implements the nine `sky.provision` entry points: create/adopt, wait, query
  status, terminate, cluster info, and the port no-ops.
* Ships its own Ray cluster-config Jinja template and wires it in through
  SkyPilot's `template_override` hook.
* Provides an **orphan reaper** (see below), which on this cloud is the
  authoritative teardown mechanism rather than a safety net.

Instance types are named `{profile}.{count}x` — `h100.8x`, `l40s.1x`, etc. —
with GPU counts restricted to 1, 2, 4, or 8.

## Why an out-of-tree plugin at all

SkyPilot 0.13 has enough public extension surface that adding a cloud no longer
requires touching SkyPilot's source:

| Mechanism | What it gives you |
| --- | --- |
| `sky.utils.registry.CLOUD_REGISTRY.register` | Decorator on your `Cloud` subclass. Makes `CLOUD_REGISTRY.from_str('lyceum')` resolve, which is how a name in task YAML — and in SkyPilot's own cluster DB — becomes a `Cloud` object. |
| `sky.provision.register_provisioner(name, module, template_override=…)` | Points the provisioner dispatcher at your module. `sky.provision._route_to_cloud_impl` then routes `run_instances`, `wait_instances`, `terminate_instances`, … to it. |
| `sky.provision.TemplateSpec` / `template_override` | Lets your provisioner hand the backend an **absolute path** to a cluster-config template shipped inside your wheel. |
| `sky.server.plugins.BasePlugin` + `~/.sky/plugins.yaml` | Loads and installs your package in every relevant API-server process context (main, uvicorn, executor, controller) at start-up. |

So the whole thing is a wheel and four lines of YAML. `skypilot_lyceum.enable()`
performs the registration; `LyceumPlugin.install()` calls it server-side.

Client and server halves differ deliberately: `enable(client_only=True)` gives a
laptop just enough to *parse* `infra: lyceum`, while the catalog, optimizer,
provisioner and credentials all live server-side. No Lyceum API key ever needs to
reach a developer machine.

## Three structural gaps in SkyPilot's out-of-tree support

This is the part most likely to be useful to someone else writing an out-of-tree
provider. All three were found the hard way, against SkyPilot 0.13.0.

### (a) The cluster-config template registry is a hardcoded dict — *solved*

`cloud_vm_ray_backend._get_cluster_config_template` ends in
`return cloud_to_template[type(cloud)]`, a bare dict lookup keyed by in-tree
cloud *classes*. For an out-of-tree cloud that is a `KeyError` on **every**
launch.

The escape hatch is real and documented: return a
`sky.provision.TemplateSpec(template_path=<absolute path>)` from your
provisioner's `template_override`, which the backend consults before falling
back to the dict. `sky/provision/__init__.py` explicitly names the absolute-path
form as the "plugin-shipped template" case. This package builds that path from
`__file__` so it survives `pip install`, and
`tests/test_template_override.py` pins it.

**Verdict:** a supported hook exists. Use it, and remember that nothing else in
your test suite will notice if you forget to.

### (b) SSH-key substitution has no hook at all — *requires a monkeypatch*

`sky.backends.backend_utils._add_auth_to_cluster_config` is an `isinstance`
chain over in-tree cloud classes that ends in:

```python
assert False, cloud
```

There is no registry, no hook, no default branch. A real
`sky.launch(infra='lyceum')` dies with a bare `AssertionError: Lyceum` at that
line — *after* the optimizer has already selected the resource, so the failure
appears to come from nowhere.

What Lyceum needs is exactly the generic branch,
`sky.authentication.configure_ssh_info`, which substitutes
`skypilot:ssh_public_key_content` and `skypilot:ssh_user` into the rendered
config. (Some in-tree clouds need a bespoke variant only because they must
pre-register the key with the account and receive an id back. Lyceum takes the
public key inline on every create call and has no key registry — which is also
why the shipped template has no `ssh_key_id`.)

So `patches.patch_cluster_auth()` wraps that function, intercepts *only* the
Lyceum cloud class, and delegates everything else to the original.

**Verdict:** unfixable from outside without a monkeypatch. The clean upstream
fix would be a `Cloud`-level hook, or an `isinstance` fallthrough that calls
`configure_ssh_info` instead of asserting.

### (c) The skylet never loads plugins, so node-side autostop/autodown cannot work — *unfixable downstream*

This is the expensive one.

SkyPilot's autodown runs **on the node**. The skylet's autostop event calls
`sky.provision.terminate_instances('lyceum', …)` locally, which requires your
provisioner to be *registered in the skylet's process*. Registration only
happens through `plugins.load_plugins()` — and the skylet never calls it. Not
once, anywhere in the tree.

Installing this package on the node is therefore **not enough**. There is no
hook at all. Consequently, for any out-of-tree cloud:

* `sky autostop --down` and any TTL you set will fire on schedule,
* the cluster will move to `AUTOSTOPPING`,
* and it will stick there, because the node cannot dispatch the termination.

Observed consequence before this was understood: a VM billing for 30+ minutes
after its job had been killed, while the control plane still considered the
cluster "known" and therefore not an orphan.

On a cloud that *has* a server-side idle timeout this is merely untidy. Lyceum
has none (see C5 below): the only thing that ever stops a Lyceum VM billing is an
explicit `DELETE`. That makes an **external reaper the only reliable teardown
mechanism**, not a backstop — hence `skypilot_lyceum/reaper.py`, and hence the
reaper's special short grace window for clusters stuck in `AUTOSTOPPING`.

`tests/test_cloud_class.py::test_the_skylet_never_loads_plugins_so_nodes_cannot_self_terminate`
scans the installed SkyPilot tree and fails the moment upstream grows
skylet-side plugin loading, at which point the reaper can be demoted to a
backstop.

**Verdict:** cannot be fixed downstream. If you are writing an out-of-tree
provider for a cloud without a TTL, plan for an external reaper from day one.

## Lyceum API quirks (C1–C12)

Twelve behaviours of the Lyceum API contradict its own documentation or the
obvious reading of its responses. Each was found empirically — several by
provisioning real GPUs — and each has a fixture-backed regression test. The
`C<n>` labels are used throughout the source.

| # | Observed behaviour | What this package does |
| --- | --- | --- |
| **C1** | The SSH user is `lyceum`. The vendor docs say `root`; both `root` and `ubuntu` are refused by the real image with `Permission denied (publickey)`. | `ssh_user: lyceum` in the template, in `ClusterInfo`, and in the deploy variables. |
| **C2** | `ip_address` is polymorphic: a bare host (`203.0.113.10`) on some VMs and `host:port` (`198.51.100.20:22`) on others — from the same account, minutes apart. | `api.parse_ip_address` splits host from port once, defaults to 22, treats a bare IPv6 literal as a host, and **raises** on a malformed port rather than guessing. |
| **C3** | The catalog lives in `/pricing`: rows with `meter_slug == "vm_running"`, keyed by `applies_to.hardware_profile` in the form `{instance_type}.{profile}.{count}x`. `unit_price` is per **second**; `unit_price_per_hour` is a **string**. | Reads `applies_to` first (`group_by` only as fallback) and coerces to plain `float` — a string price compares lexicographically, and `Decimal`/`numpy.float64` break the API server's JSON encoder. |
| **C4** | `/vms/list` defaults every `include_*` flag to `true`, so the naive call returns terminated and failed VMs — and a terminated VM keeps its `display_name` forever. | Sends `include_terminated=false&include_failed=false` **and** filters client-side; identity lookups exclude terminal VMs and order by `created_at` descending. |
| **C5** | There is no stop endpoint and **no cloud-side TTL**. A VM bills from `ready` until someone issues `DELETE` — at `b300.8x` that is $63.92/h. | `STOP` declared unsupported; every failure path terminates what it created; the orphan reaper exists. |
| **C6** | In the `/vms/create` response, top-level `hardware_profile` and `gpu_count` are `null`; the real values are under `instance_specs`. | `api._parse_vm` reads `instance_specs` first and falls back to the top level. |
| **C7** | Capacity exhaustion is reported as **HTTP 500**, distinguishable only by the detail text "could not be provisioned". | Mapped to a distinct `LyceumCapacityError` and re-raised as `ResourcesUnavailableError`, so the optimizer fails over instead of backing off against an exhausted SKU. A generic retry-on-5xx client gets this exactly wrong. |
| **C8** | `gpu_count` must be nested under `instance_specs`; a top-level one is ignored. `gpu_count: 0` is silently coerced to **1** and provisions a real, billing VM. Other invalid counts get an unhelpful 400. | Counts are validated client-side against `(1, 2, 4, 8)` and **rejected, never clamped**; the payload always nests `gpu_count`. |
| **C9** | Spot and on-demand are separate capacity *and* pricing axes. `/vms/availability` exposes both `available_hardware_profiles` (which conflates them) and `available_instance_variants` (which does not). L40S has no spot variant at all. | Reads `available_instance_variants`, keyed by `(instance_type, profile)`. `instance_type` is always sent explicitly on create so the bill never depends on a server-side default. |
| **C10** | A VM can report `status: "ready"` while `ip_address` is still `null` (observed on an H200 at 104 s). | `VM.is_usable` requires ready **and** a non-null IP; `wait_instances` gates on that, while *adoption* deliberately gates only on "not terminal". |
| **C11** | `/vms/availability` is advisory and races hard — capacity vanished within minutes during the review. | Cached for 120 s only, and treated as an optimizer-quality signal, never a cost guard: losing it serves all priced rows rather than emptying the catalog. An "exists but empty" list and an absent key are kept distinct. |
| **C12** | `vm_id` is opaque: undashed hex for some VMs, a dashed UUID for others. | Never parsed, normalised, or validated anywhere. |

Measured provisioning times: 221 s on-demand, 130 s spot (vendor docs claim
1–3 min); the provisioner's timeout is 900 s. A capacity refusal (C7) comes back
in 2.7–9.1 s with nothing created and nothing charged.

## Install

```bash
pip install git+https://github.com/onticlabs/skypilot-lyceum
```

Requires Python ≥ 3.10 and pulls in `skypilot==0.13.0` — see
[Version pinning](#version-pinning) for why that is exact.

### Enable it

Server-side (the SkyPilot API server), via `~/.sky/plugins.yaml`:

```yaml
plugins:
  - class: skypilot_lyceum.plugin.LyceumPlugin
```

Client-side, or in a plain SDK script:

```python
import skypilot_lyceum
skypilot_lyceum.enable()                    # full: catalog + provisioner
skypilot_lyceum.enable(client_only=True)    # parse `infra: lyceum` only
```

`enable()` is idempotent within a process.

## Configuration

One credential: a Lyceum API key, resolved in this order.

1. `$LYCEUM_API_KEY`
2. `~/.lyceum/api_key`

```bash
export LYCEUM_API_KEY=lk_...
# or
mkdir -p ~/.lyceum && printf %s "$LYCEUM_API_KEY" > ~/.lyceum/api_key
```

A **set-but-empty** environment variable is deliberately not treated as a
credential (container platforms commonly export an unset secret as
`LYCEUM_API_KEY=`); nor is a whitespace-only key file. Both fall through to the
next source and then to a clear error, rather than sending `Bearer ` and
producing a 401 storm.

The key is mounted onto provisioned nodes at the same path
(`~/.lyceum/api_key`). Verify it with:

```bash
sky check lyceum
```

which calls `GET /user/status`. Credential-check output is scrubbed of anything
key-shaped before it is returned, because `sky check` output is routinely pasted
into chat and captured in server logs.

## Usage

```bash
sky launch -c mycluster task.yaml     # task.yaml sets `infra: lyceum`
sky status
sky down mycluster
```

Note that `sky stop` will fail by design: Lyceum has no stop endpoint, so a
stopped cluster could never be resumed. Use `sky down`. And see gap (c) above
before relying on `--down`/autostop.

## What Lyceum cannot do

Declared in `Lyceum._CLOUD_UNSUPPORTED_FEATURES`, each with a reason string:
`STOP`, `MULTI_NODE` (enterprise-contract only, not on the public API),
`SPOT_INSTANCE`, `CUSTOM_DISK_TIER`, `CUSTOM_NETWORK_TIER`,
`CUSTOM_MULTI_NETWORK`, `DOCKER_IMAGE`, `IMAGE_ID`, `STORAGE_MOUNTING`,
`HOST_CONTROLLERS`, `HIGH_AVAILABILITY_CONTROLLERS`, `CLONE_DISK_FROM_CLUSTER`,
`LOCAL_DISK`.

Spot *provisioning* works and is verified; `SPOT_INSTANCE` is nevertheless
declared unsupported because **preemption detection is not implemented** — a
reclaimed VM carries no spot-specific field and is observable only as a status
transition.

Jobs and serve controllers must not be hosted on Lyceum: it is single-node, and
there is no cloud-side TTL to stop a leaked controller billing.

## Cluster identity

Lyceum has no tags and no server-side filtering. The **only** handle a cluster
has is `display_name == cluster_name_on_cloud`, which is why:

* matching is exact equality, never a prefix (`sky-cluster-abc` must not resolve
  `sky-cluster-abcd`);
* terminal VMs are excluded, since they keep their name forever (C4);
* among survivors, the newest `created_at` wins, since names are reused across
  cluster generations;
* `max_cluster_name_length()` is overridden as the **public** classmethod. (The
  base class defines only the public name; a cloud that overrides the private
  `_max_cluster_name_length` silently gets no limit at all.)

## The catalog

Rows are `(hardware_profile × gpu_count × instance_type)`, priced from
`/pricing` and filtered by `/vms/availability`. Prices are cached for 1 h,
availability for 120 s.

If the Lyceum API is unreachable, the catalog falls back to the packaged
`skypilot_lyceum/data/vms.csv` — never to SkyPilot's hosted catalog server,
where Lyceum is not published and whose failure mode is a warning plus an empty
DataFrame, i.e. silently removing the cloud from the optimizer.

vCPU and RAM are exposed by no Lyceum endpoint, so they are carried from direct
measurement. **L40S, A100, H100 and H200 were provisioned and inspected; B200
and B300 were never available during the review and their vCPU/RAM figures are
extrapolated, not measured.** The `SpecsMeasured` column carries that
distinction, and a `strict=True` xfail in the test suite turns red the moment
someone flips the flag — so the reminder cannot rot.

## The orphan reaper

`skypilot_lyceum/reaper.py` finds Lyceum VMs that are billing but unaccounted
for and terminates them. Because of gap (c) and C5, on this cloud it is the
authoritative teardown, not a safety net: an API server that died mid-provision,
an executor that crashed between create and record, or a cluster DB lost with
its volume all leave a GPU running with nothing that will ever collect it.

It is destructive automation pointed at production, so it is built to refuse:

* **dry run by default** — terminating requires explicitly opting in;
* a VM belonging to a live cluster is never touched;
* a VM younger than the grace window (default 1 h) is never touched, since it
  may still be provisioning;
* a VM whose `display_name` is not SkyPilot-shaped (`<name>-<8 hex>`) is never
  touched — a human made it;
* an **unknown** cluster set raises `UnknownClusterStateError` rather than
  treating "I don't know" as "nothing is running". That single rule is what
  stands between a failed cluster-DB read and a deleted fleet mid-training;
* a failed *listing* aborts the run; a failed *termination* is recorded and the
  remaining VMs are still attempted.

Clusters known to be stuck in `AUTOSTOPPING` are re-opened to collection after a
much shorter 10-minute grace, because that state means the job ended, teardown
fired, and nothing else will ever finish it.

The module exposes `select_orphans`, `describe` and `reap`; it deliberately
ships no CLI or cron wiring, since scheduling and the source of the "known
clusters" set belong to whatever operates the control plane.

## Status and maturity

Early but exercised. The provider has been run end-to-end against the live
Lyceum API: real VMs provisioned on-demand and spot, adopted, polled, and torn
down. 393 tests cover it (392 passing, 1 deliberate `strict` xfail — the
B200/B300 measurement reminder above). `tests/conftest.py` blocks real sockets,
so no unit test can provision a GPU by accident, and API response fixtures under
`tests/fixtures/lyceum_api/` are captured verbatim from the live API (with
account identifiers and host addresses replaced by placeholders).

Known gaps: spot preemption detection, multi-node, and measured B200/B300 specs.

### Version pinning

`skypilot==0.13.0` is an **exact** pin and a hard runtime dependency, not an
extra:

* two of the three seams below are monkeypatches anchored to 0.13.0 internals;
* the provisioner's signatures are bound at dispatch by
  `inspect.signature(...).bind(...)`, so a renamed or dropped parameter upstream
  is a `TypeError` in the middle of a provisioning run rather than an import
  error.

Installing against a different SkyPilot should fail loudly at resolve time, not
at launch time with a GPU already on the meter. The pin is the outer of two
guards; the anchors below are the inner one.

### The anchored patches

`skypilot_lyceum/patches.py` holds every non-public thing this package does.
Each patch **checks its upstream anchor before mutating anything** and raises
`PatchDriftError` if the anchor has moved — never a warning, and never a
half-applied patch. A server that boots "healthy" and then rejects every Lyceum
job is far worse than one that refuses to boot with the reason on stderr.

| Patch | Why it is needed | Anchor | On drift |
| --- | --- | --- | --- |
| `patch_all_clouds` | `sky.skylet.constants.ALL_CLOUDS` is what the task-YAML JSON schema builds its `cloud`/`infra` validators from. Without the name in it, **both** spellings are rejected client-side before any of this code runs. | `ALL_CLOUDS` is still a `tuple` of `str` containing known in-tree clouds. | Raises; the constant is left exactly as found. |
| `patch_catalog_module` | `sky.catalog._map_clouds_catalog` resolves catalogs with a hardcoded `importlib.import_module(f'sky.catalog.{cloud}_catalog')`. There is no catalog registry. | An in-tree cloud's catalog module still imports under that exact naming convention. | Raises rather than injecting a module nothing will look up. |
| `patch_cluster_auth` | Gap (b) above: `_add_auth_to_cluster_config` ends in `assert False, cloud`. | The function still exists, still contains the `assert False` fallthrough, and `authentication.configure_ssh_info` still exists. | Raises. If the fallthrough is gone, upstream may handle out-of-tree clouds natively and **this patch should be deleted, not kept**. |

All three are idempotent, and the wrapper keeps the original callable reachable
so the anchor stays checkable after the patch is applied.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
uv run pytest -q
```

or with plain pip:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Expected: `392 passed, 1 xfailed`.

The test suite is organised by seam — `test_api_client.py` (HTTP and the C1–C12
quirks), `test_catalog.py`, `test_cloud_class.py`, `test_provisioner.py`,
`test_registration.py` (the patches), `test_signature_conformance.py` (static
checks against SkyPilot's dispatcher), `test_template_override.py`,
`test_plugin.py`, `test_reaper.py`. Docstrings state the concrete production
failure each test prevents; they are the real documentation for this package.

## License

**No license has been chosen yet.** Until one is added, default copyright
applies — all rights reserved — which means this code cannot legally be used,
modified, or redistributed by anyone else, and cannot be contributed upstream.
Apache-2.0 would match SkyPilot's own license and keep the upstreaming path
open.
