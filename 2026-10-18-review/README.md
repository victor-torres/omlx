# Mapped weights: reduced PR scope and local evidence

PR: https://github.com/jundot/omlx/pull/4541 (Draft).

Source head `9f3723391fde1aec197c0fd7398d6c93e71da09b`; upstream main
`165be1ff5933a020f2f5a7544e1e95a2797f777f`. Experimental, off by default.
Automatic guard tiers remain supported. A small correction counts current live
file-backed buffers omitted from process footprint, without changing reserves,
admission estimates or static/Metal caps. Custom is **not required**. Fresh public
Mach object counters replace expensive page walks. The weak exporter ledger
unions live byte ranges, deduplicates files and subtracts all possible omitted-byte
and reclaimable-cache overlap; failed, partial, COW or size-changed maps get zero.
No delayed-release credit is created by residency changes. No wired override,
background sampler, checkpoint rewrite or proxy cache is added. `mlx.core.load`
is untouched; temporary loader-module proxies are thread/directory scoped.

The raw Mach query matched mincore on cold/active/idle maps (~0.1ms vs450ms).
The full ledger path was ~2ms median,4.2ms max during a stable19-sample interval;
loading/contended intervals had samples up to~278ms. This is observed latency,
not a worst-case guarantee. The implementation uses Apple's VM object counters:
https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map.c .

## Data provenance

- `bench-results/ab.jsonl`, `memory.jsonl`, `process-memory.jsonl`: exploratory
  three alternating pairs, before the final loader proxy refinement. Filter
  `model == 'Ornith-35B' and t >= 1792341740` (24 request records). Earlier rows
  are pilots/Qwen smoke checks and must not enter the three-pair aggregate.
  This was an uncommitted working state after merge `4749ad2e`, with the earlier
  scoped core hook; it is **not a pinned-source final benchmark**.
- `bench-results/final/`: exact `0c4dd117` module proxy; stock, mapped, copied
  fusion-disabled control, aligned sidecar proxy, MTP depth1/depth2, KV8 and32k
  needle. One load/two warm requests per variant. The subsequent source head
  adds fresh physical accounting; Custom benchmarks retain the same fixed-cap
  numerical path. An earlier Qwen probe returned Paris in both modes.
- `bench-results/verified-pairs/`: final three pairs against a separate clean
  main checkout and the source head; see metadata for exact revisions. This is
  the pinned-source comparison to use for the final PR numbers.
- `automatic-preview-live.json`: automatic tier preview during an active mapped
  35B request. Low physical footprint produced a dynamic ceiling below live
  MLX usage. This explains why a physical accounting correction is necessary; it is not an OOM result.
- `proxy-manifest.json`: experiment-only two-shard header-padding copy,
  2,065,829,800 extra bytes, 2.965s including payload SHA-256 verification.
  Source files unchanged. This utility is **not** part of the oMLX patch and
  only handles this checkpoint's header alignment, not arbitrary tensor layout.

All GB in system counters are decimal; configured25GiB guard and26624MiB Metal
cap are binary. Existing desktop applications, swap/compression history and
background I/O were uncontrolled. No process-footprint number represents total
RAM usage of mapped Metal buffers. Active wired memory remains near28GB in
both modes. Fusion differs in the default stock comparison, so decode differences
cannot be credited to mmap alone. The copied fusion-disabled control overlaps
mapped throughput.

## Reproduction

Use a scratch copy of this directory. Place/symlink the existing checkpoint as
`models/Ornith-35B`; it is the unmodified Ornith1.5-35B-A3B-oQ4e-fp16 checkpoint
(22shards,22.499GB). Do not rewrite its source files. Package and OS versions are
in `bench-results/metadata.json`; rebuild native kernels from synchronized main.
Use one compatible MLX0.32.4 environment for both servers and separate base dirs.
The scripts contain only a **dummy loopback API key**, never a user's credential.

Set `server.gpu_keep_warm_interval` to0 in each isolated base directory's
`settings.json`. Both use Custom25GiB and `--no-cache --no-hf-cache`. This cap is
an experiment setting, **not** a universal recommendation. Then start two servers
with `start_server.py`, which isolates oMLX's peer-interpreter helper:

```sh
# PY, PR_CHECKOUT and MAIN_CHECKOUT are absolute local paths.
PYTHONPATH="$PR_CHECKOUT:$PR_CHECKOUT/apps/omlx-web" "$PY" start_server.py serve \
  --host 127.0.0.1 --port 8011 --model-dir "$PWD/models" \
  --base-path "$PWD/bench-base" --memory-guard-gb 25 \
  --api-key bench-local-isolated --no-cache --no-hf-cache

PYTHONPATH="$MAIN_CHECKOUT:$MAIN_CHECKOUT/apps/omlx-web" "$PY" start_server.py serve \
  --host 127.0.0.1 --port 8012 --model-dir "$PWD/models" \
  --base-path "$PWD/main-bench-base" --memory-guard-gb 25 \
  --api-key bench-local-isolated --no-cache --no-hf-cache

# Run only after startup, with no other GPU benchmark/test competing.
"$PY" run_final_pairs.py
```

The harness alternates M,C,C,M,M,C. Each load is followed by1k+128tokens,
then warm1k+128 and warm8k+128; `--full-lifecycle` optionally adds30s idle and
resume1k+128, which was covered by the earlier exploratory series; the model is unloaded
between cases. Prompts are synthetic, tokenized with the checkpoint's tokenizer;
requests use temperature0 and thinkingFalse. Hardware/system samples are every2s.
Unload responses can precede teardown, so the harness waits for unloaded status.
`--no-cache` prevents prefix reuse. Both servers use the same acceleration flags.

For extras, create a second model alias `Ornith-stock-unfused` pointing to the
same checkpoint, and use `OMLX_BENCH_FUSION_CONTROL=1` on the PR server. This is
a test-only internal setting shim; the public settings API has no gate/up fusion
field. `build_proxy.py` requires the original alias and creates the optional
sidecar alias; run once only, in a disposable scratch folder. `run_extras.py`
then measures these controls and MTP/KV8/needle variants. Never treat an8-token
needle response as a sustained decode measurement.

## Validation

Final head:512 affected tests passed. Default selection with two workers:
17,242 passed,1,127 skipped,9 numeric native-kernel failures,3 subtests passed.
All9 failed nodes were reproduced on clean main with identical native binaries;
see `testing-summary.json`. New files pass Black/Ruff; `git diff --check` and
`make web` pass.12 direct JavaScript tests plus network-auth via its Python
wrapper pass. The Swift application source was unchanged.

Final three-pair medians (main / mapped): load48.59/13.86s, complete first
request58.48/20.66s, warm1k TTFT1.58/1.77s, warm1k decode42.45/65.64tok/s,
warm8k TTFT13.81/11.61s, warm8k decode43.25/60.66tok/s. Both modes had about28GB
system wired memory while active. Load ranges46.12–49.75/8.07–17.52s.
The warm short TTFT regressed slightly. Fusion differs; the copied unfused
control overlaps mapped decode, so this is not a pure mmap GPU-throughput win.
Raw medians/ranges/sample counts are in `bench-results/verified-pairs/summary.json`.

`region-qwen-auto.json`: direct model loads and Paris responses in Safe,
Balanced and Aggressive. `region-transition.json`:35B loaded under Custom25GiB,
then active generation62.01tok/s and a Paris idle-resume under Aggressive.
Cold35B admission under the desktop workload still refused estimated22GiB
against~21GiB available guard budget. This is expected policy, not an OOM or
proof of larger-than-RAM support.

Before/after screenshots are actual localhost UI captures: main before and the
new option after. Both snapshots show the actual saved model settings. No installed app replacement
or checkpoint rewrite was required. Source changes affect Python/web; Swift
application source was unchanged. A full app-bundle build was not used for these
CLI/server measurements.


Final additional probes: `bench-results/context64k/` contains one65,582-token
KV8 needle on final head, fixed25GiB. Exact LANTERN-4826; load15.73s,
prefill183.07s (358.23tok/s), complete request202.49s. The guard adapted
prefill chunks. Eight output tokens are not a sustained decode measurement;
there is no copied64k control.

`final-residency-profile.json` profiles final-head accounting separately:
1,883 exporters,100 calls, median0.718ms,p95 0.771ms,max2.463ms; after release0bytes.
This standalone profile has no concurrent inference. `residency-live.jsonl`
captures the earlier serving candidate, before the added file-size rejection:
steady19samples~1.96ms median,4.21ms max; loading/contended samples~278ms max.
Raw Mach comparisons against mincore are in `top-info-probe-results.txt`;
`MODEL_DIR=/absolute/checkpoint python top_info_probe.py` reproduces the oracle
comparison (prototype only; production contains no mincore).

GitHub CI for final head passed: https://github.com/jundot/omlx/actions/runs/39150514921/job/117743056866 .
PR stays Draft for author review.
