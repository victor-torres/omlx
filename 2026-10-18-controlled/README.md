# Controlled fusion / mapping matrix

M2 Max, 32 GiB. PR head `9f372339` (feature code identical to the squashed commit), MLX 0.32.4.
One server (`start.py`, which toggles gate/up fusion through a benchmark-only settings hook),
driven by `run.py`: 2 checkpoints x {copied + fusion, copied without fusion, mapped} x 3 loads
in Latin-square order. Each load: cold 1,055→128, three warm 1,055→128, one warm 1,055→512,
one warm 8,223→128. Greedy sampling, prefix cache off, fixed 25 GiB guard.

`ab.jsonl` has one row per request: full response (usage, timings, text), and vm_stat/swap/pressure
before and after. 108 requests, no errors; texts identical across conditions per checkpoint.

Checkpoints: Ornith 1.5 35B-A3B oQ4e (90.8% of tensor bytes 8-byte aligned) and oQ3e (6.6%).
OS file cache warm, desktop apps and swap history uncontrolled.

## Cold vs warm file cache (`cache.jsonl`, `run3.py`, `evict.py`)

First complete request (load + 1,055→128) with one load per case, on the final commit (native loader
for shards without aligned tensors). Cold: the model's pages are evicted with `msync(MS_INVALIDATE)`
before the load and checked with `mincore`. Warm: the files are read just before the load.
The `cache_state` field records residency before/after the eviction or warm-up.
