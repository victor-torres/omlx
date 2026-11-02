# Lazy vision (experimental)

Lazy vision keeps a VLM's vision tower out of memory until a request carries an
image. Text-only use never loads it. When the tower has been idle for
`lazy_vision_idle_seconds` (default 60), it is released again.

The feature is off by default. Enable **Load vision on demand** in the settings
of a model served by the VLM engine, or set it through the normal model settings
API. Reload the model for a change to take effect:

```json
{"lazy_vision_enabled": true, "lazy_vision_idle_seconds": 60}
```

## How it works

1. The model loads as usual, and every post-load transform runs (float16
   recast, packed projections, ANE prefill). The tower is whatever these leave.
2. The tower's arrays are written to a snapshot file and replaced by
   one-element placeholders of the same dtype, so code that reads the tower's
   dtype or plain attributes keeps working. The MLX buffer cache is cleared so
   the memory goes back to the system.
3. The first call to the tower loads the snapshot back. Every caller goes
   through the tower's `__call__`: mlx-vlm's `get_input_embeddings`,
   `encode_image`, and the engine's own vision feature path. A vision feature
   cache hit never calls the tower, so it never loads it.
4. A background task releases the tower once it has been idle long enough. A
   release that lands while a graph still uses the old arrays is safe: the
   graph holds its own references until it is evaluated.

The snapshot holds the exact arrays the eager engine uses, so answers match the
eager engine. It lives in `<paged SSD cache dir>/lazy_vision/` (the system temp
directory when the SSD cache is off), one file per process and model, and is
deleted when the engine stops.

## Measured on Qwen3.6 35B-A3B oQ2.5e (M2 Max, 32 GB)

Process `phys_footprint`, one request at a time, 60 s idle timeout:

| | Eager | Lazy |
|---|---:|---:|
| Memory after load | 15.38 GiB | 14.55 GiB |
| Peak, text-only requests | 15.56 GiB | 14.76 GiB |
| Memory with the tower loaded | 15.43 GiB | 15.42 GiB |
| Memory after 105 s without images | 15.26 GiB | 14.42 GiB |
| Image after the idle period, time to first token | 0.75 s | 0.81 s |

Loading the tower took 0.23 s the first time and 0.07 s once the snapshot was in
the page cache. Text decode speed was the same or better, and the answers to
four test images matched the eager engine word for word, including the one
sent after a release.

## Limits

- Only models whose vision encoder is the `vision_tower` attribute are covered.
  Others log a line and stay eager. Projectors and `embed_vision` stay resident.
- The load-time peak does not change: the model loads whole, then the tower is
  moved out.
- The snapshot costs disk space the size of the tower (0.83 GiB for Qwen3.6).
- The memory estimate used for admission still counts the whole checkpoint.
- An image request after a release waits for the tower to load.
