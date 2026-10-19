# Qwen 3.8 27B memory/pressure runs and the copied-load diagnosis

Final commit (native loader for shards without aligned tensors, padding accepted). M2 Max, 32 GiB, MLX 0.32.4.

- `align_checkpoint.py`: re-saves Jundot/Qwen3.8-27B-oQ3e-fp16-mtp with every tensor at an 8-byte offset
  (padding between tensors, same bytes; verified tensor by tensor with SHA-256).
- `run4.py` → `dense-b.jsonl`, `dense-ram-b.jsonl`: cold load, 1k→128, 2×1k→512, 8k→128, 90 s idle, resume.
  RAM samples include `footprint -p` for the server process. Single runs.
- `run5.py` + `stress.py` → `pressure.jsonl`, `pressure-req.jsonl`: same aligned checkpoint, copied vs mapped.
  After a 1k→128 request, `stress.py` allocates and touches 10 GB in a separate process for ~20 s while
  `vm_stat` and `memorystatus_vm_pressure_level` are sampled. Single runs.
- `prof.py` + `taskmem.py` → `timeline-copied-{cold,warm}.json`: Ornith 35B-A3B oQ4e outside the server.
  `mx.load` + `mx.eval` (copy), then one GPU pass over every tensor. Samples every 0.5 s: this process's
  `task_info(TASK_VM_INFO)` (resident, compressed, footprint) and system-wide `vm_stat` counters
  (compressions, decompressions, swap-ins, page-ins). Decompression/swap-in counters are system-wide.
