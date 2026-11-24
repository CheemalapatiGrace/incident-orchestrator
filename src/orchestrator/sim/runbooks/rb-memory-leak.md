---
id: rb-memory-leak
title: Memory leak and OOM restart loop
category: memory_leak
tags: memory_leak oom oomkilled outofmemoryerror heap gc monotonic_growth memory_pct restart leak
---
# Memory leak

Use when memory grows steadily until containers are OOMKilled and restart in a loop.

## Steps
1. Confirm memory growth is monotonic and not traffic-correlated.
2. Capture a heap dump from one instance before restarting, if possible.
3. Perform a rolling restart to reclaim memory (requires approval).
4. Watch restarts and GC pauses; memory should return to baseline.
5. File a bug with the heap dump; schedule a fix, the restart is only a mitigation.
