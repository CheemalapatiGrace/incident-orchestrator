---
id: rb-disk-full
title: Disk exhaustion on a data volume
category: disk_full
tags: disk_full disk_exhausted capacity_exhausted enospc volume storage expand space database write
---
# Disk full

Use when writes fail with "No space left on device" and disk usage is at 100%.

## Steps
1. Confirm volume usage and growth rate.
2. Expand the volume online (requires approval); growth is non-disruptive.
3. Verify writes succeed and replication catches up.
4. Add capacity alerting at 80% and review retention policies.
