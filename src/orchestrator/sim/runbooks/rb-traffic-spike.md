---
id: rb-traffic-spike
title: Traffic surge - scale out capacity
category: traffic_spike
tags: traffic_spike traffic_surge queue_saturation scale autoscaler hpa maxreplicas rps cpu_saturation 429 rate_limit capacity
---
# Traffic surge

Use when request rate is a multiple of baseline, CPU is saturated and clients see 429s or queueing.

## Steps
1. Check whether the surge is organic (marketing, seasonal) or abusive.
2. Scale the service out; autoscaler is capped at its maximum, so raise replicas manually (requires approval).
3. Confirm the request queue drains and 429s stop.
4. Raise the autoscaler ceiling permanently and review load-shedding.
