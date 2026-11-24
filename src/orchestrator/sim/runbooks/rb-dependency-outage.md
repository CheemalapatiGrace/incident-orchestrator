---
id: rb-dependency-outage
title: Upstream dependency outage - fail over
category: dependency_outage
tags: dependency_outage upstream_failure dependency_errors failover circuit breaker timeout provider smtp secondary
---
# Dependency outage

Use when calls to a third-party or internal dependency time out while local resource metrics look healthy.

## Steps
1. Check the provider status page and confirm the outage is upstream.
2. Fail over to the configured secondary provider (requires approval).
3. Confirm the circuit breaker closes and upstream error rate drops.
4. Keep the incident open until the primary provider recovers.
