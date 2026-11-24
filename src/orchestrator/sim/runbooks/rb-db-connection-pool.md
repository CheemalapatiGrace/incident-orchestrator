---
id: rb-db-connection-pool
title: Database connection pool exhaustion
category: unknown
tags: database connection pool exhausted timeout slow query lock postgres
---
# Connection pool exhaustion

Distractor runbook: relevant to pool exhaustion only.

## Steps
1. Inspect active connections and long-running queries.
2. Kill idle-in-transaction sessions.
3. Raise pool size only after fixing the leak.
