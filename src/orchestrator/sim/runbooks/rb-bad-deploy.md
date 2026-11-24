---
id: rb-bad-deploy
title: Faulty deployment - confirm and roll back
category: bad_deploy
tags: bad_deploy deploy release rollback regression unhandled_exception nullpointerexception 5xx error_rate_step latency_regression
---
# Faulty deployment

Use when the error rate steps up within an hour of a release and capacity metrics are normal.

## Steps
1. Compare the onset of errors with the most recent deploy timestamps.
2. Confirm stack traces point at code touched by the release.
3. Roll back to the previous known-good version (requires approval).
4. Verify the error rate and p99 latency return to baseline within 10 minutes.
5. Open a follow-up ticket for the release owner with the failing trace.
