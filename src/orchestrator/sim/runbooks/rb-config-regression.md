---
id: rb-config-regression
title: Bad configuration change - revert
category: config_error
tags: config_error config_rejected config configuration revert issuer jwt authentication 401 invalid validation
---
# Configuration regression

Use when failures start right after a configuration push, with no code deploy.

## Steps
1. Identify the config revision pushed just before the failures started.
2. Diff the revision against the previous one; look for environment mix-ups (staging value in prod).
3. Revert the offending key to the previous revision (requires approval).
4. Verify the failing validation errors disappear.
