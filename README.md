# Baseball Zerobase

Baseball Zerobase is a clean-room rewrite for a leakage-safe MLB starting-pitcher
strategy research pipeline. This repository must not use code, model weights, or
processed data from the prior project.

## Scope

- Product scope is MLB starting pitchers only. Bullpen recommendation is out of scope.
- Milestone 1-2 builds only leakage-safe data foundations and empirical baselines.
- There is no neural model, recommendation engine, API, or web UI yet.
- Locked tests must not be used during development.

## Current Interface

The initial command line interface exposes only a version command:

```bash
uv run baseball-zerobase version
```

Pipeline commands will be added by later milestone tasks.
