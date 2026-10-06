# Tests

Tests are versioned with the source code. They document expected behavior,
protect shared task interfaces during refactoring, and provide CI coverage.

- `rl/` contains policy, dataset, checkpoint, and environment tests.
- `rl/rfcl/` contains RFCL curriculum, distributed execution, task adapter,
  snapshot, rollout, and re-recording tests.

Generated reports, caches, datasets, videos, and experiment outputs remain
ignored; test source files should not be added to `.gitignore`.
