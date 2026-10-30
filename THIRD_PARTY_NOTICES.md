# Third-party and reused components

## DeepSeek Harness

- Upstream: https://github.com/deepseek-ai/deepseek-harness
- Pinned commit: `f3803c96974ad4a94195fb97e5c637e9c2a3afe3`
- Copyright: Copyright (c) 2026 DeepSeek
- Upstream license: MIT; an unchanged copy is in `third_party/deepseek-harness/LICENSE`.

The upstream checkout and its installed dependencies are restored by
`scripts/setup-harness.ps1`; they are not vendored into this repository's Git history.
The restored checkout retains its own license and third-party notices.

## Reused numerical components

`src/control_harness/reused/` contains selected numerical classes extracted from
the author's NMPC-light, mpc_agent_v5 and AIagent projects. `reuse-manifest.json`
records repository-relative source paths, class names, line numbers, exact source
SHA-256 values and adaptation notes. Paths are relative to the parent directory
that contains those original repositories.

The extracted classes are included here; the original repositories are not
required to run this integration. `scripts/extract_reused_algorithms.py` is an
optional maintenance utility and requires those source repositories only when
explicitly re-extracting the classes.

This notice attributes upstream components. It does not assign a new license to
the author's integration or original projects. No integration-wide open-source
license has been selected in this upload.
