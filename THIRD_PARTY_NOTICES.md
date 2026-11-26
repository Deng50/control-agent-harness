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

## Control engineering library

The private control bundle incorporates the working-tree snapshot of
`control-skill-library` identified in
[`snapshot.json`](harness-plugin/control-skill-library/snapshot.json).
The source library attributes its numerical algorithms and replay wrappers
to the project and the spring configurations to the research group's archive.
Its [file provenance](harness-plugin/control-skill-library/sources.json)
and [MATLAB inventory](harness-plugin/control-skill-library/cases/matlab-scenes/inventory.json)
distinguish project-authored code, adapted cases and external inputs.

The original battery thermal identification author designates the code and
simulation models as learning examples and prohibits resale of unchanged copies.
The bundle retains the case description, derived metrics and fingerprints.
Its three adapted third-party programs/models are listed in
`snapshot.json.local_study_only`; consult the source library for local study.
Other external archives, calibrations and original materials remain subject
to their respective owners' terms. The source library supplies no blanket
open-source redistribution license. The historical scoring fixtures are
stored separately under `tests/fixtures/control-library-evaluation`.
