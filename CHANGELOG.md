# Changelog

## 0.3.0 — 2026-10-24

- Keep FPEM as the production face scorer; geometric-ratio experiments are not activated.
- Document the current local TOPIQ-IAA and parameter-assisted Qwen body engines.
- Add reusable measurement definitions, pose controls, reference-comparison limits and production-isolation rules to the Skill.
- Add a production asset manifest with exact sizes, SHA-256 and pinned download sources; provide separate locked main/parser environments and a read-only runtime doctor.
- Replace automatic multi-Agent synchronization with explicit single-target installation that refuses overwrites.
- Document Apple Silicon backend requirements and Windows access through an SSH tunnel to the user’s own Mac.
- Include current local body parameter integration, small-face and image-format handling, upload feedback, photo numbers and deletion controls.
- Preserve image-specific experiments outside the Skill delivery. No geometric-ratio experiment replaces FPEM; no production rescore is performed by installation.
- Keep third-party notices separate from the root MIT license, including TOPIQ noncommercial terms.

## 0.2.0 — 2026-10-22

- Use the selected local standard-face method for new uploads and reanalysis: fixed landmark bounds, 12% margin, square 224-pixel input, no rotation, lossless scoring input.
- Preserve the existing ResNet50 weights and legacy display scale. Face scores are overall model predictions; unsupported anatomical facet scores are not fabricated.
- Record actual engines, dependency/model fingerprints and input hashes; isolate caches by version and retain previous reports after failed reanalysis.
- Preserve attempt history, cancellation checks and concurrency bounds. Existing assets and historical scores are not migrated or rescored by this release.
- Support overall-only face reports in ranking and omit empty facet grids in the UI.
- Keep research comparison tools separate from normal scoring. Trial images, databases, local environments and comparison results are not included.

### Limits

Facial scores express model preferences, not objective beauty or validated anatomical measurements. The legacy display mapping can saturate at 99. Body and photography assessments include heuristic measurements. Cross-platform execution and broad aesthetic accuracy are not newly certified by this release.
