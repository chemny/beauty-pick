# Changelog

## 0.2.0 — 2026-10-22

- Use the selected local standard-face method for new uploads and reanalysis: fixed landmark bounds, 12% margin, square 224-pixel input, no rotation, lossless scoring input.
- Preserve the existing ResNet50 weights and legacy display scale. Face scores are overall model predictions; unsupported anatomical facet scores are not fabricated.
- Record actual engines, dependency/model fingerprints and input hashes; isolate caches by version and retain previous reports after failed reanalysis.
- Preserve attempt history, cancellation checks and concurrency bounds. Existing assets and historical scores are not migrated or rescored by this release.
- Support overall-only face reports in ranking and omit empty facet grids in the UI.
- Keep research comparison tools separate from normal scoring. Trial images, databases, local environments and comparison results are not included.

### Limits

Facial scores express model preferences, not objective beauty or validated anatomical measurements. The legacy display mapping can saturate at 99. Body and photography assessments include heuristic measurements. Cross-platform execution and broad aesthetic accuracy are not newly certified by this release.
