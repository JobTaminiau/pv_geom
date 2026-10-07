"""Rule-based mounting-type classification. ARCHIVED in 0.2.0.

A 300-polygon ground-truth sample (2026-07-30) put carport precision at 5% and
pole-mount at 0%. The code and its tests are kept so the work can be resumed
with better heuristics and tested examples; it runs only when
``mounting_rules.enabled`` is set, and then adds ``mounting_*`` columns.
"""
