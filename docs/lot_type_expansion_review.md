# Lot Type Expansion — Review & Recommendations

---

## Overall Assessment

Well-structured analysis with a sensible phased approach. The technical challenges section is thorough — session median contamination and metric leakage are the real landmines. The document correctly identifies the priority order and is ready to inform an implementation plan.

---

## Recommendation — Weanling Approach

**Start with Option B — combined model**

11k weanling sold lots is borderline for a standalone model, especially once split by region (AUS ~4.4k, NZL ~674). Option B gives more training signal and a faster path to production. Split to Option A later only if accuracy is poor.

---

## Critical Issues — Resolve Before Building

### Must Fix

**Session median per lot type**
Blended medians in mixed sales will corrupt `log_price_index` for both lot types. This must be fixed before anything else — it affects both training data quality and future sale scoring.

**Prior year median fallback**
The hardcoded `salesLotTypeName = 'Yearling'` filter in `fetch_prior_year_median()` will silently produce wrong baselines for weanling future sales. Easy fix but easy to miss.

### Important

**Elite scaling threshold**
The AUS $500k threshold will rarely trigger for weanlings given their lower price points. Needs a weanling-specific threshold or a proportional approach tied to typical lot type price ratios.

**tblHorseAnalytics write-back schema**
Worth verifying the schema now before building the pipeline. The same horse appearing as both a weanling and a yearling could cause silent overwrites if the table isn't designed for multiple predictions per horse across lot types.

---

## Lot Type Grouping — Train as Families, Not Individually

Some lot types share enough price drivers that they can be treated as the same type for training purposes. Grouping by family avoids thin data problems and reduces the number of models to maintain.

| Family | Lot Types | Rationale |
|--------|-----------|-----------|
| Weanling family | Weanling + Foal at Foot | Same animal at different maturity stages, same sire/dam/vendor features apply. Pool for training. |
| Broodmare family | Broodmare + Broodmare in Foal | Same core price drivers (breeding record, progeny, covering sire). Treat as one type with an in-foal flag feature. |

Before building a new model, first ask whether an existing model family can absorb the new type with a `lot_type_encoded` feature rather than a separate pipeline.

---

## Gaps Not Covered in the Document

**NZL weanling data is very thin**
Only 674 NZL weanling sold lots since 2020. The document includes this in the appendix but doesn't flag the implication — this may be too thin for reliable NZL weanling predictions even in a combined model. Worth deciding upfront whether NZL weanling scoring falls back to the AUS model or is out of scope for Phase 1.

**Broodmare warrants its own spec document**
The document notes different feature engineering is needed but doesn't go further. Given the fundamentally different price drivers (breeding record, progeny performance, covering sire), broodmare should not be bolted onto the existing pipeline — it needs its own model spec before any implementation begins.

---

## Suggested Next Steps

1. Fix session median per lot type — prerequisite for everything else.
2. Fix prior year median fallback for lot type filtering.
3. Implement Option B — add weanling lots to training with `lot_type_encoded` feature.
4. Evaluate weanling accuracy separately after retraining — switch to Option A if insufficient.
5. Decide NZL weanling scope before Phase 1 launch.
6. Write a separate broodmare model spec before starting Phase 2.
