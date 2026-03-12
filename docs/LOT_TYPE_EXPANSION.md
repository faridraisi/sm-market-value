# Lot Type Expansion — Considerations

Current model supports **Yearling** lots only. This document outlines considerations for adding other lot types.

## Data Volume (since 2020)

| Lot Type | Total Lots | Sold (with price) | Clearance | Priority |
|----------|-----------|-------------------|-----------|----------|
| Yearling | 154,814 | 82,272 | 53% | **Current** |
| Broodmare | 25,738 | 13,449 | 52% | Medium |
| Weanling | 31,069 | 10,894 | 35% | **High** |
| Race Horse | 22,489 | 6,124 | 27% | Low |
| Breeze Up | 12,762 | 2,439 | 19% | Low |
| Horse In Training | 8,525 | 501 | 6% | Not viable |
| Stallion/Prospect | 146 | 34 | 23% | Not viable |
| Stallion Share/Nom | 46 | 26 | 57% | Not viable |
| Foal at Foot | 38 | 20 | 53% | Not viable |
| Broodmare in Foal | 814 | 0 | 0% | Not viable |
| Breeding Right | 2 | 0 | 0% | Not viable |

### Yearling — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| USA | 56,054 | 33,924 | 70% |
| AUS | 39,859 | 28,125 | 84% |
| GBR | 19,440 | 4,008 | 21% |
| JPN | 12,197 | 3,752 | 32% |
| NZL | 9,915 | 6,489 | 76% |
| FRA | 9,472 | 3,176 | 34% |
| IRE | 3,877 | 2,798 | 77% |
| ZAF | 2,598 | 0 | — |
| GER | 953 | 0 | — |
| CAN | 449 | 0 | — |

### Weanling — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| USA | 13,005 | 5,530 | 53% |
| AUS | 7,155 | 4,362 | 76% |
| IRE | 4,780 | 0 | — |
| GBR | 3,132 | 88 | 3% |
| NZL | 994 | 674 | 77% |
| JPN | 803 | 180 | 23% |
| FRA | 747 | 60 | 8% |
| GER | 453 | 0 | — |

### Broodmare — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| USA | 15,797 | 9,994 | 77% |
| AUS | 4,557 | 3,058 | 77% |
| GBR | 3,278 | 1 | 0% |
| JPN | 1,139 | 358 | 32% |
| IRE | 876 | 0 | — |
| NZL | 70 | 36 | 55% |
| FRA | 20 | 2 | 10% |
| GER | 1 | 0 | — |

### Race Horse — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| USA | 8,571 | 3,298 | 53% |
| GBR | 7,374 | 1,091 | 16% |
| FRA | 4,097 | 226 | 6% |
| AUS | 2,010 | 1,453 | 84% |
| GER | 193 | 0 | — |
| IRE | 133 | 0 | — |
| NZL | 70 | 36 | 55% |
| HKG | 41 | 20 | 49% |

### Breeze Up — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| USA | 4,086 | 637 | 17% |
| GBR | 2,972 | 0 | — |
| AUS | 2,386 | 1,323 | 74% |
| FRA | 775 | 0 | — |
| ZAF | 759 | 0 | — |
| IRE | 705 | 0 | — |
| NZL | 640 | 404 | 81% |
| JPN | 439 | 75 | 17% |

### Horse In Training — by Country

| Country | Total | Sold | Clearance |
|---------|-------|------|-----------|
| GBR | 4,155 | 12 | 0% |
| USA | 3,520 | 362 | 11% |
| JPN | 396 | 0 | — |
| IRE | 286 | 0 | — |
| AUS | 165 | 127 | 91% |
| GER | 2 | 0 | — |
| FRA | 1 | 0 | — |

---

## Phase 1: Weanling (Recommended Next)

**Why:** Most similar to yearling — same sire/dam/vendor features apply, good data volume (~11k sold).

### Option A: Separate Weanling Model

Train a dedicated model using only weanling sale data.

**Pros:**
- Model fully tuned to weanling pricing patterns
- Independent calibration offsets and elite scaling thresholds
- No risk of yearling data biasing weanling predictions
- Can evolve independently (different features, thresholds)

**Cons:**
- Less training data (~11k sold vs ~82k yearlings) — may underfit
- Additional model to maintain per region (e.g. `aus_v7_weanling`)
- Duplicate pipeline code or config to manage two model types
- Sire/dam/vendor metrics computed separately — smaller sample sizes

**Changes required:**
- `config.json` — Add `weanling_model` field per region (e.g. `"weanling_model": "aus_weanling_v1"`)
- `src/run_rebuild.py` — Parameterize lot type filter, separate session median per lot type
- `src/train_model.py` — Accept `lot_type` parameter, filter training data accordingly
- `src/score_sale.py` — Select model based on lot type
- `api.py` — Training and scoring endpoints accept lot type parameter
- New model directories per region (e.g. `models/aus_weanling_v1/`)

---

### Option B: Combined Model with Lot Type Feature (Recommended)

Include weanling lots in the existing training pipeline alongside yearlings, with a `lot_type` encoded feature.

**Pros:**
- More training data overall (~93k combined) — model learns shared patterns
- Simpler to maintain — one model per region
- `lot_type` feature lets the model learn a baseline price offset between the two
- Shared sire/dam/vendor metrics have larger sample sizes
- If accuracy isn't sufficient, can split into separate models later (Option A)

**Cons:**
- Yearling data dominates (~88% of training set) — weanling signal may be diluted
- Single calibration offsets and elite scaling — may not be optimal for both
- Session median mixing could skew predictions if not handled per lot type

**Changes required:**
- `src/run_rebuild.py` — Expand filter to include weanlings, add `lot_type` column
- `src/train_model.py` — Include `lot_type_encoded` in feature columns
- `src/score_sale.py` — Add `lot_type_encoded` to scoring features
- Session median logic — Compute per lot type within a sale

---

### Recommended Approach

**Start with Option B**, then evaluate:

1. **Add weanling lots to training data** — Change `salesLotTypeName = 'Yearling'` filter to `IN ('Yearling', 'Weanling')`
2. **Add `lot_type` feature** — Encode as `lot_type_encoded` (0 = Yearling, 1 = Weanling) in training and scoring
3. **Session median per lot type** — Compute separate session medians for yearling and weanling lots within the same sale to avoid cross-contamination
4. **Train and evaluate** — Compare MAPE and quantile coverage for yearlings (should stay the same) and weanlings separately
5. **If weanling accuracy is poor** — Switch to Option A with a dedicated weanling model

**Considerations:**
- Weanlings typically sell for less than yearlings (less physical maturity, further from racing)
- Same sire/dam/vendor metrics are relevant for both
- A Mixed Sale (type 6) can contain both yearling and weanling lots — the `lot_type` feature handles this correctly
- Elite scaling thresholds may need separate values per lot type in future

---

## Technical Challenges for Adding Lot Types

These apply to weanlings (Phase 1) and become more significant for other lot types.

### 1. Session Median Contamination
`compute_sale_median()` currently calculates one median per sale. A Mixed Sale (type 6) with both yearlings and weanlings would blend their prices into one median, skewing the `log_price_index` target variable for both lot types. **Fix:** Compute session median per lot type within each sale.

### 2. Sire/Dam/Vendor Metrics Are Lot-Type Blind
`fetch_hist_lots()` in `run_rebuild.py` only pulls yearling lots for computing sire, dam, and vendor metrics. Questions to resolve:
- Should weanling sire stats be computed from weanling sales only, or combined yearling+weanling?
- A sire with 50 yearling sales but 3 weanling sales — which sample to use?
- Combined gives larger sample sizes but may dilute lot-type-specific pricing signals
- **Recommendation:** Start with combined metrics (simpler), evaluate if separate metrics improve accuracy

### 3. Prior Year Median Fallback
`fetch_prior_year_median()` in `run_rebuild.py` hardcodes `salesLotTypeName = 'Yearling'`. When scoring a weanling sale with no sold lots, this falls back to the yearling median from prior year — wrong baseline. **Fix:** Pass lot type through and filter accordingly.

### 4. Elite Scaling Thresholds
`elite_scaling.threshold` in `config.json` is tuned for yearling prices (e.g. AUS = $500,000). Weanlings sell for significantly less — this threshold would rarely trigger, making elite scaling ineffective for weanlings. **Fix:** Per-lot-type thresholds in config, or scale threshold proportionally based on typical lot type price ratio.

### 5. Calibration Offsets
`calibration_offsets.json` per model is calibrated on yearling residuals. Adding weanlings changes the error distribution, potentially degrading calibration for both. **Fix:** Recalibrate after retraining, or maintain per-lot-type offsets.

### 6. Confidence Tiers
`close_threshold` and `extreme_threshold` for confidence scoring are yearling-calibrated. Weanling price distributions may be tighter or wider. **Fix:** Evaluate confidence tier accuracy per lot type after retraining; adjust thresholds if needed.

### 7. Database Write-Back
`upsert_to_database()` writes predictions to `tblHorseAnalytics`. The same horse could appear as a weanling in one sale and a yearling in another. Need to verify the table schema handles multiple predictions per horse across different lot types without overwriting.

### 8. Frontend Impact
The UI currently assumes yearling-only data. Areas to update:
- Sale detail view — display lot type per lot
- Scoring display — may need lot type indicator
- Filtering — allow filtering by lot type
- Sale search — lot type counts in search results

---

## Phase 2: Broodmare (Future)

**Why:** Second largest sold dataset (13.4k), commercially important.

**Challenges — different price drivers:**
- Breeding record (foals produced, their race results)
- In-foal status and covering sire quality
- Age matters differently (peak breeding years vs too old)
- Pedigree page matters more (stakes-producing mares)

**Approach:** Likely needs a **separate model** with different feature engineering:
- Dam racing record (earnings, stakes wins)
- Progeny performance (winners, stakes winners)
- Covering sire (if in foal)
- Age and breeding history
- Sire/broodmare sire quality metrics

**Data availability:** Need to verify that progeny performance and covering sire data are accessible in the database.

---

## Phase 3: Breeze Up (Future)

**Why:** 2.4k sold lots, growing sale category.

**Challenges:**
- Workout/breeze times are a key price driver (not currently in the database?)
- Physical assessment plays a bigger role than yearlings
- Smaller dataset — may need to combine with yearling model

**Approach:** Could potentially extend the yearling model with:
- `lot_type` feature (as with weanling)
- Breeze time features if available
- Otherwise, sire/dam/vendor features still apply as a baseline

---

## Phase 4: Race Horse / Horse In Training (Future — Low Priority)

**Why:** 6.1k sold racehorses, but very different valuation model.

**Challenges:**
- Price driven by race record (earnings, wins, ratings)
- Current form and soundness
- Age and distance preferences
- Training status and trainer reputation
- Very low clearance for HIT (6%) — most data is unsold

**Approach:** Would require a completely separate model and feature pipeline with access to race performance data.

---

## Not Viable

The following lot types have insufficient sold data for meaningful model training:
- **Horse In Training** — Only 501 sold (6% clearance)
- **Stallion/Stallion Prospect** — 34 sold
- **Stallion Share/Nomination** — 26 sold
- **Foal at Foot** — 20 sold
- **Broodmare in Foal** — 0 sold (no price data)
- **Breeding Right** — 0 sold
