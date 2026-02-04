# Market Value Model V2

Predict expected yearling sale prices with confidence ranges using machine learning.

**Supported Countries:** Australia (AUS), New Zealand (NZL), USA (pending model)

**Version:** v2.1.1 | [CHANGELOG](CHANGELOG.md) | [Deployment Guide](DEPLOYMENT.md)

---

## What is Market Value?

Market Value predicts the **expected hammer price** for a yearling at auction based on:

- **Sire performance** - Historical sale results (median price, clearance rate, momentum)
- **Dam production** - Previous progeny sale prices
- **Vendor track record** - Vendor's historical sale performance
- **Sale context** - Which sale, which book, session median

### Model Output

The model outputs three price points:

| Output | Meaning |
|--------|---------|
| **Expected (P50)** | The median predicted price - 50% chance of selling above/below |
| **Low (P25)** | Conservative estimate - 25% chance of selling below this |
| **High (P75)** | Optimistic estimate - 25% chance of selling above this |

### Confidence Tiers

| Tier | Meaning | Conditions |
|------|---------|------------|
| **High** | Strong data support | < $200k predicted, no data flags |
| **Medium** | Some uncertainty | < $200k with 1 flag, or $200-300k with no flags |
| **Low** | Limited data / elite tier | $300k+ predicted, or multiple data flags |

### Elite Scaling

Predictions >= $300k receive additional scaling to correct model compression at elite price tiers:

```
offset = base_offset + (ln(raw_p50) - ln(threshold)) * scaling_factor
```

---

## Model Details

### Algorithm

- **Model Type:** LightGBM Quantile Gradient Boosted Decision Trees
- **Target Variable:** `log(hammer_price / session_median_price)`
- **Quantiles:** P25, P50, P75 trained separately
- **Calibration:** Offsets applied to achieve exact 25%/75% coverage on holdout data

### Training Data

| Parameter | Value |
|-----------|-------|
| Source | Australian/NZ Yearling Sales 2020-2025 |
| Training Set | 18,839 lots (2020-2023) |
| Test Set | 9,929 lots (2024-2025) |
| Total Features | 28 |

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 0.70 | Mean absolute error in log-index space |
| R² | 0.16 | Variance explained |
| P25 Coverage | 25.0% | Perfectly calibrated |
| P75 Coverage | 75.0% | Perfectly calibrated |

> **Note:** R² of 0.16 is typical for yearling markets. Prices are heavily influenced by physical inspection, veterinary reports, and buyer competition - factors not captured in pedigree data.

---

## Feature Definitions

### Sire Features (rolling windows)

| Feature | Description |
|---------|-------------|
| `sire_sold_count_36m` | Yearlings sold in last 36 months |
| `sire_total_offered_36m` | Yearlings offered (sold + passed in) |
| `sire_clearance_rate_36m` | Sold / Offered x 100 |
| `sire_median_price_36m` | Median hammer price of sold yearlings |
| `sire_sold_count_12m` | Same metrics for 12-month window |
| `sire_median_price_12m` | |
| `sire_momentum` | 12m median - 36m median (price trend) |
| `sire_sample_flag_36m` | 1 if < 5 sold in 36m (thin sample) |

### Dam Features (lifetime prior)

| Feature | Description |
|---------|-------------|
| `dam_progeny_sold_count` | Total progeny sold at auction |
| `dam_progeny_total_offered_count` | Total progeny offered |
| `dam_progeny_median_price` | Median price of sold progeny |
| `dam_first_foal_flag` | 1 if no prior progeny sold |

### Vendor Features (rolling 36m)

| Feature | Description |
|---------|-------------|
| `vendor_sold_count_36m` | Yearlings sold by vendor |
| `vendor_total_offered_36m` | Yearlings offered by vendor |
| `vendor_clearance_rate_36m` | Vendor sell-through rate |
| `vendor_median_price_36m` | Vendor's median price |
| `vendor_volume_bucket` | New / Small / Medium / Large |
| `vendor_first_seen_flag` | 1 if vendor has no 36m history |

### Sale Context

| Feature | Description |
|---------|-------------|
| `session_median_price` | Median price for the sale/book |
| `sale_company` | Inglis / Magic Millions / NZB |
| `book_number` | Book 1, 2, 3 etc. |
| `sex` | Colt / Filly |

---

## Country Lookback Configuration

The model supports configurable historical data pooling per country:

| Sale Country | Lookback | Model | Effect |
|--------------|----------|-------|--------|
| NZL | AUS,NZL | nzl | Pool AUS+NZL history for better sire/dam/vendor metrics |
| AUS | AUS | aus | AUS-only history |
| USA | USA | usa | USA-only history (model pending) |

This allows NZL sales to benefit from larger sample sizes by including Australian sale history for shuttle stallions.

---

## Database Output: tblHorseAnalytics

| Column | Source | Notes |
|--------|--------|-------|
| `salesId` | Input sale ID | |
| `horseId` | Lot horse ID | |
| `marketValue` | mv_expected_price | P50 prediction |
| `marketValueLow` | mv_low_price | P25 prediction |
| `marketValueHigh` | mv_high_price | P75 prediction |
| `marketValueConfidence` | mv_confidence_tier | 'high'/'medium'/'low' |
| `marketValueMultiplier` | mv_expected_index | exp(adj_q50) |
| `sessionMedianPrice` | session_median_price | |
| `currencyId` | Auto from sale | AUS=1, NZL=6, USA=7 |

---

## File Structure

```
sm-market-value/
├── src/
│   ├── api.py                      # FastAPI application
│   └── score_sale.py               # Database-driven CLI scoring
├── models/
│   ├── aus/                        # Australia models
│   ├── nzl/                        # New Zealand models
│   └── usa/                        # USA models (pending)
├── Dockerfile                      # EKS container build
├── deployment.yaml                 # Kubernetes deployment config
├── service.yaml                    # Kubernetes service (LoadBalancer)
├── create-k8s-secret.sh            # Create K8s secrets (gitignored)
├── postman_collection.json         # API test collection
├── DEPLOYMENT.md                   # Setup & usage guide
├── CHANGELOG.md                    # Version history
└── README.md                       # This file (model documentation)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.1.1 | Jan 2025 | FastAPI endpoint, configurable country lookback, USA support |
| v2.1 | Jan 2025 | Database-driven scoring, elite scaling, auto country detection |
| v2.0 | Dec 2024 | Initial release. LightGBM quantile models with calibrated P25/P50/P75. |

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

---

## Getting Started

See [DEPLOYMENT.md](DEPLOYMENT.md) for installation, configuration, CLI usage, and API reference.
