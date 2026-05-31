# GTA — Global Trade Alert

Ingestion pipeline for policy-intervention data from the
[Global Trade Alert](https://www.globaltradealert.org) (GTA) database.

---

## What GTA Measures

GTA tracks government interventions that affect international trade and
investment.  Each record is a discrete policy action (e.g. a tariff change,
import quota, export subsidy, or trade-facilitation measure) implemented by
a national government or supranational body.

GTA evaluates each intervention by its likely effect on trading partners:

| Colour | Code | Meaning |
|--------|------|---------|
| Red    | harmful      | Discriminates against foreign interests (e.g. import tariff, local-content requirement) |
| Green  | liberalising | Improves market access for foreign interests (e.g. tariff cut, trade-facilitation reform) |
| Amber  | unclear      | Uncertain or mixed directional effect |

---

## Implementation Date vs Announcement Date

GTA records up to four dates per intervention:

| Field              | Meaning |
|--------------------|---------|
| `date_announced`   | When the policy was publicly announced |
| `date_implemented` | When it entered legal force |
| `date_removed`     | When it was revoked (if applicable) |
| `date_published`   | When GTA's team published the record |

**This pipeline filters and aggregates on `date_implemented`**, which is the
economically relevant date — the point at which the policy begins to affect
trade flows and business decisions.  Using announcement dates would lead to
small look-ahead bias relative to realised disruption.

Note: a substantial minority of interventions have a missing
`date_implemented` (announced but not yet in force, or retroactively
applied).  These are dropped in the cleaning step and logged.

---

## Folder Structure

```
GTA/
├── config/
│   └── gta_config.yaml          pipeline configuration
├── data/
│   ├── raw/
│   │   └── gta_interventions_<START>_<END>.json   cached API response
│   ├── interim/
│   │   ├── gta_interventions_clean.parquet        cleaned intervention panel
│   │   └── gta_interventions_clean.csv
│   └── processed/
│       ├── gta_country_day_<START>_<END>.parquet  country × day panel
│       └── gta_country_day_<START>_<END>.csv
└── src/
    ├── gta_client.py            GTA API HTTP client
    ├── gta_pipeline.py          main pipeline entrypoint
    └── utils.py                 logging, missingness report, ISO3→UN mapping
```

---

## How to Run

### 1. Set your API key

```bash
export GTA_API_KEY="your_key_here"
```

Or pass it programmatically via `GTAClientConfig(api_key="...")`.

### 2. Run the pipeline

```bash
# From repo root or GTA/ folder:
python "External databases/GTA/src/gta_pipeline.py"

# Override dates:
python src/gta_pipeline.py --start 2020-01-01 --end 2024-12-31

# Force re-download (ignore cache):
python src/gta_pipeline.py --no-cache
```

### 3. Configure features in `config/gta_config.yaml`

Key flags:

| Config key                      | Default | Effect |
|---------------------------------|---------|--------|
| `features.harmful_events`       | `true`  | Add `gta_harmful_events` column |
| `features.liberalising_events`  | `true`  | Add `gta_liberalising_events` column |
| `features.rolling_windows`      | `[30, 90]` | Add `gta_30d_count`, `gta_90d_count` |
| `cache.use_cache`               | `true`  | Skip API if raw JSON exists |

---

## Country-Day Panel Construction

### Steps

1. **Fetch** all interventions with `date_implemented` in [start, end] via
   paginated POST requests to the GTA API.  Results cached as a single JSON.

2. **Clean** the raw records:
   - Parse `date_implemented` as datetime; drop rows where it is missing.
   - Explode multi-country interventions so each implementing jurisdiction
     gets its own row.
   - Deduplicate on `(intervention_id, implementing_country)`.
   - Map `gta_evaluation` string → binary `harmful` / `liberalising` flags.

3. **Aggregate** to a country × day panel:
   - Full date spine for every country that appears in the cleaned data.
   - `gta_policy_events` = count of interventions implemented that day.
   - `gta_harmful_events` / `gta_liberalising_events` = sub-counts by type.
   - Zero-filled for days with no interventions.

4. **Rolling features** (optional):
   - `gta_Nd_count` = rolling N-day trailing sum of `gta_policy_events`,
     computed within each country.

### Output schema

| Column                  | Type    | Description |
|-------------------------|---------|-------------|
| `country_iso3`          | str     | ISO3 country code of implementing jurisdiction |
| `date`                  | date    | Calendar date |
| `gta_policy_events`     | int     | All interventions implemented on this date |
| `gta_harmful_events`    | int     | Red (harmful) interventions |
| `gta_liberalising_events` | int   | Green (liberalising) interventions |
| `gta_30d_count`         | int     | 30-day trailing policy-event count |
| `gta_90d_count`         | int     | 90-day trailing policy-event count |

---

## Limitations

- **Policy ≠ unrest.**  GTA records government trade-policy actions, not
  political instability or social unrest.  A high GTA event count reflects
  an active legislative period, not necessarily economic crisis.

- **Coverage is uneven.**  Wealthier countries with transparent legislative
  processes are better covered than low-income countries.

- **Date uncertainty.**  Implementation dates are sometimes imprecise
  (quarterly or annual resolution) and may be retroactively corrected by GTA.

- **Intervention scope.**  One `intervention_id` may cover many products or
  trading partners.  The panel counts interventions, not affected trade value.

- **ISO3 codes.**  The `iso` field returned by the API for implementing
  jurisdictions is taken at face value as ISO3.  Verify against your country
  universe if joining to other datasets.

---

## Integration with the Modelling Pipeline

The processed panel is designed to join directly with the country × day
modelling spine in `Likelihood_modelling_social/src/build_panel_country_day.py`
on `(country_iso3, date)`.  Missing join keys (countries not covered by GTA)
should be filled with 0.

Suggested feature treatment in the model:
- Use `gta_policy_events` or `gta_harmful_events` as a contemporaneous
  count feature.
- Consider lagging by 1–7 days to account for publication delay.
- Normalise with `expanding_zscore` per country for comparability.
