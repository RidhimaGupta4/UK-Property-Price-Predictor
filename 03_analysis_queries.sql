-- ============================================================
-- UK Property Price Predictor — SQL Analysis Queries
-- ============================================================
-- Compatible with: SQLite, DuckDB, PostgreSQL
--
-- Load tables (DuckDB):
--   CREATE TABLE properties  AS SELECT * FROM read_csv_auto('data/processed/properties.csv');
--   CREATE TABLE predictions AS SELECT * FROM read_csv_auto('data/processed/test_predictions.csv');
--   CREATE TABLE features    AS SELECT * FROM read_csv_auto('data/processed/feature_importance.csv');
--   CREATE TABLE regional    AS SELECT * FROM read_csv_auto('data/processed/regional_accuracy.csv');
--   CREATE TABLE ons_hpi     AS SELECT * FROM read_csv_auto('data/processed/ons_hpi.csv');
-- ============================================================


-- ── 1. REGIONAL PRICE SUMMARY ──────────────────────────────────────────────
-- Median, mean, and distribution by region — core market overview

SELECT
    region,
    COUNT(*)                                      AS transactions,
    ROUND(MIN(price_gbp), 0)                      AS min_price,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
          (ORDER BY price_gbp), 0)                AS p25_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY price_gbp), 0)                AS median_price,
    ROUND(AVG(price_gbp), 0)                      AS mean_price,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
          (ORDER BY price_gbp), 0)                AS p75_price,
    ROUND(MAX(price_gbp), 0)                      AS max_price,
    ROUND(STDDEV(price_gbp), 0)                   AS price_stddev
FROM properties
GROUP BY region
ORDER BY median_price DESC;


-- ── 2. PRICE BY PROPERTY TYPE AND REGION ──────────────────────────────────
-- Cross-tab: which type commands the biggest premium where?

SELECT
    region,
    ROUND(AVG(CASE WHEN property_type = 'Detached'      THEN price_gbp END), 0) AS detached_avg,
    ROUND(AVG(CASE WHEN property_type = 'Semi-Detached' THEN price_gbp END), 0) AS semi_avg,
    ROUND(AVG(CASE WHEN property_type = 'Terraced'      THEN price_gbp END), 0) AS terraced_avg,
    ROUND(AVG(CASE WHEN property_type = 'Flat'          THEN price_gbp END), 0) AS flat_avg,
    ROUND(AVG(CASE WHEN property_type = 'Detached'      THEN price_gbp END) /
          NULLIF(AVG(CASE WHEN property_type = 'Flat'   THEN price_gbp END), 0), 2)
                                                          AS detached_to_flat_ratio
FROM properties
GROUP BY region
ORDER BY detached_avg DESC NULLS LAST;


-- ── 3. EPC RATING PRICE PREMIUM ────────────────────────────────────────────
-- How much is energy efficiency worth? (Green premium analysis)

WITH base AS (
    SELECT AVG(price_gbp) AS overall_avg FROM properties
)
SELECT
    epc_rating,
    COUNT(*)                                      AS properties,
    ROUND(AVG(price_gbp), 0)                      AS avg_price,
    ROUND(AVG(price_gbp) / (SELECT overall_avg FROM base) * 100 - 100, 1)
                                                  AS pct_premium_vs_avg,
    ROUND(AVG(floor_area_sqm), 0)                 AS avg_floor_area,
    ROUND(AVG(commute_mins), 0)                   AS avg_commute_mins
FROM properties
GROUP BY epc_rating
ORDER BY epc_rating;


-- ── 4. SCHOOL RATING PRICE EFFECT ──────────────────────────────────────────
-- Ofsted outstanding vs inadequate — the school premium quantified

SELECT
    nearest_school_ofsted,
    COUNT(*)                                       AS properties,
    ROUND(AVG(price_gbp), 0)                       AS avg_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY price_gbp), 0)                 AS median_price,
    ROUND(AVG(price_gbp) - LAG(AVG(price_gbp)) OVER
          (ORDER BY CASE nearest_school_ofsted
              WHEN 'Outstanding'          THEN 1
              WHEN 'Good'                 THEN 2
              WHEN 'Requires Improvement' THEN 3
              WHEN 'Inadequate'           THEN 4 END), 0)
                                                   AS price_diff_from_better_band
FROM properties
GROUP BY nearest_school_ofsted
ORDER BY avg_price DESC;


-- ── 5. COMMUTE TIME vs PRICE CORRELATION ───────────────────────────────────
-- Every 10 minutes of commute = how much price reduction?

SELECT
    CASE
        WHEN commute_mins <= 15  THEN '0–15 min'
        WHEN commute_mins <= 30  THEN '16–30 min'
        WHEN commute_mins <= 45  THEN '31–45 min'
        WHEN commute_mins <= 60  THEN '46–60 min'
        WHEN commute_mins <= 75  THEN '61–75 min'
        ELSE '75+ min'
    END AS commute_band,
    COUNT(*)                                       AS properties,
    ROUND(AVG(price_gbp), 0)                       AS avg_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY price_gbp), 0)                 AS median_price,
    ROUND(AVG(floor_area_sqm), 0)                  AS avg_floor_area
FROM properties
GROUP BY commute_band
ORDER BY MIN(commute_mins);


-- ── 6. YEAR-ON-YEAR PRICE GROWTH ───────────────────────────────────────────
-- National and regional price change — COVID boom and 2023 correction

SELECT
    sale_year,
    COUNT(*)                                        AS transactions,
    ROUND(AVG(price_gbp), 0)                        AS avg_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY price_gbp), 0)                  AS median_price,
    ROUND(AVG(price_gbp) / LAG(AVG(price_gbp))
          OVER (ORDER BY sale_year) * 100 - 100, 1) AS yoy_pct_change,
    ROUND(SUM(price_gbp) / 1e9, 2)                 AS total_market_value_bn
FROM properties
GROUP BY sale_year
ORDER BY sale_year;


-- ── 7. BEDROOM PREMIUM ANALYSIS ────────────────────────────────────────────
-- Marginal value of each additional bedroom

WITH bed_prices AS (
    SELECT
        property_type,
        bedrooms,
        ROUND(AVG(price_gbp), 0) AS avg_price,
        COUNT(*) AS n
    FROM properties
    GROUP BY property_type, bedrooms
    HAVING COUNT(*) >= 10
)
SELECT
    property_type,
    bedrooms,
    avg_price,
    n,
    ROUND(avg_price - LAG(avg_price) OVER
          (PARTITION BY property_type ORDER BY bedrooms), 0) AS marginal_bedroom_value
FROM bed_prices
ORDER BY property_type, bedrooms;


-- ── 8. MODEL ACCURACY BY REGION ────────────────────────────────────────────
-- Where does the model predict best and worst?

SELECT
    region,
    ROUND(median_actual, 0)                        AS median_actual_price,
    ROUND(median_predicted, 0)                     AS median_predicted_price,
    ROUND(mae, 0)                                  AS mean_abs_error,
    mae_pct                                        AS mae_as_pct_of_price,
    count                                          AS test_transactions,
    CASE
        WHEN mae_pct <= 8  THEN 'Excellent (<8%)'
        WHEN mae_pct <= 12 THEN 'Good (8–12%)'
        WHEN mae_pct <= 18 THEN 'Fair (12–18%)'
        ELSE 'Needs improvement (>18%)'
    END AS accuracy_band
FROM regional
ORDER BY mae_pct ASC;


-- ── 9. PREDICTION ERROR DISTRIBUTION ──────────────────────────────────────
-- Understanding where the model over/underestimates

SELECT
    CASE
        WHEN error_pct BETWEEN -5  AND 5  THEN 'Within ±5%'
        WHEN error_pct BETWEEN -10 AND 10 THEN 'Within ±10%'
        WHEN error_pct BETWEEN -20 AND 20 THEN 'Within ±20%'
        WHEN error_pct > 20               THEN 'Overestimate >20%'
        ELSE 'Underestimate >20%'
    END AS error_band,
    COUNT(*) AS predictions,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_total,
    ROUND(AVG(actual_price), 0) AS avg_actual_price
FROM predictions
GROUP BY error_band
ORDER BY MIN(ABS(error_pct));


-- ── 10. HIGH VALUE PROPERTY PREMIUM FACTORS ────────────────────────────────
-- What features do properties in the top 10% price bracket share?

WITH quartiles AS (
    SELECT PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price_gbp) AS p90
    FROM properties
)
SELECT
    'Top 10% Properties'           AS segment,
    COUNT(*)                        AS count,
    ROUND(AVG(price_gbp), 0)        AS avg_price,
    ROUND(AVG(bedrooms), 1)         AS avg_bedrooms,
    ROUND(AVG(floor_area_sqm), 0)   AS avg_floor_area,
    ROUND(AVG(commute_mins), 0)     AS avg_commute_mins,
    ROUND(AVG(CASE WHEN epc_rating IN ('A','B','C') THEN 1.0 ELSE 0.0 END)*100,1) AS pct_good_epc,
    ROUND(AVG(CASE WHEN nearest_school_ofsted IN ('Outstanding','Good') THEN 1.0 ELSE 0.0 END)*100,1) AS pct_good_school,
    ROUND(AVG(has_garden), 2)       AS pct_with_garden,
    ROUND(AVG(has_garage), 2)       AS pct_with_garage
FROM properties, quartiles
WHERE price_gbp >= p90
UNION ALL
SELECT
    'Bottom 90% Properties',
    COUNT(*),
    ROUND(AVG(price_gbp), 0),
    ROUND(AVG(bedrooms), 1),
    ROUND(AVG(floor_area_sqm), 0),
    ROUND(AVG(commute_mins), 0),
    ROUND(AVG(CASE WHEN epc_rating IN ('A','B','C') THEN 1.0 ELSE 0.0 END)*100,1),
    ROUND(AVG(CASE WHEN nearest_school_ofsted IN ('Outstanding','Good') THEN 1.0 ELSE 0.0 END)*100,1),
    ROUND(AVG(has_garden), 2),
    ROUND(AVG(has_garage), 2)
FROM properties, quartiles
WHERE price_gbp < p90;


-- ── 11. FLOOD RISK PRICE DISCOUNT ─────────────────────────────────────────
-- Quantifying the climate risk discount on property values

SELECT
    flood_risk,
    COUNT(*)                           AS properties,
    ROUND(AVG(price_gbp), 0)           AS avg_price,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
          (ORDER BY price_gbp), 0)     AS median_price,
    ROUND(AVG(floor_area_sqm), 0)      AS avg_floor_area,
    ROUND(AVG(commute_mins), 0)        AS avg_commute_mins
FROM properties
GROUP BY flood_risk
ORDER BY avg_price DESC;


-- ── 12. FEATURE IMPORTANCE RANKING ─────────────────────────────────────────
-- What the Gradient Boosting model says matters most

SELECT
    feature,
    importance_pct,
    ROUND(SUM(importance_pct) OVER
          (ORDER BY importance_pct DESC), 1) AS cumulative_importance_pct,
    CASE
        WHEN importance_pct >= 20  THEN 'Critical'
        WHEN importance_pct >= 10  THEN 'High'
        WHEN importance_pct >=  5  THEN 'Medium'
        WHEN importance_pct >=  1  THEN 'Low'
        ELSE 'Minimal'
    END AS importance_band
FROM features
ORDER BY importance_pct DESC
LIMIT 20;
