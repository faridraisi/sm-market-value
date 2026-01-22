-- date: 20260120

-- Full rebuild script for AUS Market Value Golden Table (mv_yearling_lot_features_v1)
-- This script truncates all staging tables and rebuilds from scratch

-- ============================================================================
-- STEP 1: TRUNCATE ALL TABLES
-- ============================================================================

TRUNCATE TABLE [G1StallionMatchProductionV5].[dbo].[stg_mv_sale_median_v1];
TRUNCATE TABLE [G1StallionMatchProductionV5].[dbo].[stg_mv_sire_metrics_v1];
TRUNCATE TABLE [G1StallionMatchProductionV5].[dbo].[stg_mv_dam_stats_v1];
TRUNCATE TABLE [G1StallionMatchProductionV5].[dbo].[stg_mv_vendor_metrics_v1];
TRUNCATE TABLE [G1StallionMatchProductionV5].[dbo].[mv_yearling_lot_features_v1];

SELECT 'Tables truncated' AS Status;


-- ============================================================================
-- STEP 2: BUILD #BaseLots (2020-2026)
-- ============================================================================

IF OBJECT_ID('tempdb..#BaseLots') IS NOT NULL DROP TABLE #BaseLots;

SELECT
    LT.Id AS lot_id,
    LT.salesId,
    CAST(SL.startDate AS DATE) AS asOfDate,
    SC.Id AS salesCompanyId,
    SC.salescompanyName AS sale_company,
    SL.salesName AS sale_name,
    YEAR(SL.startDate) AS sale_year,
    LT.bookNumber AS book_number,
    LT.dayNumber AS day_number,
    LT.lotNumber AS lot_number,
    LT.horseGender AS sex,
    H.sireId,
    H.damId,
    LT.vendorId,
    LT.horseId,
    CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
    CAST(ISNULL(LT.isPassedIn, 0) AS BIT) AS isPassedIn,
    CAST(ISNULL(LT.isWithdrawn, 0) AS BIT) AS isWithdrawn,
    CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
    CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
INTO #BaseLots
FROM [G1StallionMatchProductionV5].[dbo].[tblSalesLot] LT
JOIN [G1StallionMatchProductionV5].[dbo].[tblSales] SL ON LT.salesId = SL.Id
JOIN [G1StallionMatchProductionV5].[dbo].[tblSalesCompany] SC ON SL.salesCompanyId = SC.Id
JOIN [G1StallionMatchProductionV5].[dbo].[tblSalesLotType] LTP ON LT.lotType = LTP.Id
JOIN [G1StallionMatchProductionV5].[dbo].[tblCountry] CN ON SL.countryId = CN.id
JOIN [G1StallionMatchProductionV5].[dbo].[tblHorse] H ON LT.horseId = H.id
WHERE CN.countryCode = 'AUS'
    AND LTP.salesLotTypeName = 'Yearling'
    AND YEAR(SL.startDate) BETWEEN 2020 AND 2026
    AND ISNULL(LT.isWithdrawn, 0) = 0;

SELECT COUNT(*) AS BaseLotCount FROM #BaseLots;


-- ============================================================================
-- STEP 3: BUILD #HistLots (Historical reference for lookbacks)
-- ============================================================================

IF OBJECT_ID('tempdb..#HistLots') IS NOT NULL DROP TABLE #HistLots;

SELECT
    LT.Id AS lot_id,
    LT.salesId,
    CAST(SL.startDate AS DATE) AS saleDate,
    H.sireId,
    H.damId,
    LT.vendorId,
    CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
    CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
    CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
INTO #HistLots
FROM [G1StallionMatchProductionV5].[dbo].[tblSalesLot] LT
JOIN [G1StallionMatchProductionV5].[dbo].[tblSales] SL ON LT.salesId = SL.Id
JOIN [G1StallionMatchProductionV5].[dbo].[tblSalesLotType] LTP ON LT.lotType = LTP.Id
JOIN [G1StallionMatchProductionV5].[dbo].[tblCountry] CN ON SL.countryId = CN.id
JOIN [G1StallionMatchProductionV5].[dbo].[tblHorse] H ON LT.horseId = H.id
WHERE CN.countryCode = 'AUS'
    AND LTP.salesLotTypeName = 'Yearling'
    AND ISNULL(LT.isWithdrawn, 0) = 0;

SELECT COUNT(*) AS HistLotCount FROM #HistLots;


-- ============================================================================
-- STEP 4: SALE MEDIAN STAGING
-- ============================================================================

INSERT INTO [G1StallionMatchProductionV5].[dbo].[stg_mv_sale_median_v1] (salesId, session_median_price)
SELECT
    salesId,
    (
        SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hammer_price) OVER ()
        FROM #BaseLots B2
        WHERE B2.salesId = B1.salesId
            AND B2.isSold_int = 1
            AND B2.hammer_price > 0
    ) AS session_median_price
FROM #BaseLots B1
GROUP BY salesId;

SELECT COUNT(*) AS SaleMedianCount FROM [G1StallionMatchProductionV5].[dbo].[stg_mv_sale_median_v1];


-- ============================================================================
-- STEP 5: SIRE METRICS STAGING (36m and 12m lookback)
-- ============================================================================

;WITH SireSaleCombos AS (
    SELECT DISTINCT sireId, salesId, asOfDate
    FROM #BaseLots
    WHERE sireId IS NOT NULL
)
INSERT INTO [G1StallionMatchProductionV5].[dbo].[stg_mv_sire_metrics_v1] (
    sireId, salesId, asOfDate,
    sire_sold_count_36m, sire_passedin_count_36m, sire_total_offered_36m, sire_clearance_rate_36m, sire_median_price_36m,
    sire_sold_count_12m, sire_passedin_count_12m, sire_total_offered_12m, sire_clearance_rate_12m, sire_median_price_12m,
    sire_momentum, sire_sample_flag_36m
)
SELECT
    SSC.sireId,
    SSC.salesId,
    SSC.asOfDate,
    
    -- 36m metrics
    COALESCE(SM36.sire_sold_count_36m, 0),
    COALESCE(SM36.sire_passedin_count_36m, 0),
    COALESCE(SM36.sire_total_offered_36m, 0),
    SM36.sire_clearance_rate_36m,
    SM36.sire_median_price_36m,
    
    -- 12m metrics
    COALESCE(SM12.sire_sold_count_12m, 0),
    COALESCE(SM12.sire_passedin_count_12m, 0),
    COALESCE(SM12.sire_total_offered_12m, 0),
    SM12.sire_clearance_rate_12m,
    SM12.sire_median_price_12m,
    
    -- Momentum = 12m median - 36m median
    CASE 
        WHEN SM12.sire_median_price_12m IS NOT NULL AND SM36.sire_median_price_36m IS NOT NULL
        THEN SM12.sire_median_price_12m - SM36.sire_median_price_36m
    END AS sire_momentum,
    
    -- Sample flag: 1 if >= 5 sold in 36m
    CASE WHEN COALESCE(SM36.sire_sold_count_36m, 0) >= 5 THEN 1 ELSE 0 END AS sire_sample_flag_36m

FROM SireSaleCombos SSC
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS sire_sold_count_36m,
        SUM(HL.isPassedIn_int) AS sire_passedin_count_36m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_36m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_36m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM #HistLots HL2
            WHERE HL2.sireId = SSC.sireId
                AND HL2.saleDate < SSC.asOfDate
                AND HL2.saleDate >= DATEADD(MONTH, -36, SSC.asOfDate)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS sire_median_price_36m
    FROM #HistLots HL
    WHERE HL.sireId = SSC.sireId
        AND HL.saleDate < SSC.asOfDate
        AND HL.saleDate >= DATEADD(MONTH, -36, SSC.asOfDate)
) SM36
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS sire_sold_count_12m,
        SUM(HL.isPassedIn_int) AS sire_passedin_count_12m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_12m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_12m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM #HistLots HL2
            WHERE HL2.sireId = SSC.sireId
                AND HL2.saleDate < SSC.asOfDate
                AND HL2.saleDate >= DATEADD(MONTH, -12, SSC.asOfDate)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS sire_median_price_12m
    FROM #HistLots HL
    WHERE HL.sireId = SSC.sireId
        AND HL.saleDate < SSC.asOfDate
        AND HL.saleDate >= DATEADD(MONTH, -12, SSC.asOfDate)
) SM12;

SELECT COUNT(*) AS SireMetricsCount FROM [G1StallionMatchProductionV5].[dbo].[stg_mv_sire_metrics_v1];


-- ============================================================================
-- STEP 6: DAM STATS STAGING
-- ============================================================================

;WITH DamSaleCombos AS (
    SELECT DISTINCT damId, salesId, asOfDate
    FROM #BaseLots
    WHERE damId IS NOT NULL
)
INSERT INTO [G1StallionMatchProductionV5].[dbo].[stg_mv_dam_stats_v1] (
    damId, salesId, asOfDate,
    dam_progeny_sold_count, dam_progeny_passedin_count, dam_progeny_total_offered_count,
    dam_progeny_median_price, dam_first_foal_flag
)
SELECT
    DSC.damId,
    DSC.salesId,
    DSC.asOfDate,
    
    COALESCE(DM.dam_progeny_sold_count, 0),
    COALESCE(DM.dam_progeny_passedin_count, 0),
    COALESCE(DM.dam_progeny_total_offered_count, 0),
    DM.dam_progeny_median_price,
    CASE WHEN COALESCE(DM.dam_progeny_sold_count, 0) = 0 THEN 1 ELSE 0 END AS dam_first_foal_flag

FROM DamSaleCombos DSC
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS dam_progeny_sold_count,
        SUM(HL.isPassedIn_int) AS dam_progeny_passedin_count,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS dam_progeny_total_offered_count,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM #HistLots HL2
            WHERE HL2.damId = DSC.damId
                AND HL2.saleDate < DSC.asOfDate
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS dam_progeny_median_price
    FROM #HistLots HL
    WHERE HL.damId = DSC.damId
        AND HL.saleDate < DSC.asOfDate
) DM;

SELECT COUNT(*) AS DamStatsCount FROM [G1StallionMatchProductionV5].[dbo].[stg_mv_dam_stats_v1];


-- ============================================================================
-- STEP 7: VENDOR METRICS STAGING (36m lookback)
-- ============================================================================

;WITH VendorSaleCombos AS (
    SELECT DISTINCT vendorId, salesId, asOfDate
    FROM #BaseLots
    WHERE vendorId IS NOT NULL
)
INSERT INTO [G1StallionMatchProductionV5].[dbo].[stg_mv_vendor_metrics_v1] (
    vendorId, salesId, asOfDate,
    vendor_sold_count_36m, vendor_passedin_count_36m, vendor_total_offered_36m,
    vendor_clearance_rate_36m, vendor_median_price_36m,
    vendor_volume_bucket, vendor_first_seen_flag
)
SELECT
    VSC.vendorId,
    VSC.salesId,
    VSC.asOfDate,
    
    COALESCE(VM.vendor_sold_count_36m, 0),
    COALESCE(VM.vendor_passedin_count_36m, 0),
    COALESCE(VM.vendor_total_offered_36m, 0),
    VM.vendor_clearance_rate_36m,
    VM.vendor_median_price_36m,
    
    -- Volume bucket based on 36m sold count
    CASE
        WHEN COALESCE(VM.vendor_sold_count_36m, 0) = 0 THEN 'New'
        WHEN VM.vendor_sold_count_36m BETWEEN 1 AND 5 THEN 'Small'
        WHEN VM.vendor_sold_count_36m BETWEEN 6 AND 20 THEN 'Medium'
        ELSE 'Large'
    END AS vendor_volume_bucket,
    
    CASE WHEN COALESCE(VM.vendor_sold_count_36m, 0) = 0 THEN 1 ELSE 0 END AS vendor_first_seen_flag

FROM VendorSaleCombos VSC
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS vendor_sold_count_36m,
        SUM(HL.isPassedIn_int) AS vendor_passedin_count_36m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS vendor_total_offered_36m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS vendor_clearance_rate_36m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM #HistLots HL2
            WHERE HL2.vendorId = VSC.vendorId
                AND HL2.saleDate < VSC.asOfDate
                AND HL2.saleDate >= DATEADD(MONTH, -36, VSC.asOfDate)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS vendor_median_price_36m
    FROM #HistLots HL
    WHERE HL.vendorId = VSC.vendorId
        AND HL.saleDate < VSC.asOfDate
        AND HL.saleDate >= DATEADD(MONTH, -36, VSC.asOfDate)
) VM;

SELECT COUNT(*) AS VendorMetricsCount FROM [G1StallionMatchProductionV5].[dbo].[stg_mv_vendor_metrics_v1];


-- ============================================================================
-- STEP 8: ASSEMBLE GOLDEN TABLE
-- ============================================================================

INSERT INTO [G1StallionMatchProductionV5].[dbo].[mv_yearling_lot_features_v1] (
    lot_id, salesId, asOfDate, salesCompanyId, sale_company, sale_name, sale_year,
    book_number, day_number, lot_number, sex,
    sireId, damId, vendorId, horseId,
    session_median_price,
    hammer_price, isPassedIn, isWithdrawn,
    price_index, log_price_index,
    sire_sold_count_36m, sire_passedin_count_36m, sire_total_offered_36m, sire_clearance_rate_36m, sire_median_price_36m,
    sire_sold_count_12m, sire_passedin_count_12m, sire_total_offered_12m, sire_clearance_rate_12m, sire_median_price_12m,
    sire_momentum, sire_sample_flag_36m,
    dam_progeny_sold_count, dam_progeny_passedin_count, dam_progeny_total_offered_count, dam_progeny_median_price, dam_first_foal_flag,
    vendor_sold_count_36m, vendor_passedin_count_36m, vendor_total_offered_36m, vendor_clearance_rate_36m, vendor_median_price_36m,
    vendor_volume_bucket, vendor_first_seen_flag,
    sire_vs_sale_median_delta, dam_vs_sale_median_delta, vendor_vs_sale_median_delta,
    feature_contract_version, feature_generated_at
)
SELECT
    BL.lot_id,
    BL.salesId,
    BL.asOfDate,
    BL.salesCompanyId,
    BL.sale_company,
    BL.sale_name,
    BL.sale_year,
    BL.book_number,
    BL.day_number,
    BL.lot_number,
    BL.sex,
    BL.sireId,
    BL.damId,
    BL.vendorId,
    BL.horseId,
    SM.session_median_price,
    BL.hammer_price,
    BL.isPassedIn,
    BL.isWithdrawn,
    
    -- Price index (target variable)
    CASE
        WHEN BL.hammer_price > 0 AND SM.session_median_price > 0
        THEN CAST(BL.hammer_price / SM.session_median_price AS DECIMAL(10,4))
    END AS price_index,
    
    CASE
        WHEN BL.hammer_price > 0 AND SM.session_median_price > 0
        THEN CAST(LOG(BL.hammer_price / SM.session_median_price) AS DECIMAL(10,6))
    END AS log_price_index,
    
    -- Sire metrics
    SR.sire_sold_count_36m, SR.sire_passedin_count_36m, SR.sire_total_offered_36m, SR.sire_clearance_rate_36m, SR.sire_median_price_36m,
    SR.sire_sold_count_12m, SR.sire_passedin_count_12m, SR.sire_total_offered_12m, SR.sire_clearance_rate_12m, SR.sire_median_price_12m,
    SR.sire_momentum, SR.sire_sample_flag_36m,
    
    -- Dam metrics
    DS.dam_progeny_sold_count, DS.dam_progeny_passedin_count, DS.dam_progeny_total_offered_count, DS.dam_progeny_median_price, DS.dam_first_foal_flag,
    
    -- Vendor metrics
    VD.vendor_sold_count_36m, VD.vendor_passedin_count_36m, VD.vendor_total_offered_36m, VD.vendor_clearance_rate_36m, VD.vendor_median_price_36m,
    VD.vendor_volume_bucket, VD.vendor_first_seen_flag,
    
    -- Delta features
    CASE WHEN SR.sire_median_price_36m IS NOT NULL AND SM.session_median_price > 0
         THEN CAST(SR.sire_median_price_36m - SM.session_median_price AS DECIMAL(12,2)) END AS sire_vs_sale_median_delta,
    CASE WHEN DS.dam_progeny_median_price IS NOT NULL AND SM.session_median_price > 0
         THEN CAST(DS.dam_progeny_median_price - SM.session_median_price AS DECIMAL(12,2)) END AS dam_vs_sale_median_delta,
    CASE WHEN VD.vendor_median_price_36m IS NOT NULL AND SM.session_median_price > 0
         THEN CAST(VD.vendor_median_price_36m - SM.session_median_price AS DECIMAL(12,2)) END AS vendor_vs_sale_median_delta,
    
    'v1.0' AS feature_contract_version,
    SYSUTCDATETIME() AS feature_generated_at

FROM #BaseLots BL
JOIN [G1StallionMatchProductionV5].[dbo].[stg_mv_sale_median_v1] SM ON SM.salesId = BL.salesId
LEFT JOIN [G1StallionMatchProductionV5].[dbo].[stg_mv_sire_metrics_v1] SR ON SR.salesId = BL.salesId AND SR.sireId = BL.sireId
LEFT JOIN [G1StallionMatchProductionV5].[dbo].[stg_mv_dam_stats_v1] DS ON DS.salesId = BL.salesId AND DS.damId = BL.damId
LEFT JOIN [G1StallionMatchProductionV5].[dbo].[stg_mv_vendor_metrics_v1] VD ON VD.salesId = BL.salesId AND VD.vendorId = BL.vendorId;


-- ============================================================================
-- STEP 9: VERIFICATION
-- ============================================================================

SELECT 
    sale_year,
    COUNT(*) AS TotalLots,
    COUNT(CASE WHEN isPassedIn = 0 AND hammer_price > 0 THEN 1 END) AS SoldLots,
    COUNT(CASE WHEN session_median_price IS NULL THEN 1 END) AS NullMedian
FROM [G1StallionMatchProductionV5].[dbo].[mv_yearling_lot_features_v1]
GROUP BY sale_year
ORDER BY sale_year;


-- ============================================================================
-- CLEANUP TEMP TABLES
-- ============================================================================

DROP TABLE IF EXISTS #BaseLots;
DROP TABLE IF EXISTS #HistLots;

SELECT 'Rebuild complete' AS Status;