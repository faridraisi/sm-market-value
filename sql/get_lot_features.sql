-- get_lot_features.sql
-- Parameterized query to fetch all features for a single sale
-- Parameters: @sale_id (INT), @country_code (VARCHAR)
--
-- This query builds features on-the-fly without requiring staging tables.
-- It uses OUTER APPLY for lookback calculations (sire/dam/vendor metrics).

DECLARE @as_of_date DATE = (SELECT startDate FROM tblSales WHERE Id = @sale_id);

-- Session median for this sale (sold lots only)
;WITH SessionMedian AS (
    SELECT TOP 1
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LT.price) OVER () AS session_median_price
    FROM tblSalesLot LT
    WHERE LT.salesId = @sale_id
        AND LT.price > 0
        AND ISNULL(LT.isPassedIn, 0) = 0
        AND ISNULL(LT.isWithdrawn, 0) = 0
),
-- Historical lots for lookback calculations
HistLots AS (
    SELECT
        LT.Id AS lot_id,
        LT.salesId,
        CAST(SL.startDate AS DATE) AS saleDate,
        H.sireId,
        H.damId,
        LT.vendorId,
        CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
        CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0
             AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
        CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
    FROM tblSalesLot LT
    JOIN tblSales SL ON LT.salesId = SL.Id
    JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
    JOIN tblCountry CN ON SL.countryId = CN.id
    JOIN tblHorse H ON LT.horseId = H.id
    WHERE CN.countryCode = @country_code
        AND LTP.salesLotTypeName = 'Yearling'
        AND ISNULL(LT.isWithdrawn, 0) = 0
        AND SL.startDate < @as_of_date  -- Only historical data
)
SELECT
    LT.Id AS lot_id,
    LT.salesId,
    LT.horseId,
    LT.lotNumber AS lot_number,
    LT.bookNumber AS book_number,
    LT.dayNumber AS day_number,
    LT.horseGender AS sex,
    LT.price AS hammer_price,
    H.horseName AS horse_name,
    SIRE.horseName AS sire_name,
    H.sireId,
    H.damId,
    LT.vendorId,
    SM.session_median_price,
    SC.salescompanyName AS sale_company,

    -- Sire 36m metrics
    COALESCE(SR36.sire_sold_count_36m, 0) AS sire_sold_count_36m,
    COALESCE(SR36.sire_total_offered_36m, 0) AS sire_total_offered_36m,
    SR36.sire_clearance_rate_36m,
    SR36.sire_median_price_36m,
    CASE WHEN COALESCE(SR36.sire_sold_count_36m, 0) >= 5 THEN 1 ELSE 0 END AS sire_sample_flag_36m,

    -- Sire 12m metrics
    COALESCE(SR12.sire_sold_count_12m, 0) AS sire_sold_count_12m,
    COALESCE(SR12.sire_total_offered_12m, 0) AS sire_total_offered_12m,
    SR12.sire_clearance_rate_12m,
    SR12.sire_median_price_12m,

    -- Sire momentum
    CASE
        WHEN SR12.sire_median_price_12m IS NOT NULL AND SR36.sire_median_price_36m IS NOT NULL
        THEN SR12.sire_median_price_12m - SR36.sire_median_price_36m
    END AS sire_momentum,

    -- Dam stats
    COALESCE(DS.dam_progeny_sold_count, 0) AS dam_progeny_sold_count,
    COALESCE(DS.dam_progeny_total_offered_count, 0) AS dam_progeny_total_offered_count,
    DS.dam_progeny_median_price,
    CASE WHEN COALESCE(DS.dam_progeny_sold_count, 0) = 0 THEN 1 ELSE 0 END AS dam_first_foal_flag,

    -- Vendor metrics
    COALESCE(VD.vendor_sold_count_36m, 0) AS vendor_sold_count_36m,
    COALESCE(VD.vendor_total_offered_36m, 0) AS vendor_total_offered_36m,
    VD.vendor_clearance_rate_36m,
    VD.vendor_median_price_36m,
    CASE
        WHEN COALESCE(VD.vendor_sold_count_36m, 0) = 0 THEN 'New'
        WHEN VD.vendor_sold_count_36m BETWEEN 1 AND 5 THEN 'Small'
        WHEN VD.vendor_sold_count_36m BETWEEN 6 AND 20 THEN 'Medium'
        ELSE 'Large'
    END AS vendor_volume_bucket,
    CASE WHEN COALESCE(VD.vendor_sold_count_36m, 0) = 0 THEN 1 ELSE 0 END AS vendor_first_seen_flag

FROM tblSalesLot LT
JOIN tblSales SL ON LT.salesId = SL.Id
JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
JOIN tblHorse H ON LT.horseId = H.id
LEFT JOIN tblHorse SIRE ON H.sireId = SIRE.id
CROSS JOIN SessionMedian SM

-- Sire 36m lookback
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS sire_sold_count_36m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_36m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_36m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM HistLots HL2
            WHERE HL2.sireId = H.sireId
                AND HL2.saleDate < @as_of_date
                AND HL2.saleDate >= DATEADD(MONTH, -36, @as_of_date)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS sire_median_price_36m
    FROM HistLots HL
    WHERE HL.sireId = H.sireId
        AND HL.saleDate < @as_of_date
        AND HL.saleDate >= DATEADD(MONTH, -36, @as_of_date)
) SR36

-- Sire 12m lookback
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS sire_sold_count_12m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_12m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_12m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM HistLots HL2
            WHERE HL2.sireId = H.sireId
                AND HL2.saleDate < @as_of_date
                AND HL2.saleDate >= DATEADD(MONTH, -12, @as_of_date)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS sire_median_price_12m
    FROM HistLots HL
    WHERE HL.sireId = H.sireId
        AND HL.saleDate < @as_of_date
        AND HL.saleDate >= DATEADD(MONTH, -12, @as_of_date)
) SR12

-- Dam stats (all-time prior)
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS dam_progeny_sold_count,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS dam_progeny_total_offered_count,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM HistLots HL2
            WHERE HL2.damId = H.damId
                AND HL2.saleDate < @as_of_date
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS dam_progeny_median_price
    FROM HistLots HL
    WHERE HL.damId = H.damId
        AND HL.saleDate < @as_of_date
) DS

-- Vendor 36m lookback
OUTER APPLY (
    SELECT
        SUM(HL.isSold_int) AS vendor_sold_count_36m,
        SUM(HL.isSold_int + HL.isPassedIn_int) AS vendor_total_offered_36m,
        CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS vendor_clearance_rate_36m,
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
            FROM HistLots HL2
            WHERE HL2.vendorId = LT.vendorId
                AND HL2.saleDate < @as_of_date
                AND HL2.saleDate >= DATEADD(MONTH, -36, @as_of_date)
                AND HL2.isSold_int = 1
                AND HL2.hammer_price > 0
        ) AS vendor_median_price_36m
    FROM HistLots HL
    WHERE HL.vendorId = LT.vendorId
        AND HL.saleDate < @as_of_date
        AND HL.saleDate >= DATEADD(MONTH, -36, @as_of_date)
) VD

WHERE LT.salesId = @sale_id
    AND ISNULL(LT.isWithdrawn, 0) = 0
ORDER BY LT.lotNumber;
