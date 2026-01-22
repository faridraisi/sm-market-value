USE G1StallionMatchProductionV5
GO
SELECT
    F.lot_id, 
    F.horseId, 
    F.salesId, 
    F.lot_number, 
    F.book_number,
    F.sex,
    H.horseName AS horse_name,
    S.horseName AS sire_name,
    F.sale_company, 
    F.sale_year, 
    F.day_number,
    F.session_median_price,
    F.sire_sold_count_36m, F.sire_total_offered_36m, F.sire_clearance_rate_36m, F.sire_median_price_36m,
    F.sire_sold_count_12m, F.sire_total_offered_12m, F.sire_clearance_rate_12m, F.sire_median_price_12m,
    F.sire_momentum, F.sire_sample_flag_36m,
    F.dam_progeny_sold_count, F.dam_progeny_total_offered_count, F.dam_progeny_median_price, F.dam_first_foal_flag,
    F.vendor_sold_count_36m, F.vendor_total_offered_36m, F.vendor_clearance_rate_36m, F.vendor_median_price_36m,
    F.vendor_volume_bucket, F.vendor_first_seen_flag
FROM dbo.mv_yearling_lot_features_v1 F
LEFT JOIN dbo.tblHorse H ON F.horseId = H.id
LEFT JOIN dbo.tblHorse S ON F.sireId = S.id
WHERE F.salesId = 2005
ORDER BY F.lot_number;