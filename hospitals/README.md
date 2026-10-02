# Hospital lists: definitions and provenance

Verified 2 October 2026 against county EMS agency (LEMSA) receiving-facility lists, the Rhode Island Department of Health stroke designation list, hospital and health-system pages, and the 2024 national roster of accredited Thrombectomy-Capable and Comprehensive Stroke Centers. Coordinates checked against each campus's street address (all within ~300 m). No acute-care hospital with a general 24-hour emergency department in the nine Bay Area counties or in Rhode Island is missing.

## Model type (column `Type`)

| Type | Meaning in the MDP | Includes |
|---|---|---|
| CSC | Endovascular thrombectomy on site; no onward transfer | Joint Commission Comprehensive Stroke Center (CSC) or Thrombectomy-Capable Stroke Center (TSC); DNV Comprehensive; or a LEMSA thrombectomy/comprehensive designation |
| PSC | IV thrombolysis on site; LVO transferred for EVT | Joint Commission or DNV Primary / Advanced Primary Stroke Center; Acute Stroke Ready Hospital (ASRH) |
| NSC | Non-stroke-center hospital: 24-h emergency department, no stroke certification (was `CLINIC` before 2 Oct 2026) | |

`CertLevel` records the verified certification behind the `Type`. `EVTOnSite` is `yes` for every hospital known to perform thrombectomy on site, certified or not; setting the environment variable `EVT_DEFINITION=onsite` before running any script retypes all `EVTOnSite == yes` hospitals as CSC (the default, `certified`, uses `Type` as written). Where it says "unverified", the designation could not be confirmed from a current primary source under the hospital's current name.

Hospitals that perform thrombectomy without CSC/TSC certification or a LEMSA EVT designation are typed PSC in the primary analysis and noted in `CertLevel` ("EVT on site (uncertified)"): CPMC Van Ness, Zuckerberg SF General, Washington Hospital Fremont, Alta Bates Summit (Summit campus). Run with `EVT_DEFINITION=onsite` for the sensitivity analysis that retypes these as CSC.

## Changes made 2 October 2026

| Hospital | Was | Now | Basis |
|---|---|---|---|
| CPMC Davies Campus | PSC | CSC | TJC CSC certification announced 1 Jun 2026 (Sutter Health) |
| CPMC Mission Bernal Campus | NSC (CLINIC) | PSC | Newly TJC PSC (Sutter Health, Jun 2026) |
| Alta Bates Summit, Alta Bates (Berkeley) campus | PSC | NSC | Alameda County EMS: basic emergency services only; the PSC designation is the Summit campus |
| Westerly Hospital (RI) | NSC (CLINIC) | PSC | TJC Acute Stroke Ready Hospital (recertified Aug 2023); consistent with Sonoma Valley (ASRH) in the PSC bucket |
| Women & Infants Hospital (RI) | NSC (CLINIC) | removed | Women's specialty ED; not a 911 stroke destination |

Display-name updates (keys unchanged): Washington Health (Washington Hospital); Wilma Chan Highland Hospital Campus; Providence Healdsburg Hospital. UCSF Health Hyde and Stanyan Hospitals are the former Saint Francis Memorial and St Mary's (renamed 14 Oct 2025). Rhode Island: Lifespan is now Brown University Health; Fatima and Roger Williams passed to CharterCARE Health of Rhode Island (Centurion Foundation) in March 2026.

## Terminology for the paper

Use capability-based names rather than certification names: EVT-capable centre (code: CSC), thrombolysis-capable centre (code: PSC), non-stroke-center hospital (code: NSC). Reviewer 2 on the JAMA Network Open submission objected both to "clinic" for a hospital and to using PSC/CSC as if they were capability classes.

## EMS routing context

San Francisco EMS Policy 5000 (Apr 2025) routes stroke patients to the closest designated stroke centre with no LVO tiering. Santa Clara County Policy 602 (Sep 2025) routes Comprehensive Stroke Alert patients to the closest Comprehensive Stroke Center (five in county). Marin County transports LVO patients out of county, usually by air. Sonoma County has no CSC or TSC; the nearest are in Marin-adjacent San Francisco and the East Bay, 79 to 103 km from the Santa Rosa hospitals.

## Principal sources

- Alameda County EMS receiving facilities: https://health.alamedacountyca.gov/for-providers-ems/receiving-facilities-hospitals/
- Contra Costa EMS stroke system: https://www.cchealth.org/about-contra-costa-health/divisions/ems/systems-of-care/stroke-system
- San Francisco EMS Policy 5000 (Destination) and 5020 (Diversion, Feb 2026): https://media.api.sf.gov/documents/
- San Mateo County EMS: FAC-1 Receiving Hospitals (2021); 2024 EMS Plan
- Santa Clara County EMS Policy 602 via AO 2025-006: https://files.santaclaracounty.gov/exjcpb1541/2025-08/administrative-order-2025-006.pdf
- Marin 2023-24, Napa 2018, Solano 2019, Coastal Valleys 2019-23 EMS plans (California EMSA postings)
- 2024 national TSC/CSC roster: https://cdn.amegroups.cn/static/public/asj-24-45-1.pdf
- Sutter Health, CPMC Davies CSC (1 Jun 2026): https://vitals.sutterhealth.org/sutters-cpmc-davies-campus-achieves-comprehensive-stroke-center-certification-from-the-joint-commission/
- Providence Santa Rosa Memorial stroke center (Advanced PSC): https://www.providence.org/locations/norcal/santa-rosa-memorial-hospital/stroke-center
- Rhode Island DOH Stroke Task Force designation list: https://health.ri.gov/heart-disease-and-stroke/rhode-island-stroke-task-force
- Brown University Health, Rhode Island Hospital CSC: https://www.brownhealth.org/centers-services/comprehensive-stroke-center
- Yale New Haven Health, Westerly ASRH recertification (Aug 2023): https://www.ynhhs.org/news/westerly-hospital-recertified-from-the-joint-commission-for-its-acute-stroke-ready-hospital-program

Not accessible at verification time: Joint Commission Quality Check records (403), San Mateo EMS Policies 519 and 522 (2025), SF EMS Policy 5010, Alameda EMS PDF host, current Coastal Valleys and Marin destination policies.
