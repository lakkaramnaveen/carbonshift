# Data

| File | Source | Licence |
|---|---|---|
| `combined_carbon.csv` | Hourly average carbon intensity (gCO2eq/kWh), 123 zones, 2020-2022, from [Electricity Maps](https://www.electricitymaps.com), as released with Sukprasert et al. (EuroSys'24) in [umassos/decarbonization-potential](https://github.com/umassos/decarbonization-potential) ([Zenodo 10790855](https://zenodo.org/records/10790855)) | ODbL 1.0 (database), DbCL 1.0 (contents) |
| `gcp_latency_matrix.csv`, `gcp_dc_zonecode_mapper.json` | GCP inter-region latency, AT&T Center for Virtualization, SMU (via the same artifact) | Apache 2.0 |
| `region_groups.csv`, `provider_regions.csv` | Zone → continent and cloud-provider membership (same artifact, `global_modules/static_files`) | Apache 2.0 |
| `google_power/*.csv.gz` | Google PowerData2019: 5-minute power utilisation of 57 data-centre power domains, May 2019 (github.com/google/cluster-data, public bucket gs://powerdata_2019); Sakalkar et al., ASPLOS 2020 | CC-BY 4.0 |
| `eia930/signals_2022.csv.gz` | Derived hourly average and marginal carbon signals for 53 US balancing authorities, built by `src/carbonshift/eia930.py` from EIA-930 six-month balance files (raw files in `eia930/raw/`, git-ignored, re-downloadable from eia.gov/electricity/gridmonitor) | Public domain (U.S. government data) |
| `caiso/caiso_hourly_{2021..2024}.csv.gz` | Hourly CAISO load, solar, wind, thermal, imports and wind+solar curtailment (MWh), aggregated by `src/carbonshift/curtailment.py` from CAISO's public "Production and Curtailments Data" workbooks (5-minute MW; raw files in `caiso/raw/`, git-ignored, re-downloadable from caiso.com/documents/productionandcurtailmentsdata_<year>.xlsx; 2024 is production-and-curtailments-data-2024.xlsx) | Publicly posted by CAISO, provided as is (not settlement quality) |
| `weather/hub_weather_2022.csv.gz` | Hourly 2-m dry-bulb and wet-bulb temperature (UTC, 0.1 °C) at one data-centre hub per cloud-hosting zone (`carbonshift.cooling.HUBS`), from the [Open-Meteo](https://open-meteo.com) historical archive (ERA5 reanalysis, Copernicus) | CC BY 4.0 |

Files are copied unchanged, except the derived EIA signals. The original repository was cloned to `external/` (git-ignored) for reference only. None of its code is executed here; `src/carbonshift` is a clean re-implementation.
