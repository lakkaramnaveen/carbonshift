# Data

| File | Source | Licence |
|---|---|---|
| `combined_carbon.csv` | Hourly average carbon intensity (gCO2eq/kWh), 123 zones, 2020-2022, from [Electricity Maps](https://www.electricitymaps.com), as released with Sukprasert et al. (EuroSys'24) in [umassos/decarbonization-potential](https://github.com/umassos/decarbonization-potential) ([Zenodo 10790855](https://zenodo.org/records/10790855)) | ODbL 1.0 (database), DbCL 1.0 (contents) |
| `gcp_latency_matrix.csv`, `gcp_dc_zonecode_mapper.json` | GCP inter-region latency, AT&T Center for Virtualization, SMU (via the same artifact) | Apache 2.0 |
| `region_groups.csv`, `provider_regions.csv` | Zone → continent and cloud-provider membership (same artifact, `global_modules/static_files`) | Apache 2.0 |
| `google_power/*.csv.gz` | Google PowerData2019: 5-minute power utilisation of 57 data-centre power domains, May 2019 (github.com/google/cluster-data, public bucket gs://powerdata_2019); Sakalkar et al., ASPLOS 2020 | CC-BY 4.0 |

Files are copied unchanged. The original repository was cloned to `external/` (git-ignored) for reference only. None of its code is executed here; `src/carbonshift` is a clean re-implementation.
