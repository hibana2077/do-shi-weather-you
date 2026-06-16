# Datasets

The monthly station files were downloaded manually from [CODIS](https://codis.cwa.gov.tw/StationData/).
Each monthly CSV uses English column names only. The aggregated file `agg_data.csv`
is produced by:

```bash
.venv/bin/python src/lab/data/agg_data.py
```

The study uses daily observations from Taiwan CWA station `467050` and forecasts
the next-day mean temperature.
