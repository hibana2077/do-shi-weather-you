# Disagreement-Aware Local Weather Forecasting

This repository studies next-day mean temperature forecasting for Taiwan CWA
station `467050` using daily station observations from CODIS. The original
notebook workflow has been replaced by reproducible Python scripts.

## Research Question

The project tests whether a lightweight hybrid of an LTSF-Linear branch and an
LSTM branch can improve small-data, single-station temperature forecasting. The
candidate novelty is a learned fusion head that may use the absolute
disagreement between the two branch predictions as an explicit meta-feature.

The expanded experiment does not support a broad claim that the hybrid is best.
Under a chronological evaluation, a persistence baseline is the strongest model
for next-day temperature. The useful research finding is diagnostic: temporal
continuity dominates this local one-step task, and branch disagreement is not a
reliable positive feature without stronger regularization or a harder horizon.

## Repository Layout

```text
src/lab/data/agg_data.py          Aggregate monthly English CSV files
src/lab/data/agg_data.csv         Chronological daily dataset with English headers
src/lab/model.py                  Forecasting models and ablations
src/lab/train.py                  Full experiment runner
src/lab/results/result.py         Matplotlib figure generator
src/lab/results/runs/             Per-run JSON outputs
src/lab/results/predictions/      Per-run prediction CSV outputs
paper/                            IEEE-style draft and figures
```

There are no notebooks in the workflow.

## Reproduce

```bash
.venv/bin/python src/lab/data/agg_data.py
.venv/bin/python src/lab/train.py \
  --models persistence,weekly_naive,rolling_mean_7,ltsf_linear,lstm,fusion_no_disagreement,fusion \
  --windows 7,14,30,60,90 \
  --seeds 13,29,47 \
  --epochs 80 \
  --patience 12 \
  --batch-size 64
.venv/bin/python src/lab/results/result.py
```

## Final Results

The evaluation uses chronological splits: 70% train, 15% validation, and 15%
test. Metrics are averaged over seeds where the model is stochastic.

| Rank | Model | Window | MAE (deg C) | RMSE (deg C) | MAPE (%) | R2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | persistence | 60 | 1.123 | 1.559 | 5.437 | 0.913 |
| 2 | persistence | 90 | 1.126 | 1.565 | 5.463 | 0.914 |
| 3 | persistence | 30 | 1.161 | 1.638 | 5.627 | 0.903 |
| 4 | persistence | 14 | 1.164 | 1.637 | 5.639 | 0.903 |
| 5 | persistence | 7 | 1.173 | 1.649 | 5.675 | 0.901 |
| 6 | rolling_mean_7 | 90 | 1.452 | 1.998 | 7.066 | 0.859 |
| 7 | rolling_mean_7 | 60 | 1.454 | 1.999 | 7.057 | 0.857 |
| 8 | fusion_no_disagreement | 7 | 1.564 | 2.162 | 7.827 | 0.829 |
| 9 | lstm | 7 | 1.709 | 2.236 | 8.462 | 0.818 |
| 10 | lstm | 90 | 1.730 | 2.236 | 8.694 | 0.824 |

The best learned model is the 7-day fusion model without the disagreement
feature. The full disagreement-aware fusion is worse at 7 days (MAE 1.967) and
90 days (MAE 2.012), and it is weak at intermediate windows. This rejects the
original positive pilot interpretation.

## Figures

The final figures are generated with Matplotlib, use no titles, and enlarge axis
ticks and value labels.

- `src/lab/results/aggregate_mae.png`
- `src/lab/results/disagreement_ablation.png`
- `src/lab/results/best_prediction_trace.png`

## Main Limitation

This is a single-station, one-step-ahead dataset. Persistence is a very strong
baseline because next-day mean temperature changes smoothly. Future work should
test longer horizons, rolling-origin evaluation, additional stations, and
regularized fusion constraints before making stronger novelty claims.
