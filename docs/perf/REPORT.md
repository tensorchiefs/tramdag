# tramdag cross-machine benchmark

Fixed 200-epoch workloads (see `experiments/perf_machine.py`); `final_val_nll` must agree across machines (same seed & data).

| host | chip | gpu | code | workload | device | fit_s | epochs_per_s | sample_100k_s | final_val_nll |
|---|---|---|---|---|---|---|---|---|---|
| MAC-MINI | arm | mps | c9d84bf | intro | cpu | 6.52 | 30.66 | 1.8 | 5.3741 |
| MAC-MINI | arm | mps | c9d84bf | intro | mps | 20.03 | 9.98 | 0.91 | 5.3736 |
| MAC-MINI | arm | mps | c9d84bf | large | cpu | 20.73 | 9.65 | 1.71 | 4.9152 |
| MAC-MINI | arm | mps | c9d84bf | large | mps | 31.6 | 6.33 | 0.89 | 4.9132 |
