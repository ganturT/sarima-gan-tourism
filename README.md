# SARIMA-GAN: adversarial residual correction for tourism demand forecasting

Code for the paper

> G. Togtokh and S. Khurelbaatar, *A hybrid SARIMA-GAN model with adversarial residual correction for forecasting short and seasonal tourism demand series* 

SARIMA-GAN adds to the one-step SARIMA forecast a forecast of the SARIMA residual made by the generator of a
generative adversarial network (GAN). The generator is a two-layer GRU network; the discriminator is a
one-dimensional convolutional network that judges sequences of 13 residuals (the next residual and the previous 12).
The models are evaluated on the monthly international tourist arrivals to Mongolia (January 2017 to October 2025)
with four rolling origins and ten random seeds.

## Repository structure

```
Code/      all scripts (run them from the repository root, see below)
Data/      main_dataset.csv: monthly arrivals (columns: date, volume)
Results/   outputs used in the paper (forecasts, metrics, tuning logs, tables and figures)
```

The `Results/` folder contains the outputs reported in the paper, so the tables and figures can be checked
without retraining. Rerunning the training scripts may give slightly different numbers on other hardware or
with other library versions.

## Requirements

Python 3.12.2 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

`python Code/Versions.py` prints the installed versions in the same format.
The experiments were run on a CPU (Intel Core i7-1165G7, 8 GB RAM, Windows 11) without a GPU.
The GAN scripts run every training run in a separate process to release memory after each run.

## How to reproduce the results

Run the scripts **from the repository root** (for example `python Code/BM_SARIMA_rolling.py`), in this order.

| Step | Script | Model / output |
|---|---|---|
| 1 | `Data_preprocessing_rolling.py` | input windows, splits and scalers of the four origins (`Data/origin1` ... `origin4`) |
| 2 | `BM_SARIMA_rolling.py` | SARIMA forecasts and residuals (`Results/sarima_origin1.csv` ...), needed by all hybrid models |
| 3 | `BM_SNAIVE_rolling.py` | seasonal naive benchmark |
| 4 | `BM_ANN_rolling.py`, `BM_GRU_rolling.py` | stand-alone ANN and GRU (MSE loss) |
| 5 | `GAN_ANN_bce_rolling.py`, `GAN_GRU_bce_rolling.py` | stand-alone GAN with an ANN or a GRU generator |
| 6 | `BM_SARIMA_ANN_rolling.py`, `BM_SARIMA_GRU_rolling.py` | SARIMA hybrids with MSE-trained residual models |
| 7 | `HY_SARIMA_GAN_ANN_rolling.py` | SARIMA-GAN with an ANN generator |
| 8 | `HY_SARIMA_GAN_GRU_rolling.py` | **SARIMA-GAN (proposed model)** and its training curves (Fig. 3) |
| 9 | `PR_testset_rolling_table.py` | test accuracy, mean ± standard deviation over origins (Table 3) |
| 10 | `PR_dm_test.py` | Diebold–Mariano tests with the Harvey–Leybourne–Newbold correction (Table 5) |
| 11 | `PR_lb_test.py` | Ljung–Box test on the SARIMA training residuals (Table 6) and their ACF |
| 12 | `PR_SARIMA_GAN_model_testset_plots_4origns.py` | forecasts of SARIMA and SARIMA-GAN on the four origins (Fig. 2) |

Steps 4 to 8 can be run in any order once steps 1 and 2 are done. The `PR_` scripts (steps 9 to 12)
only read the files in `Results/`.


## Citation

If you use this code, please cite the paper (the reference will be added after publication).

## License

See `LICENSE`.
