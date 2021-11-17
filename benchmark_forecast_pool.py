import os
import fire
from joblib import Parallel, delayed
from nbm_bench.io import run_cmd

MODELS = [
    "ARIMA",
    "SARIMA",
    "AutoSARIMA",
    "ETS",
    "MSES",
    "Prophet",
    "AutoProphet",
    "VAR",
    "RandomForestForecaster",
    "ExtraTreesForecaster",
    "LGBMForecaster",
    "InformerForecaster",
]
DATASETS = ["EnergyPower", "SeattleTrail", "SolarPlant"]


def run_single(args):
    model, dataset, dry, forecast_dir = args
    cmd = f"python benchmark_forecast.py --dataset {dataset} --model {model} --debug --visualize --forecast_dir {forecast_dir}"
    if dry:
        print("dry run")
        print(cmd)
    else:
        run_cmd(cmd)


def main(models="all", datasets="all", dry=False, forecast_dir="forecast", n_jobs=-2):
    if models == "all":
        models = MODELS

    if datasets == "all":
        datasets = DATASETS

    job_args = []
    for dataset in datasets:
        job_args.extend([(model, dataset, dry, forecast_dir) for model in models])

    pool = Parallel(n_jobs=n_jobs)(delayed(run_single)(args) for args in job_args)


if __name__ == "__main__":
    fire.Fire(main)
