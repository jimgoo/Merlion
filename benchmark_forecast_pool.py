import os
import fire
from joblib import Parallel, delayed
from nbm_bench.io import run_cmd

MODELS = [
    "Arima",
    "Sarima",
    "AutoSarima",
    "ETS",
    "MSES",
    "Prophet",
    "AutoProphet",
    "VectorAR",
    "RandomForestForecaster",
    "ExtraTreesForecaster",
    "LGBMForecaster",
    "InformerForecaster",
    "RepeatRecent",
    # "\{Arima,Prophet\}"
]

DATASETS = ["EnergyPower", "SeattleTrail", "SolarPlant", "ETT", "WTH", "ECL"]


def run_single(args):
    model, dataset, dry, forecast_dir, n_train, n_test = args
    cmd = f"""
    python benchmark_forecast.py \
        --dataset {dataset} \
        --models {model} \
        --debug --visualize \
        --forecast_dir {forecast_dir} \
        --n_train {n_train} \
        --n_test {n_test}
    """
    if dry:
        print("dry run")
        print(cmd)
    else:
        run_cmd(cmd)


def main(
    models="all",
    datasets="all",
    forecast_dir="forecast",
    n_train=None,
    n_test=None,
    n_jobs=-2,
    dry=False,
):
    if models == "all":
        models = MODELS

    if datasets == "all":
        datasets = DATASETS

    job_args = []
    for dataset in datasets:
        job_args.extend(
            [(model, dataset, dry, forecast_dir, n_train, n_test) for model in models]
        )

    pool = Parallel(n_jobs=n_jobs)(delayed(run_single)(args) for args in job_args)


if __name__ == "__main__":
    fire.Fire(main)
