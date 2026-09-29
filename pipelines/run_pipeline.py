import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_ALS_MAX_ITER,
    DEFAULT_ALS_RANK,
    DEFAULT_ALS_REG_PARAM,
    DEFAULT_TRAIN_RATIO,
    RESULTS_DIR,
    get_hdfs_paths,
)
from src.data import (
    ingest_dataset_to_hdfs,
    load_movies,
    load_ratings,
    preprocess_ratings,
    split_train_test_by_time,
)
from src.hdfs import hdfs_mkdir, hdfs_write_text
from src.metrics import (
    evaluate_popularity_baseline,
    evaluate_ranking_metrics,
    evaluate_regression,
)
from src.models import (
    run_grid_tuning,
    save_spark_model,
    train_als,
)
from src.spark import check_system_resources, get_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MovieLensPipeline")


def run_baseline_pipeline(dataset_size: str = "100k") -> Dict[str, Any]:
    """Execute baseline ALS pipeline for specified dataset size."""
    paths = get_hdfs_paths(dataset_size)
    total_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Executing MovieLens %s Baseline Pipeline", dataset_size.upper())
    if dataset_size == "25m":
        check_system_resources()

    partitions = 4 if dataset_size == "100k" else 16
    spark = get_spark_session(
        app_name=f"MovieLens-{dataset_size}-Baseline",
        shuffle_partitions=80 if dataset_size == "25m" else 8,
        checkpoint_dir=paths.checkpoints if dataset_size == "25m" else None,
    )

    try:
        ingest_dataset_to_hdfs(spark, dataset_size=dataset_size)
        raw_ratings = load_ratings(spark, dataset_size=dataset_size)
        load_movies(spark, dataset_size=dataset_size)

        cleaned_ratings = preprocess_ratings(
            spark=spark,
            ratings_df=raw_ratings,
            dataset_size=dataset_size,
            num_partitions=partitions,
        )

        train_df, test_df, _, train_count, test_count = split_train_test_by_time(
            ratings_df=cleaned_ratings,
            train_ratio=DEFAULT_TRAIN_RATIO,
        )
        train_df.cache()

        model, train_time = train_als(
            train_df=train_df,
            rank=DEFAULT_ALS_RANK,
            reg_param=DEFAULT_ALS_REG_PARAM,
            max_iter=DEFAULT_ALS_MAX_ITER,
        )

        predictions = model.transform(test_df)
        rmse, mae = evaluate_regression(predictions)
        logger.info("Baseline Evaluation -> RMSE: %.4f, MAE: %.4f", rmse, mae)

        hdfs_mkdir(spark, paths.model_baseline)
        save_spark_model(model, f"{paths.model_baseline}/spark_model")

        metrics: Dict[str, Any] = {
            "dataset_size": dataset_size,
            "train_count": train_count,
            "test_count": test_count,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "train_time_sec": round(train_time, 2),
            "rank": DEFAULT_ALS_RANK,
            "reg_param": DEFAULT_ALS_REG_PARAM,
            "max_iter": DEFAULT_ALS_MAX_ITER,
            "total_execution_time_sec": round(time.time() - total_start, 2),
        }
        metrics_json = json.dumps(metrics, indent=2)

        hdfs_write_text(spark, f"{paths.model_baseline}/metrics.json", metrics_json)
        with open(RESULTS_DIR / f"metrics_{dataset_size}.json", "w", encoding="utf-8") as f:
            f.write(metrics_json)

        logger.info("Baseline pipeline for %s completed in %.1fs", dataset_size, time.time() - total_start)
        return metrics

    finally:
        spark.stop()


def run_tuning_suite(dataset_size: str = "25m") -> Dict[str, Any]:
    """Execute hyperparameter grid search, train optimal model, and compute ranking metrics."""
    paths = get_hdfs_paths(dataset_size)
    total_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Executing MovieLens %s Tuning & Evaluation Suite", dataset_size.upper())
    check_system_resources()

    spark = get_spark_session(
        app_name=f"MovieLens-{dataset_size}-Tuning-Evaluation",
        shuffle_partitions=80,
        checkpoint_dir=f"{paths.checkpoints}_tuning",
    )

    try:
        raw_ratings = load_ratings(spark, dataset_size=dataset_size)
        cleaned_ratings = preprocess_ratings(
            spark=spark,
            ratings_df=raw_ratings,
            dataset_size=dataset_size,
            num_partitions=16,
        )

        train_df, test_df, _, train_count, test_count = split_train_test_by_time(
            ratings_df=cleaned_ratings,
            train_ratio=DEFAULT_TRAIN_RATIO,
        )
        train_df.cache()
        test_df.cache()

        csv_tuning_file = str(RESULTS_DIR / f"tuning_results_{dataset_size}.csv")
        best_rank, best_reg, _ = run_grid_tuning(
            train_df=train_df,
            test_df=test_df,
            ranks=[10, 20, 50],
            reg_params=[0.01, 0.1, 0.5],
            max_iter=10,
            csv_log_path=csv_tuning_file,
        )

        final_max_iter = 15
        final_model, train_time = train_als(
            train_df=train_df,
            rank=best_rank,
            reg_param=best_reg,
            max_iter=final_max_iter,
        )

        final_preds = final_model.transform(test_df)
        final_rmse, final_mae = evaluate_regression(final_preds)
        logger.info("Final Tuned Model -> RMSE: %.4f, MAE: %.4f", final_rmse, final_mae)

        hdfs_mkdir(spark, paths.model_final)
        save_spark_model(final_model, f"{paths.model_final}/spark_model")

        pop_rmse, pop_mae, top_popular_movies = evaluate_popularity_baseline(
            train_df=train_df,
            test_df=test_df,
            min_ratings=20,
            top_k=20,
        )

        ranking_results = evaluate_ranking_metrics(
            model=final_model,
            train_df=train_df,
            test_df=test_df,
            k_list=[5, 10, 20],
            min_rating_thresh=4.0,
        )

        summary: Dict[str, Any] = {
            "dataset_size": dataset_size,
            "best_hyperparameters": {
                "rank": best_rank,
                "regParam": best_reg,
                "maxIter": final_max_iter,
            },
            "models_comparison": {
                "popularity_baseline": {"rmse": round(pop_rmse, 4), "mae": round(pop_mae, 4)},
                "als_final": {"rmse": round(final_rmse, 4), "mae": round(final_mae, 4)},
            },
            "ranking_metrics": ranking_results,
            "total_execution_time_sec": round(time.time() - total_start, 2),
        }
        summary_json = json.dumps(summary, indent=2)

        with open(RESULTS_DIR / f"evaluation_summary_{dataset_size}.json", "w", encoding="utf-8") as f:
            f.write(summary_json)

        final_metrics: Dict[str, Any] = {
            "dataset_size": dataset_size,
            "train_count": train_count,
            "test_count": test_count,
            "best_rank": best_rank,
            "best_reg_param": best_reg,
            "final_max_iter": final_max_iter,
            "final_rmse": round(final_rmse, 6),
            "final_mae": round(final_mae, 6),
            "train_time_sec": round(train_time, 2),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(RESULTS_DIR / f"metrics_{dataset_size}_final.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        hdfs_write_text(spark, f"{paths.model_final}/evaluation_summary_{dataset_size}.json", summary_json)
        hdfs_write_text(spark, f"{paths.model_final}/metrics_{dataset_size}_final.json", json.dumps(final_metrics, indent=2))

        logger.info("Tuning & evaluation suite finished in %.1f minutes", (time.time() - total_start) / 60)
        return summary

    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens Spark ALS Recommendation Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["100k", "1m", "10m", "25m"],
        default="100k",
        help="Target MovieLens dataset size (default: 100k)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter grid search and comprehensive evaluation suite",
    )

    args = parser.parse_args()
    if args.tune:
        run_tuning_suite(dataset_size=args.dataset)
    else:
        run_baseline_pipeline(dataset_size=args.dataset)


if __name__ == "__main__":
    main()
