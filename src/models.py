import csv
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame

from src.metrics import evaluate_regression

logger = logging.getLogger(__name__)


def build_als(
    rank: int = 10,
    reg_param: float = 0.1,
    max_iter: int = 10,
    seed: int = 42,
    checkpoint_interval: int = 2,
    num_user_blocks: int = 10,
    num_item_blocks: int = 10,
) -> ALS:
    """Build configured ALS estimator."""
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        coldStartStrategy="drop",
        nonnegative=True,
        seed=seed,
        checkpointInterval=checkpoint_interval,
        numUserBlocks=num_user_blocks,
        numItemBlocks=num_item_blocks,
    )


def train_als(
    train_df: DataFrame,
    rank: int = 10,
    reg_param: float = 0.1,
    max_iter: int = 10,
    seed: int = 42,
    checkpoint_interval: int = 2,
) -> Tuple[ALSModel, float]:
    """Train ALS recommendation model on given training DataFrame."""
    als = build_als(
        rank=rank,
        reg_param=reg_param,
        max_iter=max_iter,
        seed=seed,
        checkpoint_interval=checkpoint_interval,
    )
    t0 = time.time()
    model: ALSModel = als.fit(train_df)
    train_time = time.time() - t0
    logger.info("ALS model trained in %.1fs (rank=%d, regParam=%.2f, maxIter=%d)", train_time, rank, reg_param, max_iter)
    return model, train_time


def save_spark_model(model: ALSModel, hdfs_model_path: str) -> None:
    """Persist ALS model to HDFS."""
    model.write().overwrite().save(hdfs_model_path)
    logger.info("Persisted ALS model to HDFS: %s", hdfs_model_path)


def run_grid_tuning(
    train_df: DataFrame,
    val_df: Optional[DataFrame] = None,
    ranks: Optional[List[int]] = None,
    reg_params: Optional[List[float]] = None,
    max_iter: int = 10,
    csv_log_path: Optional[str] = None,
    test_df: Optional[DataFrame] = None,  # Backward-compatible alias
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """Perform grid search tuning across rank and regParam parameters evaluated on Validation Set."""
    eval_df = val_df if val_df is not None else test_df
    if eval_df is None:
        raise ValueError("Either val_df or test_df must be provided for grid tuning evaluation.")

    if ranks is None:
        ranks = [10, 20, 50]
    if reg_params is None:
        reg_params = [0.01, 0.1, 0.5]

    combinations: List[Tuple[int, float]] = [(r, reg) for r in ranks for reg in reg_params]
    results: List[Dict[str, Any]] = []

    if csv_log_path:
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "regParam", "val_rmse", "val_mae", "train_time_sec"])

    for idx, (r, reg) in enumerate(combinations, start=1):
        logger.info("[%d/%d] Grid trial: rank=%d, regParam=%.2f, maxIter=%d", idx, len(combinations), r, reg, max_iter)
        model, t_train = train_als(train_df, rank=r, reg_param=reg, max_iter=max_iter)
        preds = model.transform(eval_df)
        rmse, mae = evaluate_regression(preds)

        record: Dict[str, Any] = {
            "rank": r,
            "regParam": reg,
            "val_rmse": round(rmse, 6),
            "val_mae": round(mae, 6),
            "train_time_sec": round(t_train, 2),
        }
        results.append(record)

        if csv_log_path:
            with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([r, reg, record["val_rmse"], record["val_mae"], record["train_time_sec"]])

        logger.info("[%d/%d] -> Val RMSE: %.4f, Val MAE: %.4f (took %.1fs)", idx, len(combinations), rmse, mae, t_train)

    best_config = min(results, key=lambda x: x["val_rmse"])
    logger.info("Best Grid Configuration: rank=%d, regParam=%.2f (Val RMSE=%.4f)", best_config["rank"], best_config["regParam"], best_config["val_rmse"])

    return int(best_config["rank"]), float(best_config["regParam"]), results

