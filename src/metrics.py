import logging
from typing import Any, Dict, List, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, Window, functions as F

logger = logging.getLogger(__name__)


def evaluate_regression(
    predictions: DataFrame,
    label_col: str = "rating",
    pred_col: str = "prediction",
) -> Tuple[float, float]:
    """Compute RMSE and MAE on non-null, non-NaN predictions."""
    valid = predictions.filter(F.col(pred_col).isNotNull() & ~F.isnan(pred_col))
    rmse = float(RegressionEvaluator(metricName="rmse", labelCol=label_col, predictionCol=pred_col).evaluate(valid))
    mae = float(RegressionEvaluator(metricName="mae", labelCol=label_col, predictionCol=pred_col).evaluate(valid))
    return rmse, mae


def evaluate_popularity_baseline(
    train_df: DataFrame,
    test_df: DataFrame,
    min_ratings: int = 20,
    top_k: int = 20,
) -> Tuple[float, float, List[int]]:
    """
    Evaluate popularity-based baseline:
    1. Predict test rating using item's training mean rating (or global average if unrated).
    2. Identify Top-K globally popular movies for reference.
    """
    movie_stats = (
        train_df
        .groupBy("movieId")
        .agg(
            F.avg("rating").alias("movie_avg_rating"),
            F.count("rating").alias("rating_count"),
        )
        .filter(F.col("rating_count") >= min_ratings)
        .cache()
    )

    global_avg: float = float(train_df.select(F.avg("rating")).collect()[0][0])
    logger.info("Global training rating mean: %.4f", global_avg)

    pop_pred_df = (
        test_df
        .join(movie_stats, on="movieId", how="left")
        .withColumn("prediction", F.coalesce(F.col("movie_avg_rating"), F.lit(global_avg)))
    )

    pop_rmse, pop_mae = evaluate_regression(pop_pred_df)
    logger.info("Popularity Baseline -> RMSE: %.4f, MAE: %.4f", pop_rmse, pop_mae)

    top_popular_movies: List[int] = [
        int(row.movieId)
        for row in (
            movie_stats
            .orderBy(F.col("movie_avg_rating").desc(), F.col("rating_count").desc())
            .limit(top_k)
            .collect()
        )
    ]

    return pop_rmse, pop_mae, top_popular_movies


def evaluate_ranking_metrics(
    model: Any,
    train_df: DataFrame,
    test_df: DataFrame,
    k_list: List[int] = [5, 10, 20],
    min_rating_thresh: float = 4.0,
) -> List[Dict[str, float]]:
    """
    Compute Precision@K and Recall@K via Test Candidate Ranking:
    - Evaluates on overlap users (users with history in training set).
    - Ranks candidate items in the test set using predicted ratings from ALS vs Popularity baseline.
    - Compares top-K ranked items against ground truth relevant items (rating >= min_rating_thresh).
    """
    train_users = train_df.select("userId").distinct()
    eval_test_df = test_df.join(train_users, on="userId", how="inner").cache()

    actual_df = (
        eval_test_df
        .filter(F.col("rating") >= min_rating_thresh)
        .groupBy("userId")
        .agg(
            F.collect_set("movieId").alias("actual_items"),
            F.count("movieId").alias("num_actual"),
        )
        .filter(F.col("num_actual") >= 1)
        .cache()
    )

    eval_users = actual_df.count()
    logger.info("Ranking Evaluation: %s overlap test users with relevant items (rating >= %.1f)", f"{eval_users:,}", min_rating_thresh)

    # 1. ALS Candidate Ranking
    als_preds = model.transform(eval_test_df).filter(
        F.col("prediction").isNotNull() & ~F.isnan("prediction")
    )
    w_als = Window.partitionBy("userId").orderBy(F.col("prediction").desc())
    max_k = max(k_list)
    ranked_als = (
        als_preds
        .withColumn("rank", F.row_number().over(w_als))
        .filter(F.col("rank") <= max_k)
        .groupBy("userId")
        .agg(F.collect_list("movieId").alias("rec_items"))
    )
    als_eval_df = actual_df.join(ranked_als, on="userId", how="inner").cache()

    # 2. Popularity Baseline Candidate Ranking
    movie_stats = (
        train_df
        .groupBy("movieId")
        .agg(
            F.avg("rating").alias("movie_avg_rating"),
            F.count("rating").alias("rating_count"),
        )
        .filter(F.col("rating_count") >= 20)
    )
    global_avg: float = float(train_df.select(F.avg("rating")).collect()[0][0])
    pop_preds = (
        eval_test_df
        .join(movie_stats, on="movieId", how="left")
        .withColumn("prediction", F.coalesce(F.col("movie_avg_rating"), F.lit(global_avg)))
    )
    w_pop = Window.partitionBy("userId").orderBy(F.col("prediction").desc())
    ranked_pop = (
        pop_preds
        .withColumn("rank", F.row_number().over(w_pop))
        .filter(F.col("rank") <= max_k)
        .groupBy("userId")
        .agg(F.collect_list("movieId").alias("rec_items"))
    )
    pop_eval_df = actual_df.join(ranked_pop, on="userId", how="inner").cache()

    results: List[Dict[str, float]] = []
    for k in k_list:
        als_stat = (
            als_eval_df
            .withColumn("k_recs", F.slice("rec_items", 1, k))
            .withColumn("hits", F.size(F.array_intersect("k_recs", "actual_items")))
            .withColumn("k_actual_size", F.size("k_recs"))
            .withColumn("prec", F.col("hits") / F.col("k_actual_size"))
            .withColumn("rec", F.col("hits") / F.col("num_actual"))
            .agg(F.avg("prec").alias("p"), F.avg("rec").alias("r"))
            .collect()[0]
        )

        pop_stat = (
            pop_eval_df
            .withColumn("k_recs", F.slice("rec_items", 1, k))
            .withColumn("hits", F.size(F.array_intersect("k_recs", "actual_items")))
            .withColumn("k_actual_size", F.size("k_recs"))
            .withColumn("prec", F.col("hits") / F.col("k_actual_size"))
            .withColumn("rec", F.col("hits") / F.col("num_actual"))
            .agg(F.avg("prec").alias("p"), F.avg("rec").alias("r"))
            .collect()[0]
        )

        results.append({
            "k": k,
            "als_prec": float(als_stat["p"]),
            "als_recall": float(als_stat["r"]),
            "pop_prec": float(pop_stat["p"]),
            "pop_recall": float(pop_stat["r"]),
        })

    return results
