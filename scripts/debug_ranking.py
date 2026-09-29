import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F

from src.config import get_hdfs_paths
from src.data import load_movies, load_ratings, split_train_test_by_time
from src.metrics import evaluate_popularity_baseline, evaluate_ranking_metrics
from src.spark import get_spark_session


def main():
    paths = get_hdfs_paths("25m")
    spark = get_spark_session(app_name="DebugRankingMetrics25M", shuffle_partitions=80)

    try:
        print("\n" + "=" * 70)
        print("BẮT ĐẦU DEBUG CHI TIẾT RANKING METRICS TRÊN MOVIELENS 25M")
        print("=" * 70)

        # 0. Load Data & Model
        ratings_df = spark.read.parquet(f"{paths.processed}/ratings")
        train_df, test_df, _, train_cnt, test_cnt = split_train_test_by_time(ratings_df, train_ratio=0.8)
        movies_df = load_movies(spark, "25m")

        final_model_path = f"{paths.model_final}/spark_model"
        print(f"\n[INFO] Loading final ALS model from {final_model_path}...")
        model = ALSModel.load(final_model_path)

        # ---------------------------------------------------------------------
        # 1. SAMPLE OUTPUT CỦA recommendForUserSubset CHO 3-5 USERS
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("1. SAMPLE KẾT QUẢ recommendForUserSubset CHO 3 USERS CỤ THỂ:")
        print("-" * 70)

        # Lấy 3 user có đánh giá trong test set
        sample_users = (
            test_df.filter(F.col("rating") >= 4.0)
            .groupBy("userId")
            .agg(F.count("movieId").alias("cnt"))
            .filter(F.col("cnt") >= 5)
            .select("userId")
            .limit(3)
        )

        recs_df = model.recommendForUserSubset(sample_users, 5)
        recs_df.show(truncate=False)

        # ---------------------------------------------------------------------
        # 2. KIỂM TRA DATA TYPE (DTYPE) Ở CẢ HAI BÊN
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("2. KIỂM TRA KIỂU DỮ LIỆU (DTYPE) GIỮA RECOMMENDATIONS VÀ TEST SET:")
        print("-" * 70)
        print("Schema của Test Set:")
        test_df.printSchema()

        print("\nSchema của Output ALS recommendForUserSubset:")
        recs_df.printSchema()

        # ---------------------------------------------------------------------
        # 3. KIỂM TRA BƯỚC UNPACK / EXPLODE
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("3. KIỂM TRA BƯỚC EXPLODE / TRANSFORM recommendations:")
        print("-" * 70)
        exploded_df = (
            recs_df
            .select("userId", F.explode("recommendations").alias("rec"))
            .select(
                "userId",
                F.col("rec.movieId").alias("rec_movieId"),
                F.col("rec.rating").alias("rec_predicted_rating")
            )
        )
        exploded_df.show(15, truncate=False)

        # ---------------------------------------------------------------------
        # 4. KIỂM TRA THỦ CÔNG 1 USER CỤ THỂ
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("4. KIỂM TRA THỦ CÔNG 1 USER CỤ THỂ:")
        print("-" * 70)
        target_uid = sample_users.first()["userId"]
        print(f"Target User ID: {target_uid}")

        # Relevant movies trong test
        user_test_relevant = (
            test_df
            .filter((F.col("userId") == target_uid) & (F.col("rating") >= 4.0))
            .join(movies_df, on="movieId", how="left")
            .select("movieId", "title", "rating")
        )
        print(f"\n[Ground Truth] Các phim User {target_uid} đánh giá >= 4.0 trong tập Test:")
        user_test_relevant.show(10, truncate=False)

        # Top-5 ALS gợi ý từ toàn bộ catalog
        user_als_recs = (
            exploded_df
            .filter(F.col("userId") == target_uid)
            .join(movies_df, F.col("rec_movieId") == F.col("movieId"), how="left")
            .select("rec_movieId", "title", "rec_predicted_rating")
        )
        print(f"\n[ALS Recs] Top 5 phim ALS gợi ý trên TOÀN CATALOG 62K phim:")
        user_als_recs.show(truncate=False)

        # ---------------------------------------------------------------------
        # 5. CHẠY LẠI ĐÁNH GIÁ RANKING (TEST CANDIDATE RANKING)
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("5. CHẠY LẠI ĐÁNH GIÁ PRECISION@K VÀ RECALL@K:")
        print("-" * 70)
        ranking_metrics = evaluate_ranking_metrics(
            model=model,
            train_df=train_df,
            test_df=test_df,
            k_list=[5, 10, 20],
            min_rating_thresh=4.0
        )

        print("\n" + "=" * 70)
        print("BẢNG KẾT QUẢ ĐÁNH GIÁ RANKING MỚI (ALS vs Popularity):")
        print(f"{'Model':<15}{'K':<6}{'Precision@K':<18}{'Recall@K':<18}")
        print("-" * 70)
        for r in ranking_metrics:
            print(f"{'Popularity':<15}{r['k']:<6}{r['pop_prec']*100:<17.2f}%{r['pop_recall']*100:<17.2f}%")
        print("-" * 70)
        for r in ranking_metrics:
            print(f"{'ALS (Final)':<15}{r['k']:<6}{r['als_prec']*100:<17.2f}%{r['als_recall']*100:<17.2f}%")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
