import logging
import time
import urllib.request
import zipfile
from datetime import datetime
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.config import DATASET_URLS, LOCAL_TEMP_DIR, get_hdfs_paths
from src.hdfs import hdfs_dir_size_bytes, hdfs_exists, hdfs_mkdir, hdfs_put

logger = logging.getLogger(__name__)

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("movieId", IntegerType(), nullable=False),
    StructField("rating", FloatType(), nullable=False),
    StructField("timestamp", LongType(), nullable=False),
])

MOVIES_25M_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("title", StringType(), nullable=True),
    StructField("genres", StringType(), nullable=True),
])

MOVIES_100K_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("release_date", StringType(), True),
    StructField("video_release", StringType(), True),
    StructField("imdb_url", StringType(), True),
    *[StructField(f"genre_{i}", StringType(), True) for i in range(19)],
])


def ingest_dataset_to_hdfs(spark: SparkSession, dataset_size: str = "100k") -> None:
    """Download MovieLens dataset and upload raw files to HDFS."""
    paths = get_hdfs_paths(dataset_size)
    url = DATASET_URLS.get(dataset_size)

    if not url:
        raise ValueError(f"Unsupported dataset size: {dataset_size}")

    marker_file = "u.data" if dataset_size == "100k" else "ratings.dat" if dataset_size == "10m" else "ratings.csv"
    if hdfs_exists(spark, f"{paths.raw}/{marker_file}"):
        logger.info("Dataset '%s' already exists in HDFS: %s", dataset_size, paths.raw)
        return

    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_target = LOCAL_TEMP_DIR / f"ml-{dataset_size}.zip"

    logger.info("Downloading dataset from %s...", url)
    t0 = time.time()
    urllib.request.urlretrieve(url, str(zip_target))
    logger.info("Downloaded %s in %.1fs", zip_target.name, time.time() - t0)

    with zipfile.ZipFile(str(zip_target), "r") as archive:
        archive.extractall(str(LOCAL_TEMP_DIR))

    extract_folder = LOCAL_TEMP_DIR / f"ml-{dataset_size}"
    if not extract_folder.exists():
        for candidate in LOCAL_TEMP_DIR.iterdir():
            if candidate.is_dir() and (candidate.name.lower().startswith("ml-") or candidate.name.lower().startswith("ml_")):
                extract_folder = candidate
                break

    hdfs_mkdir(spark, paths.raw)

    target_files = (
        ["u.data", "u.item", "u.user", "u.genre", "u.info"]
        if dataset_size == "100k"
        else ["ratings.dat", "movies.dat", "tags.dat"]
        if dataset_size == "10m"
        else ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]
    )

    for filename in target_files:
        src = extract_folder / filename
        if src.exists():
            hdfs_put(spark, str(src), f"{paths.raw}/{filename}")
            logger.info("Uploaded %s -> %s/%s", filename, paths.raw, filename)

    if zip_target.exists():
        zip_target.unlink()


def load_ratings(spark: SparkSession, dataset_size: str = "100k") -> DataFrame:
    """Load ratings CSV/TSV/DAT from HDFS into DataFrame."""
    paths = get_hdfs_paths(dataset_size)
    if dataset_size == "100k":
        path = f"{paths.raw}/u.data"
        df = spark.read.option("sep", "\t").option("header", "false").schema(RATINGS_SCHEMA).csv(path)
    elif dataset_size == "10m":
        path = f"{paths.raw}/ratings.dat"
        df = spark.read.option("sep", "::").option("header", "false").schema(RATINGS_SCHEMA).csv(path)
    else:
        path = f"{paths.raw}/ratings.csv"
        df = spark.read.option("sep", ",").option("header", "true").schema(RATINGS_SCHEMA).csv(path)

    logger.info("Loaded raw ratings (%s): %s rows", dataset_size, f"{df.count():,}")
    return df


def load_movies(spark: SparkSession, dataset_size: str = "100k") -> DataFrame:
    """Load movie metadata from HDFS."""
    paths = get_hdfs_paths(dataset_size)
    if dataset_size == "100k":
        df = spark.read.option("sep", "|").option("header", "false").schema(MOVIES_100K_SCHEMA).csv(f"{paths.raw}/u.item")
        df = df.select("movieId", "title", "release_date")
    elif dataset_size == "10m":
        df = spark.read.option("sep", "::").option("header", "false").schema(MOVIES_25M_SCHEMA).csv(f"{paths.raw}/movies.dat")
    else:
        df = spark.read.option("sep", ",").option("header", "true").schema(MOVIES_25M_SCHEMA).csv(f"{paths.raw}/movies.csv")

    logger.info("Loaded movies metadata (%s): %s rows", dataset_size, f"{df.count():,}")
    return df


def preprocess_ratings(
    spark: SparkSession,
    ratings_df: DataFrame,
    dataset_size: str = "100k",
    num_partitions: int = 16,
) -> DataFrame:
    """Clean data (remove nulls, deduplicate, filter rating range [0.5, 5.0]) and save as Parquet."""
    raw_count = ratings_df.count()

    cleaned = (
        ratings_df
        .dropna(subset=["userId", "movieId", "rating", "timestamp"])
        .dropDuplicates(["userId", "movieId", "timestamp"])
        .filter(F.col("rating").between(0.5, 5.0))
    )

    out_path = f"{get_hdfs_paths(dataset_size).processed}/ratings"
    t0 = time.time()
    cleaned.repartition(num_partitions).write.mode("overwrite").parquet(out_path)
    logger.info("Preprocessed %s / %s records in %.1fs -> %s", f"{cleaned.count():,}", f"{raw_count:,}", time.time() - t0, out_path)

    raw_file = "u.data" if dataset_size == "100k" else "ratings.csv"
    raw_size_mb = hdfs_dir_size_bytes(spark, f"{get_hdfs_paths(dataset_size).raw}/{raw_file}") / (1024 * 1024)
    pq_size_mb = hdfs_dir_size_bytes(spark, out_path) / (1024 * 1024)
    if raw_size_mb > 0 and pq_size_mb > 0:
        logger.info("Storage Footprint: Raw CSV %.1fMB -> Processed Parquet %.1fMB", raw_size_mb, pq_size_mb)

    return spark.read.parquet(out_path)


def split_train_test_by_time(
    ratings_df: DataFrame,
    train_ratio: float = 0.8,
    relative_error: float = 0.005,
) -> Tuple[DataFrame, DataFrame, float, int, int]:
    """Split data temporally via approxQuantile into train and test sets."""
    threshold = float(ratings_df.approxQuantile("timestamp", [train_ratio], relative_error)[0])
    split_date = datetime.fromtimestamp(threshold).strftime("%Y-%m-%d %H:%M:%S")

    train_df = ratings_df.filter(F.col("timestamp") <= threshold)
    test_df = ratings_df.filter(F.col("timestamp") > threshold)

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count

    logger.info("Time Split @ %s (%s) -> Train: %s (%.1f%%) | Test: %s (%.1f%%)", threshold, split_date, f"{train_count:,}", train_count / total * 100, f"{test_count:,}", test_count / total * 100)

    return train_df, test_df, threshold, train_count, test_count
