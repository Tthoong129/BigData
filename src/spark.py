import logging
import os
import shutil
from typing import Optional

from pyspark.sql import SparkSession

from src.config import HDFS_BASE, SPARK_MASTER_URL

logger = logging.getLogger(__name__)


def get_spark_session(
    app_name: str = "MovieLens-Pipeline",
    master: str = SPARK_MASTER_URL,
    shuffle_partitions: int = 80,
    driver_max_result_size: str = "512m",
    memory_fraction: str = "0.8",
    checkpoint_dir: Optional[str] = None,
) -> SparkSession:
    """Build and return an optimized SparkSession for MovieLens workloads."""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.hadoop.fs.defaultFS", HDFS_BASE)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.maxResultSize", driver_max_result_size)
        .config("spark.memory.fraction", memory_fraction)
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if checkpoint_dir:
        spark.sparkContext.setCheckpointDir(checkpoint_dir)

    logger.info("SparkSession initialized: version=%s, master=%s", spark.version, spark.sparkContext.master)
    return spark


def check_system_resources(min_ram_gb: float = 2.5, min_disk_gb: float = 2.5) -> None:
    """Inspect and log host/container resource headroom."""
    total_ram_gb: float = 4.0
    avail_ram_gb: float = 3.0

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "MemTotal:" in line:
                        total_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    elif "MemAvailable:" in line:
                        avail_ram_gb = int(line.split()[1]) / (1024 * 1024)
        except (IOError, ValueError) as exc:
            logger.warning("Failed to parse /proc/meminfo: %s", exc)

    disk = shutil.disk_usage("/tmp")
    free_disk_gb = disk.free / (1024 ** 3)

    logger.info("Resource Check -> Available RAM: %.2fGB / %.2fGB | Free /tmp Disk: %.2fGB", avail_ram_gb, total_ram_gb, free_disk_gb)

    if avail_ram_gb < min_ram_gb:
        logger.warning("Available RAM (%.2fGB) is below threshold (%.2fGB)", avail_ram_gb, min_ram_gb)
    if free_disk_gb < min_disk_gb:
        logger.warning("Free disk space (%.2fGB) is below threshold (%.2fGB)", free_disk_gb, min_disk_gb)
