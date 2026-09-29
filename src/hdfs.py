import logging
from typing import Any, List

from pyspark.sql import SparkSession

from src.config import HDFS_BASE

logger = logging.getLogger(__name__)


def get_hdfs_fs(spark: SparkSession, base_uri: str = HDFS_BASE) -> Any:
    """Retrieve Hadoop FileSystem instance via Spark JVM bridge."""
    sc = spark.sparkContext
    uri = sc._jvm.java.net.URI(base_uri)
    return sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, sc._jsc.hadoopConfiguration())


def hdfs_path(spark: SparkSession, path_str: str) -> Any:
    """Convert string to Hadoop Path object."""
    return spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path_str)


def hdfs_exists(spark: SparkSession, path: str) -> bool:
    """Check if a path exists on HDFS."""
    return bool(get_hdfs_fs(spark).exists(hdfs_path(spark, path)))


def hdfs_mkdir(spark: SparkSession, path: str) -> None:
    """Create a directory on HDFS if it does not exist."""
    get_hdfs_fs(spark).mkdirs(hdfs_path(spark, path))
    logger.debug("HDFS mkdir: %s", path)


def hdfs_put(spark: SparkSession, local_path: str, remote_path: str) -> None:
    """Upload a local file to HDFS."""
    fs = get_hdfs_fs(spark)
    path_cls = spark.sparkContext._jvm.org.apache.hadoop.fs.Path
    fs.copyFromLocalFile(False, True, path_cls(f"file://{local_path}"), path_cls(remote_path))
    logger.info("HDFS put: %s -> %s", local_path, remote_path)


def hdfs_write_text(spark: SparkSession, remote_path: str, content: str) -> None:
    """Write text directly to a file on HDFS with overwrite enabled."""
    fs = get_hdfs_fs(spark)
    out_stream = fs.create(hdfs_path(spark, remote_path), True)
    out_stream.write(bytearray(content.encode("utf-8")))
    out_stream.close()
    logger.info("HDFS saved text: %s", remote_path)


def hdfs_ls(spark: SparkSession, path: str) -> List[str]:
    """List directory contents on HDFS."""
    fs = get_hdfs_fs(spark)
    statuses = fs.listStatus(hdfs_path(spark, path))
    return [str(s.getPath()) for s in statuses]


def hdfs_dir_size_bytes(spark: SparkSession, path: str) -> int:
    """Get total directory size in bytes."""
    try:
        return int(get_hdfs_fs(spark).getContentSummary(hdfs_path(spark, path)).getLength())
    except Exception:
        return 0
