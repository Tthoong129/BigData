import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOCAL_TEMP_DIR: Path = Path("/tmp/movielens_staging")

HDFS_BASE: str = os.environ.get("HDFS_BASE", "hdfs://namenode:9000")
SPARK_MASTER_URL: str = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")

DATASET_URLS: Dict[str, str] = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

DEFAULT_TRAIN_RATIO: float = 0.8
DEFAULT_ALS_RANK: int = 10
DEFAULT_ALS_REG_PARAM: float = 0.1
DEFAULT_ALS_MAX_ITER: int = 10
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class HDFSPaths:
    raw: str
    processed: str
    model_baseline: str
    model_final: str
    checkpoints: str


def get_hdfs_paths(dataset_size: str) -> HDFSPaths:
    base = f"{HDFS_BASE}/movielens/{dataset_size}"
    return HDFSPaths(
        raw=f"{base}/raw",
        processed=f"{base}/processed",
        model_baseline=f"{base}/model/als_baseline",
        model_final=f"{base}/model/als_final",
        checkpoints=f"{base}/checkpoints",
    )
