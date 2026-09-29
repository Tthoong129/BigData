import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.run_pipeline import run_baseline_pipeline

if __name__ == "__main__":
    run_baseline_pipeline(dataset_size="100k")
