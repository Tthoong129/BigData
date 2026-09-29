#!/usr/bin/env bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERR]${NC}  $1"; }

# 1. Prepare local directories
log_info "Initializing volumes and results directories..."
mkdir -p volumes/namenode volumes/datanode volumes/spark-logs results
log_success "Local directories ready."

# 2. Start Docker containers
log_info "Starting Docker Compose services..."
docker compose up -d

log_info "Waiting for cluster services to be healthy (90s)..."
sleep 90

# 3. Healthcheck NameNode
log_info "Checking HDFS NameNode Web UI..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:9870 | grep -q "200"; then
    log_success "HDFS NameNode UI ready at http://localhost:9870"
else
    log_warn "HDFS NameNode UI not responding yet. Retrying in 30s..."
    sleep 30
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:9870 | grep -q "200"; then
        log_success "HDFS NameNode UI ready at http://localhost:9870"
    else
        log_error "HDFS NameNode UI failed to start. Check logs: docker logs hadoop-namenode"
    fi
fi

# 4. Healthcheck Spark Master
log_info "Checking Spark Master Web UI..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q "200"; then
    log_success "Spark Master UI ready at http://localhost:8080"
else
    log_warn "Spark Master UI not responding. Check logs: docker logs spark-master"
fi

# 5. Initialize HDFS project directories
log_info "Creating project HDFS directories..."
docker exec hadoop-namenode bash -c "
    hdfs dfs -mkdir -p /movielens/100k/raw /movielens/100k/processed /movielens/100k/model && \
    hdfs dfs -mkdir -p /movielens/1m/raw /movielens/10m/raw /movielens/25m/raw && \
    hdfs dfs -chmod -R 777 /movielens
" && log_success "HDFS directories created." || log_warn "HDFS directory setup failed. Verify cluster health."

# 6. Verify HDFS status
docker exec hadoop-namenode bash -c "hdfs dfs -ls /" > /dev/null 2>&1 \
    && log_success "HDFS filesystem operational." \
    || log_error "HDFS filesystem error. Check logs: docker logs hadoop-namenode"

# 7. Test PySpark & HDFS integration
log_info "Verifying Spark-HDFS integration..."
docker exec spark-master bash -c "
python3 - <<'EOF'
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName('ClusterHealthCheck') \
    .master('spark://spark-master:7077') \
    .config('spark.hadoop.fs.defaultFS', 'hdfs://namenode:9000') \
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
test_df = spark.createDataFrame([(1, 'ping')], ['id', 'status'])
test_df.write.mode('overwrite').parquet('hdfs://namenode:9000/movielens/healthcheck')
read_df = spark.read.parquet('hdfs://namenode:9000/movielens/healthcheck')
assert read_df.count() == 1
spark.stop()
EOF
" && log_success "Spark ↔ HDFS integration verified." || log_warn "Integration test failed."

echo ""
echo "=================================================="
echo "  Setup Completed Successfully!"
echo "  - HDFS UI:       http://localhost:9870"
echo "  - Spark Master:  http://localhost:8080"
echo "  - Spark Worker:  http://localhost:8081"
echo ""
echo "  Run pipelines:"
echo "    docker exec spark-master python3 /opt/project/pipelines/run_100k.py"
echo "=================================================="
