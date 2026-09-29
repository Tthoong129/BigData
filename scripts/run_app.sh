#!/usr/bin/env bash
set -e

echo "======================================================="
echo "  KHOI DONG HE THONG GOI Y MOVIELENS 25M (SPARK ALS)"
echo "======================================================="
echo ""

if ! docker ps | grep -q spark-master; then
    echo "[WARN] Container spark-master chua chay. Dang khoi dong docker compose..."
    docker compose up -d
    sleep 5
fi

echo "[OK]   Cluster Spark & HDFS da san sang."
echo "[INFO] Dang khoi chay Aether Cinema Web App tren cong 3000..."
echo "[INFO] Truy cap trinh duyet tai: http://localhost:3000"
echo ""

python web_server.py
