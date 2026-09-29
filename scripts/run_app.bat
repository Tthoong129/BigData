@echo off
setlocal

echo =======================================================
echo   KHOI DONG HE THONG GOI Y MOVIELENS 25M (SPARK ALS)
echo =======================================================
echo.

echo [INFO] Kiem tra ket noi container Spark & Hadoop...
docker ps | findstr /i spark-master > nul
if %errorlevel% neq 0 (
    echo [WARN] Container spark-master chua chay. Dang khoi dong docker compose...
    docker compose up -d
    timeout /t 5 /nobreak > nul
)

echo [OK]   Cluster Spark & HDFS da san sang.
echo [INFO] Dang khoi chay Aether Cinema Web App tren cong 3000...
echo [INFO] Truy cap trinh duyet tai: http://localhost:3000
echo.

start http://localhost:3000
python web_server.py
