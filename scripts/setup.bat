@echo off
setlocal

echo [INFO] Tao thu muc volumes va results...
if not exist "volumes\namenode" mkdir volumes\namenode
if not exist "volumes\datanode" mkdir volumes\datanode
if not exist "volumes\spark-logs" mkdir volumes\spark-logs
if not exist "results" mkdir results
echo [OK]   Thu muc volumes va results da san sang.

echo [INFO] Khoi dong Docker Compose...
docker compose up -d

echo [INFO] Doi cluster khoi dong (90s)...
timeout /t 90 /nobreak > nul

echo [INFO] Kiem tra HDFS NameNode...
curl -s -o nul http://localhost:9870
if %errorlevel% == 0 (
    echo [OK]   HDFS NameNode UI: http://localhost:9870
) else (
    echo [WARN] HDFS chua san sang. Thu lai sau 30 giay...
    timeout /t 30 /nobreak > nul
)

echo [INFO] Khoi tao thu muc HDFS...
docker exec hadoop-namenode bash -c "hdfs dfs -mkdir -p /movielens/100k/raw /movielens/100k/processed /movielens/100k/model /movielens/1m/raw /movielens/10m/raw /movielens/25m/raw && hdfs dfs -chmod -R 777 /movielens"
if %errorlevel% == 0 (
    echo [OK]   Thu muc HDFS da duoc tao.
) else (
    echo [WARN] Khong the khoi tao thu muc HDFS.
)

echo.
echo ==================================================
echo   Setup hoan thanh!
echo   - HDFS UI:       http://localhost:9870
echo   - Spark Master:  http://localhost:8080
echo   - Spark Worker:  http://localhost:8081
echo.
echo   Chay pipeline 100k:
echo     docker exec spark-master python3 /opt/project/pipelines/run_100k.py
echo ==================================================
pause
