import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Đường dẫn dự án & Cấu hình môi trường
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
HDFS_BASE = os.environ.get("HDFS_BASE", "hdfs://namenode:9000")
SPARK_MASTER_URL = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
MODEL_HDFS_PATH = f"{HDFS_BASE}/movielens/25m/model/als_final/spark_model"
MOVIES_HDFS_PATH = f"{HDFS_BASE}/movielens/25m/raw/movies.csv"
RATINGS_HDFS_PATH = f"{HDFS_BASE}/movielens/25m/processed/ratings"

# Mapping từ điển thể loại phim sang tiếng Việt chuẩn mực
GENRE_TRANSLATIONS: Dict[str, str] = {
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children": "Thiếu nhi",
    "Comedy": "Hài hước",
    "Crime": "Tội phạm",
    "Documentary": "Tài liệu",
    "Drama": "Chính kịch",
    "Fantasy": "Kỳ ảo",
    "Film-Noir": "Phim đen (Noir)",
    "Horror": "Kinh dị",
    "IMAX": "IMAX",
    "Musical": "Ca nhạc",
    "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn",
    "Sci-Fi": "Khoa học viễn tưởng",
    "Thriller": "Kịch tính",
    "War": "Chiến tranh",
    "Western": "Viễn Tây",
    "(no genres listed)": "Chưa phân loại",
}

# -----------------------------------------------------------------------------
# 2. Cấu hình Trang Streamlit & Theme Power BI Hiện Đại (Indigo / Violet Luxury)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieLens 25M Enterprise Analytics | Power BI Style",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOOD_MAPPING = {
    "Tất cả tâm trạng (All)": None,
    "Giải trí / Hài hước (Cười sảng khoái)": ["Comedy", "Action", "Adventure"],
    "Sâu lắng / Cảm động (Suy ngẫm)": ["Drama", "Romance", "Documentary"],
    "Hồi hộp / Kịch tính (Kích thích)": ["Thriller", "Horror", "Mystery", "Crime"],
    "Phiêu lưu / Kỳ ảo (Thoát ly thực tại)": ["Fantasy", "Sci-Fi", "Animation"],
}

MOOD_MAPPING = {
    "Tất cả tâm trạng (All)": None,
    "Giải trí / Hài hước (Cười sảng khoái)": ["Comedy", "Action", "Adventure"],
    "Sâu lắng / Cảm động (Suy ngẫm)": ["Drama", "Romance", "Documentary"],
    "Hồi hộp / Kịch tính (Kích thích)": ["Thriller", "Horror", "Mystery", "Crime"],
    "Phiêu lưu / Kỳ ảo (Thoát ly thực tại)": ["Fantasy", "Sci-Fi", "Animation"],
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Segoe+UI:wght@400;600;700;800&display=swap');

    /* Typography hiện đại chuẩn Power BI Analytics */
    html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
    }

    /* Nền toàn trang: Màu Lavender Ice thanh lịch chuẩn Dashboard ngân hàng / tài chính */
    .stApp {
        background-color: #F0F2FA !important;
    }

    /* Chiếm trọn chiều ngang màn hình với padding hài hòa */
    .main .block-container {
        max-width: 100% !important;
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }

    /* Header Bar dạng thẻ trắng thanh lịch (Header style Bank Analysis) */
    .pbi-topbar {
        background: #ffffff;
        border-radius: 12px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #CBD1E9;
        box-shadow: 0 6px 20px rgba(94, 72, 232, 0.08);
    }

    .pbi-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .pbi-brand-logo {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #1E1B4B;
        font-weight: 800;
        font-size: 1.2rem;
        padding: 0.4rem 0.85rem;
        border-radius: 8px;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.2);
    }

    .pbi-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .pbi-brand-sub {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        margin: 0;
    }

    .pbi-topbar-pills {
        display: flex;
        gap: 0.5rem;
    }

    .pbi-pill {
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        background: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
    }

    .pbi-pill.active {
        background: #4F46E5;
        color: #1E1B4B;
        border: 1px solid #4F46E5;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
    }

    /* Bảng Slicers bên trái (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #CBD1E9 !important;
        padding-top: 1rem !important;
    }

    .slicer-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 0.85rem 1.05rem;
        margin-bottom: 0.85rem;
        border: 1px solid #CBD1E9;
        box-shadow: 0 4px 12px rgba(94, 72, 232, 0.04);
    }

    .slicer-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1E1B4B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    /* KPI Cards Chuẩn Bank Analysis (Bo góc 12px, số liệu tím nổi bật) */
    .pbi-kpi-card {
        background: #ffffff;
        border: 1px solid #CBD1E9;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 6px 16px rgba(94, 72, 232, 0.06);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .pbi-kpi-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    .pbi-kpi-main {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.3rem 0;
    }

    .pbi-kpi-val {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }

    .pbi-kpi-chart {
        display: flex;
        align-items: center;
    }

    .pbi-kpi-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .pbi-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        display: inline-block;
    }

    .pbi-badge.green {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    .pbi-badge.purple {
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #E0E7FF;
    }

    .pbi-kpi-subtext {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
    }

    /* Container Panel bao bọc nội dung / đồ thị */
    .pbi-panel {
        background: #ffffff;
        border: 1px solid #CBD1E9;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 6px 20px rgba(94, 72, 232, 0.08);
        margin-bottom: 1.25rem;
    }

    .pbi-panel-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0F172A;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.6rem;
    }

    /* Thanh Chuyển Tab Phong cách Nút Bấm Tím Doanh Nghiệp (Bank Analysis) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #E2E5F5 !important;
        padding: 5px !important;
        border-radius: 10px !important;
        margin-bottom: 1.1rem !important;
        border: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 22px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        color: #4B5563 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.7) !important;
        color: #5E48E8 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #5E48E8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(94, 72, 232, 0.35) !important;
    }

    /* Ẩn bớt thanh điều khiển thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def translate_genres_str(genres_raw: str) -> str:
    """Chuyển chuỗi thể loại từ 'Action|Sci-Fi' sang tiếng Việt chuẩn."""
    if not genres_raw or pd.isna(genres_raw):
        return "Chưa phân loại"
    tokens = str(genres_raw).split("|")
    translated = [GENRE_TRANSLATIONS.get(t.strip(), t.strip()) for t in tokens if t.strip()]
    return ", ".join(translated) if translated else "Chưa phân loại"


def generate_area_sparkline_svg(values: List[float], stroke: str = "#5E48E8", width: int = 86, height: int = 30) -> str:
    """Tạo biểu đồ sparkline diện tích (area curve) phủ gradient tím chuẩn phong cách Bank Analysis."""
    if not values or len(values) < 2:
        return ""
    min_v, max_v = min(values), max(values)
    span = max_v - min_v if max_v != min_v else 1.0
    step = width / (len(values) - 1)
    coords = []
    for i, v in enumerate(values):
        x = i * step
        y = height - 3 - ((v - min_v) / span) * (height - 6)
        coords.append((x, y))

    path_d = f"M {coords[0][0]:.1f},{coords[0][1]:.1f} " + " ".join([f"L {x:.1f},{y:.1f}" for x, y in coords[1:]])
    fill_d = f"{path_d} L {width:.1f},{height:.1f} L 0,{height:.1f} Z"
    grad_id = f"grad_{abs(hash(str(values))) % 100000}"

    return f"""<svg width="{width}" height="{height}" style="vertical-align: middle;">
        <defs>
            <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{stroke}" stop-opacity="0.45"/>
                <stop offset="100%" stop-color="{stroke}" stop-opacity="0.05"/>
            </linearGradient>
        </defs>
        <path d="{fill_d}" fill="url(#{grad_id})"/>
        <path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""


def generate_bar_sparkline_svg(values: List[float], color: str = "#5E48E8", width: int = 86, height: int = 30) -> str:
    """Tạo biểu đồ sparkline dạng cột đứng (mini bar sparkline) theo mẫu Bank Analysis."""
    if not values:
        return ""
    max_v = max(values) if max(values) > 0 else 1.0
    n = len(values)
    bar_w = max(3, int((width - (n - 1) * 3) / n))
    rects = []
    for i, v in enumerate(values):
        h = max(4, int((v / max_v) * (height - 4)))
        x = i * (bar_w + 3)
        y = height - h
        rects.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="1.5" fill="{color}" opacity="0.85"/>')

    return f"""<svg width="{width}" height="{height}" style="vertical-align: middle;">
        {"".join(rects)}
    </svg>"""


def render_kpi_card(label: str, val: str, spark_svg: str, badge_text: str, is_green: bool, sub_text: str) -> str:
    """Render thẻ KPI Power BI phong cách hiện đại Bank Analysis."""
    badge_cls = "green" if is_green else "purple"
    return f"""<div class="pbi-kpi-card">
<div class="pbi-kpi-label">{label}</div>
<div class="pbi-kpi-main">
<div class="pbi-kpi-val">{val}</div>
<div class="pbi-kpi-chart">{spark_svg}</div>
</div>
<div class="pbi-kpi-footer">
<span class="pbi-badge {badge_cls}">{badge_text}</span>
<span class="pbi-kpi-subtext">{sub_text}</span>
</div>
</div>"""


# -----------------------------------------------------------------------------
# 3. Khởi tạo & Cache Tài nguyên PySpark / Model / Dữ liệu tĩnh
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_shared_spark_session():
    """Khởi tạo SparkSession dùng chung tối ưu cho ứng dụng Streamlit."""
    try:
        from pyspark.sql import SparkSession

        builder = (
            SparkSession.builder.appName("MovieLens-Streamlit-PBI")
            .master(SPARK_MASTER_URL)
            .config("spark.hadoop.fs.defaultFS", HDFS_BASE)
            .config("spark.driver.memory", "1536m")
            .config("spark.driver.maxResultSize", "512m")
            .config("spark.sql.shuffle.partitions", "16")
        )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception:
        from pyspark.sql import SparkSession

        builder = (
            SparkSession.builder.appName("MovieLens-Streamlit-Local")
            .master("local[2]")
            .config("spark.hadoop.fs.defaultFS", HDFS_BASE)
        )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark


@st.cache_resource(show_spinner=False)
def get_cached_als_model(_spark):
    """Load model ALS đã train sẵn từ HDFS đúng 1 lần khi khởi động."""
    from pyspark.ml.recommendation import ALSModel

    return ALSModel.load(MODEL_HDFS_PATH)


@st.cache_data(show_spinner=False)
def get_cached_movies(_spark) -> pd.DataFrame:
    """Đọc movies.csv từ HDFS sang Pandas DataFrame và tạo cột tiếng Việt."""
    try:
        spark_df = _spark.read.option("header", "true").option("inferSchema", "true").csv(MOVIES_HDFS_PATH)
        pdf = spark_df.toPandas()
    except Exception:
        local_csv = PROJECT_ROOT / "movies.csv"
        if local_csv.exists():
            pdf = pd.read_csv(local_csv)
        else:
            raise RuntimeError(f"Không thể đọc file movies từ HDFS ({MOVIES_HDFS_PATH}) hoặc local.")

    pdf["movieId"] = pdf["movieId"].astype(int)
    pdf["genres"] = pdf["genres"].fillna("(no genres listed)")
    pdf["genres_vi"] = pdf["genres"].apply(translate_genres_str)
    pdf["year"] = pdf["title"].str.extract(r'\((\d{4})\)').astype(float)
    return pdf


@st.cache_data(show_spinner=False)
def load_static_results() -> Dict[str, Any]:
    """Đọc các tệp metrics tĩnh trong thư mục results/."""
    results: Dict[str, Any] = {}

    eval_json = RESULTS_DIR / "evaluation_summary_25m.json"
    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            results["evaluation_summary_25m"] = json.load(f)

    metrics_final_json = RESULTS_DIR / "metrics_25m_final.json"
    if metrics_final_json.exists():
        with open(metrics_final_json, "r", encoding="utf-8") as f:
            results["metrics_25m_final"] = json.load(f)

    tuning_csv = RESULTS_DIR / "tuning_results_25m.csv"
    if tuning_csv.exists():
        results["tuning_df"] = pd.read_csv(tuning_csv)

    scalability_csv = RESULTS_DIR / "scalability_results.csv"
    if scalability_csv.exists():
        results["scalability_df"] = pd.read_csv(scalability_csv)

    pop_json = RESULTS_DIR / "movie_popularity_25m.json"
    if pop_json.exists():
        with open(pop_json, "r", encoding="utf-8") as f:
            results["movie_popularity"] = json.load(f)
    else:
        results["movie_popularity"] = {"movie_stats": {}, "top_popular_movies": []}

    return results


def compute_genre_explanation(
    rec_genres_raw: str,
    user_liked_movies: List[Dict[str, Any]],
    movies_map: Dict[int, Tuple[str, str, str]],
) -> str:
    """Tạo dòng giải thích ngắn gọn dựa trên giao thoa thể loại với các phim user đã thích."""
    if not rec_genres_raw or rec_genres_raw == "(no genres listed)":
        return "Phù hợp gu tiềm ẩn từ phân rã ma trận ALS"

    rec_genres = set(g.strip() for g in str(rec_genres_raw).split("|") if g.strip())
    matches = []

    for h in user_liked_movies:
        m_info = movies_map.get(h["movieId"])
        if not m_info or not m_info[1] or m_info[1] == "(no genres listed)":
            continue
        liked_genres = set(g.strip() for g in str(m_info[1]).split("|") if g.strip())
        common = rec_genres & liked_genres
        if common:
            score = len(common) * 10 + float(h.get("rating", 0))
            raw_title = m_info[0]
            clean_title = raw_title.split(" (")[0] if " (" in raw_title else raw_title
            matches.append((score, clean_title))

    matches.sort(key=lambda x: x[0], reverse=True)
    if matches:
        distinct_titles = []
        for _, t in matches:
            if t not in distinct_titles:
                distinct_titles.append(t)
            if len(distinct_titles) == 2:
                break
        return "Vì bạn thích: " + ", ".join(distinct_titles)

    return "Phù hợp nhóm người dùng có cùng gu sở thích (ALS)"


def render_movie_rec_card(
    rank: int,
    movie_id: int,
    title: str,
    genres_vi: str,
    score: float,
    pop_count: int,
    explanation: str,
) -> str:
    """Render thẻ phim gợi ý dạng Card theo chuẩn Power BI Modern Indigo (Tương phản cao, sang trọng)."""
    clean_title = title if len(title) <= 46 else title[:43] + "..."
    pop_str = f"{pop_count:,}" if pop_count < 10000 else f"{pop_count/1000:.1f}K"

    match_pct = min(99, int((score / 5.0) * 100) + 5) if score >= 4.0 else int((score / 5.0) * 100)
    match_color = "#16A34A" if match_pct >= 90 else ("#D97706" if match_pct >= 75 else "#475569")

    return f"""<div style="background: #ffffff; border: 1px solid #CBD1E9; border-radius: 12px; padding: 14px 16px; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06); min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 12px; transition: all 0.2s ease;">
<div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="background: #4F46E5; color: #1E1B4B; font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.03em; box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25);">TOP {rank}</span>
<span style="font-size: 13px; font-weight: 900; color: {match_color};">{match_pct}% Phù hợp</span>
</div>
<div style="font-size: 14px; font-weight: 800; color: #0F172A; line-height: 1.4; min-height: 40px; margin-bottom: 6px;" title="{title}">{clean_title}</div>
<div style="font-size: 11.5px; color: #4338CA; font-weight: 600; margin-bottom: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{genres_vi}">{genres_vi}</div>
</div>
<div>
<div style="background: #F8FAFC; border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; border: 1px solid #E2E8F0;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
<span style="font-size: 11px; color: #64748B; font-weight: 700;">Dự đoán ALS:</span>
<span style="color: #D97706; font-size: 14px; font-weight: 800;">★ {score:.2f} <span style="font-size: 10px; color: #64748B;">/ 5</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 11px; color: #64748B; font-weight: 700;">Độ phổ biến:</span>
<span style="font-size: 12px; font-weight: 800; color: #0F172A;">{pop_str} lượt</span>
</div>
</div>
<div style="font-size: 11px; color: #4338CA; background: #EEF2FF; padding: 6px 10px; border-radius: 6px; border-left: 3px solid #5E48E8; line-height: 1.4; font-weight: 600;" title="{explanation}">
{explanation}
</div>
</div>
</div>"""


@st.cache_data(show_spinner=False)
def fetch_user_history(_spark, user_id: int) -> List[Dict[str, Any]]:
    """Truy vấn các lượt rating trong quá khứ của 1 userId từ tập Parquet."""
    from pyspark.sql.functions import col

    df = _spark.read.parquet(RATINGS_HDFS_PATH)
    user_rows = (
        df.filter(col("userId") == int(user_id))
        .select("movieId", "rating", "timestamp")
        .collect()
    )
    return [
        {
            "movieId": int(r["movieId"]),
            "rating": float(r["rating"]),
            "timestamp": int(r["timestamp"]),
        }
        for r in user_rows
    ]


# -----------------------------------------------------------------------------
# 4. Giao diện Chính & Báo Cáo Doanh Nghiệp Chuẩn Mực
# -----------------------------------------------------------------------------
def main() -> None:
    # Thanh Header Bar Chuẩn Mẫu Bank Analysis
    st.markdown(
        """<div class="pbi-topbar">
<div class="pbi-brand">
<div class="pbi-brand-logo">ML</div>
<div>
<div class="pbi-brand-title">MovieLens 25M Enterprise Analytics</div>
<div class="pbi-brand-sub">Apache Spark 3.5 MLlib (ALS) & Hadoop HDFS Distributed Platform</div>
</div>
</div>
<div class="pbi-topbar-pills">
<span class="pbi-pill active">Spark Master 7077</span>
<span class="pbi-pill">HDFS Port 9000</span>
<span class="pbi-pill">Final Model: Rank 50</span>
<span class="pbi-pill">Dataset: 25.0M</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    spark = get_shared_spark_session()
    model = get_cached_als_model(spark)
    movies_df = get_cached_movies(spark)
    static_results = load_static_results()

    movies_map = {
        row.movieId: (row.title, row.genres, row.genres_vi, getattr(row, 'year', 0))
        for row in movies_df.itertuples(index=False)
    }

    # -------------------------------------------------------------------------
    # Slicers Panel Bên Trái (Thẻ trắng thanh lịch)
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """<div style="font-size: 0.88rem; font-weight: 800; color: #1E1B4B; text-transform: uppercase; margin-bottom: 0.8rem; letter-spacing: 0.04em;">
BỘ LỌC ĐIỀU KHIỂN (SLICERS)
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="slicer-card"><div class="slicer-label">Mã người dùng (User ID)</div>', unsafe_allow_html=True)
        sample_users = [
            ("User #1 (70 phim - Phiêu lưu & Kịch tính)", 1),
            ("User #42 (Gu Hành động & Khoa học viễn tưởng)", 42),
            ("User #100 (Người dùng thường xuyên)", 100),
            ("User #250 (Gu điện ảnh đa dạng)", 250),
            ("User #75 (Trường hợp Cold-Start - 3 phim)", 75),
            ("Nhập mã User ID khác...", -1),
        ]
        user_choice = st.selectbox("Chọn User:", options=sample_users, format_func=lambda x: x[0], index=0, label_visibility="collapsed")
        if user_choice[1] == -1:
            target_user_id = st.number_input("Nhập User ID (1 - 162,541):", min_value=1, max_value=162541, value=1, step=1)
        else:
            target_user_id = user_choice[1]
        st.session_state["selected_user_id"] = target_user_id
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="slicer-card"><div class="slicer-label">Số lượng gợi ý (Top K)</div>', unsafe_allow_html=True)
        top_k_select = st.select_slider("Số lượng phim:", options=[5, 10, 15, 20], value=10, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="slicer-card"><div class="slicer-label">Lọc theo Thể loại</div>', unsafe_allow_html=True)
        genre_options = ["Tất cả thể loại (All)"] + sorted(list(GENRE_TRANSLATIONS.values()))
        selected_genre_filter = st.selectbox("Thể loại:", options=genre_options, index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="slicer-card"><div class="slicer-label">Cảm hứng & Tâm trạng</div>', unsafe_allow_html=True)
        selected_mood = st.selectbox("Tâm trạng:", options=list(MOOD_MAPPING.keys()), index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="slicer-card"><div class="slicer-label">Kỷ nguyên phát hành</div>', unsafe_allow_html=True)
        selected_era = st.selectbox("Năm phát hành:", options=["Tất cả mọi thời đại", "Phim Mới (Sau 2010)", "Thập niên 2000s", "Thập niên 90s", "Phim Cổ Điển (Trước 1990)"], index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""<div class="slicer-card" style="background: #F8FAFC; border-left: 3px solid #5E48E8;">
<div class="slicer-label" style="color: #5E48E8; margin-bottom: 0.3rem;">TRẠNG THÁI HIỆN TẠI</div>
<div style="font-size: 0.76rem; color: #1E1B4B; line-height: 1.5;">
<div>Đang chọn: <b>User #{target_user_id}</b></div>
<div>Mục tiêu: <b>{top_k_select} phim {selected_era.split(" ")[0]}</b></div>
<div>Thể loại: <b>{selected_genre_filter}</b></div>
<div>Cảm xúc: <b>{selected_mood.split(" ")[0]}</b></div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Hàng 5 Thẻ KPI Chuẩn Mẫu Bank Analysis
    # -------------------------------------------------------------------------
    sp_ratings = generate_bar_sparkline_svg([1.2, 1.8, 2.5, 3.1, 4.0, 4.8, 6.2, 6.6, 7.8, 8.5], color="#5E48E8")
    sp_users = generate_area_sparkline_svg([10, 25, 45, 70, 95, 120, 140, 162], stroke="#5E48E8")
    sp_movies = generate_bar_sparkline_svg([15, 22, 34, 45, 52, 58, 61, 62.4], color="#5E48E8")
    sp_rmse = generate_area_sparkline_svg([1.0222, 0.9972, 0.9111, 0.8861, 0.8506, 0.8455], stroke="#5E48E8")
    sp_prec = generate_area_sparkline_svg([64.3, 66.1, 67.2, 70.1, 69.6, 73.4], stroke="#5E48E8")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi_card("TỔNG RATINGS", "25.0M", sp_ratings, "YoY% +100%", True, "Parquet 16P Partition"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("NGƯỜI DÙNG", "162.5K", sp_users, "Active 100%", True, "TB 153.8 ratings/user"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("DANH MỤC PHIM", "62.4K", sp_movies, "Độ thưa 99.75%", False, "20 Thể loại điện ảnh"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("ALS TEST RMSE", "0.8455", sp_rmse, "Cải thiện -16.3%", True, "Baseline 1.0097"), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi_card("PRECISION@5", "73.4%", sp_prec, "Vượt trội +3.8%", True, "Popularity 69.6%"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5 Tab Báo Cáo Không Chứa Emoji
    # -------------------------------------------------------------------------
    tabs = st.tabs([
        "Tổng quan Dữ liệu",
        "Hiệu năng Mô hình",
        "Gợi ý Phim cho Người dùng",
        "Lịch sử Đánh giá",
        "Khảo sát Khả năng Mở rộng (Scalability)",
    ])

    # =========================================================================
    # TAB 1: TỔNG QUAN DỮ LIỆU (OVERVIEW)
    # =========================================================================
    with tabs[0]:
        c1, c2 = st.columns([1.3, 1])

        with c1:
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">SẢN LƯỢNG ĐÁNH GIÁ THEO THANG ĐIỂM (RATING DISTRIBUTION)</div>""",
                unsafe_allow_html=True,
            )
            rdf = pd.DataFrame({
                "Mức điểm": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "Số lượng": [393068, 776815, 399490, 1640868, 1262797, 4896928, 3177301, 6639798, 2200539, 3612491],
            })
            rdf["Tỉ lệ (%)"] = (rdf["Số lượng"] / 25000095 * 100).round(1)

            # Combo Chart Chuẩn Mẫu Bank Analysis: Cột Tím + Đường Line Vàng có data labels
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_combo.add_trace(
                go.Bar(
                    x=rdf["Mức điểm"].astype(str) + " sao",
                    y=rdf["Số lượng"],
                    name="Số lượt đánh giá",
                    marker_color="#5E48E8",
                ),
                secondary_y=False,
            )
            fig_combo.add_trace(
                go.Scatter(
                    x=rdf["Mức điểm"].astype(str) + " sao",
                    y=rdf["Tỉ lệ (%)"],
                    name="Tỉ lệ %",
                    mode="lines+markers+text",
                    line=dict(color="#F59E0B", width=3),
                    marker=dict(size=7, color="#F59E0B"),
                    text=rdf["Tỉ lệ (%)"].astype(str) + "%",
                    textposition="top center",
                    textfont=dict(size=9, color="#1E1B4B", family="Plus Jakarta Sans"),
                ),
                secondary_y=True,
            )
            fig_combo.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F3FA", title="Số lượt đánh giá"),
                yaxis2=dict(showgrid=False, title="Tỉ lệ (%)", range=[0, 35]),
            )
            st.plotly_chart(fig_combo, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">TỶ TRỌNG CÁC THỂ LOẠI PHIM HÀNG ĐẦU (GENRE SHARE DONUT)</div>""",
                unsafe_allow_html=True,
            )
            all_genres = []
            for g_str in movies_df["genres"].dropna():
                for g in g_str.split("|"):
                    if g.strip() and g.strip() != "(no genres listed)":
                        all_genres.append(g.strip())

            genres_count = pd.Series(all_genres).value_counts().head(6).reset_index()
            genres_count.columns = ["Thể loại gốc", "Số bộ phim"]
            genres_count["Thể loại"] = genres_count["Thể loại gốc"].map(GENRE_TRANSLATIONS)

            # Donut Chart Chuẩn Bank Analysis: Vành khuyên dày, màu tím gradient
            donut_colors = ["#5E48E8", "#8B7CF6", "#A78BFA", "#C4B5FD", "#60A5FA", "#CBD5E1"]
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=genres_count["Thể loại"],
                        values=genres_count["Số bộ phim"],
                        hole=0.62,
                        marker=dict(colors=donut_colors),
                        textinfo="percent",
                        textfont=dict(size=10, color="#ffffff"),
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig_donut.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=10)),
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Bảng dữ liệu ma trận tổng thể
        st.markdown(
            """<div class="pbi-panel">
<div class="pbi-panel-title">BẢNG CHI TIẾT TỶ TRỌNG ĐÁNH GIÁ (DATASET MATRIX)</div>""",
            unsafe_allow_html=True,
        )
        matrix_df = pd.DataFrame([
            {"Nhóm Đánh giá": "Tích cực (High Rating)", "Thang điểm": "4.0 - 5.0 sao", "Lượt tương tác": 12452828, "Tỉ trọng": "49.8%", "Ý nghĩa": "Hài lòng cao (Ngưỡng Relevant cho Precision@K)"},
            {"Nhóm Đánh giá": "Trung bình (Medium Rating)", "Thang điểm": "2.5 - 3.5 sao", "Lượt tương tác": 9337026, "Tỉ trọng": "37.3%", "Ý nghĩa": "Đánh giá trung lập trong ma trận tương tác"},
            {"Nhóm Đánh giá": "Tiêu cực (Low Rating)", "Thang điểm": "0.5 - 2.0 sao", "Lượt tương tác": 3210241, "Tỉ trọng": "12.9%", "Ý nghĩa": "Định hình nhân tố phạt trong hàm mất mát ALS"},
        ])
        st.dataframe(
            matrix_df.style.format({"Lượt tương tác": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: HIỆU NĂNG MÔ HÌNH (MODEL INSIGHTS)
    # =========================================================================
    with tabs[1]:
        eval_summary = static_results.get("evaluation_summary_25m", {})
        models_comp = eval_summary.get("models_comparison", {})

        als_rmse = models_comp.get("als_final", {}).get("rmse", 0.8455)
        als_mae = models_comp.get("als_final", {}).get("mae", 0.6333)
        pop_rmse = models_comp.get("popularity_baseline", {}).get("rmse", 1.0097)
        pop_mae = models_comp.get("popularity_baseline", {}).get("mae", 0.7720)

        t1, t2 = st.columns([1.1, 1])
        with t1:
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">SO SÁNH ĐỊNH LƯỢNG: ALS FINAL VS POPULARITY BASELINE</div>""",
                unsafe_allow_html=True,
            )
            comp_df = pd.DataFrame([
                {"Chỉ số Đánh giá": "Root Mean Squared Error (RMSE)", "ALS Final Model": f"{als_rmse:.4f}", "Popularity Baseline": f"{pop_rmse:.4f}", "Mức Cải thiện": "-16.3% (Tốt)", "Đánh giá": "Giảm thiểu triệt để sai số bình phương lớn"},
                {"Chỉ số Đánh giá": "Mean Absolute Error (MAE)", "ALS Final Model": f"{als_mae:.4f}", "Popularity Baseline": f"{pop_mae:.4f}", "Mức Cải thiện": "-18.0% (Tốt)", "Đánh giá": "Độ lệch trung bình chỉ ~0.63 sao trên thang 5"},
                {"Chỉ số Đánh giá": "Precision@5 (Top-5 Đúng Gu)", "ALS Final Model": "73.4%", "Popularity Baseline": "69.6%", "Mức Cải thiện": "+3.8% (Tốt)", "Đánh giá": "73.4% phim trong Top 5 được người dùng đánh giá cao"},
                {"Chỉ số Đánh giá": "Precision@10 (Top-10 Đúng Gu)", "ALS Final Model": "70.1%", "Popularity Baseline": "67.2%", "Mức Cải thiện": "+2.9% (Tốt)", "Đánh giá": "Độ chính xác cao và ổn định trên diện rộng"},
                {"Chỉ số Đánh giá": "Precision@20 (Top-20 Đúng Gu)", "ALS Final Model": "66.1%", "Popularity Baseline": "64.3%", "Mức Cải thiện": "+1.8% (Tốt)", "Đánh giá": "Duy trì chất lượng khi mở rộng danh sách"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with t2:
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">SO SÁNH PRECISION VÀ RECALL THEO K</div>""",
                unsafe_allow_html=True,
            )
            fig_pr = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pr.add_trace(
                go.Bar(
                    x=["Top 5", "Top 10", "Top 20"],
                    y=[73.4, 70.1, 66.1],
                    name="ALS Precision (%)",
                    marker_color="#5E48E8",
                    text=["73.4%", "70.1%", "66.1%"],
                    textposition="auto",
                ),
                secondary_y=False,
            )
            fig_pr.add_trace(
                go.Bar(
                    x=["Top 5", "Top 10", "Top 20"],
                    y=[69.6, 67.2, 64.3],
                    name="Popularity Prec (%)",
                    marker_color="#CBD5E1",
                    text=["69.6%", "67.2%", "64.3%"],
                    textposition="auto",
                ),
                secondary_y=False,
            )
            fig_pr.add_trace(
                go.Scatter(
                    x=["Top 5", "Top 10", "Top 20"],
                    y=[28.4, 40.2, 53.0],
                    name="ALS Recall (%)",
                    mode="lines+markers+text",
                    line=dict(color="#F59E0B", width=3),
                    marker=dict(size=7, color="#F59E0B"),
                    text=["28.4%", "40.2%", "53.0%"],
                    textposition="top center",
                ),
                secondary_y=True,
            )
            fig_pr.update_layout(
                height=265,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#F1F3FA", title="Precision (%)", range=[50, 85]),
                yaxis2=dict(showgrid=False, title="Recall (%)", range=[15, 65]),
            )
            st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Tuning Grid Search
        st.markdown(
            """<div class="pbi-panel">
<div class="pbi-panel-title">KẾT QUẢ 9 THỰC NGHIỆM TUNING SIÊU THAM SỐ (RANK × REGPARAM)</div>""",
            unsafe_allow_html=True,
        )
        tuning_df = static_results.get("tuning_df")
        if tuning_df is not None and not tuning_df.empty:
            tc1, tc2 = st.columns([1, 1.2])
            with tc1:
                disp_tune = tuning_df.copy()
                disp_tune.columns = ["Rank (Bậc)", "RegParam (L2)", "Test RMSE", "Test MAE", "Thời gian (s)"]
                st.dataframe(
                    disp_tune.style.highlight_min(subset=["Test RMSE", "Test MAE"], color="#EDE9FE").format({
                        "Test RMSE": "{:.4f}",
                        "Test MAE": "{:.4f}",
                        "Thời gian (s)": "{:.1f}s",
                        "RegParam (L2)": "{:.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=240,
                )
            with tc2:
                fig_tune_line = px.line(
                    tuning_df,
                    x="rank",
                    y="rmse",
                    color="regParam",
                    markers=True,
                    labels={"rank": "Rank", "rmse": "Test RMSE", "regParam": "L2 Reg"},
                    color_discrete_map={0.01: "#38BDF8", 0.1: "#5E48E8", 0.5: "#F43F5E"},
                )
                fig_tune_line.update_traces(line=dict(width=2.5), marker=dict(size=7))
                fig_tune_line.update_layout(
                    height=240,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickvals=[10, 20, 50], showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#F1F3FA", title="Test RMSE"),
                )
                st.plotly_chart(fig_tune_line, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """<div style="background: #F8FAFC; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.82rem; color: #1E1B4B; border-left: 4px solid #5E48E8; margin-top: 0.6rem;">
<b>Kết luận cấu hình tối ưu:</b> Cấu hình <code>rank = 50</code>, <code>regParam = 0.1</code>, <code>maxIter = 15</code> đạt Test RMSE thấp nhất (<b>0.8455</b>) và MAE (<b>0.6333</b>), dung hòa hoàn hảo giữa độ phức tạp mô hình và khả năng khái quát hóa trên dữ liệu thưa 99.75%.
</div>
</div>""",
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 3: GỢI Ý PHIM CHO NGƯỜI DÙNG (TRUNG TÂM PHÂN TÍCH & ĐỀ XUẤT)
    # =========================================================================
    with tabs[2]:
        active_uid = st.session_state.get("selected_user_id", 1)
        movie_pop_data = static_results.get("movie_popularity", {"movie_stats": {}, "top_popular_movies": []})
        pop_stats = movie_pop_data.get("movie_stats", {})
        top_pop_ids = movie_pop_data.get("top_popular_movies", [])

        # ---------------------------------------------------------------------
        # 1. Khối "Hồ sơ người dùng" (Hiển thị ngay khi chọn User từ dữ liệu lịch sử)
        # ---------------------------------------------------------------------
        user_history = fetch_user_history(spark, active_uid)
        history_count = len(user_history)
        rated_ids: Set[int] = {h["movieId"] for h in user_history}

        if history_count == 0:
            st.error(f"Người dùng #{active_uid} không tồn tại trong tập dữ liệu MovieLens 25M.")
        else:
            ratings_series = [float(h["rating"]) for h in user_history]
            avg_user_rating = float(np.mean(ratings_series))

            # Xác định thể loại yêu thích nhất từ các phim user chấm >= 4.0 sao
            fav_genres = []
            for h in user_history:
                if float(h["rating"]) >= 4.0:
                    m_info = movies_map.get(h["movieId"])
                    if m_info and m_info[1] != "(no genres listed)":
                        fav_genres.extend([g.strip() for g in str(m_info[1]).split("|") if g.strip()])
            if not fav_genres:
                for h in user_history:
                    m_info = movies_map.get(h["movieId"])
                    if m_info and m_info[1] != "(no genres listed)":
                        fav_genres.extend([g.strip() for g in str(m_info[1]).split("|") if g.strip()])

            top_fav_genre_en = str(pd.Series(fav_genres).value_counts().index[0]) if fav_genres else "Drama"
            top_fav_genre_vi = GENRE_TRANSLATIONS.get(top_fav_genre_en, top_fav_genre_en)

            # Top 5 phim user chấm điểm cao nhất trong lịch sử
            sorted_history = sorted(user_history, key=lambda x: (float(x["rating"]), int(x.get("timestamp", 0))), reverse=True)
            top_5_history = sorted_history[:5]
            user_liked_movies = [h for h in user_history if float(h["rating"]) >= 4.0]
            if not user_liked_movies:
                user_liked_movies = sorted_history[:10]

            # Render Khối Hồ Sơ Người Dùng
            st.markdown(
                f"""<div class="pbi-panel">
<div class="pbi-panel-title" style="display: flex; justify-content: space-between; align-items: center;">
<span>HỒ SƠ ĐIỆN ẢNH NGƯỜI DÙNG: USER #{active_uid}</span>
<span style="font-size: 0.75rem; font-weight: 800; color: #4F46E5; background: #EEF2FF; border: 1px solid #C7D2FE; padding: 4px 12px; border-radius: 20px;">DỮ LIỆU THỰC TẾ (GROUND TRUTH)</span>
</div>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px;">
<div style="background: #F8FAFC; border: 1px solid #CBD1E9; border-radius: 10px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
<div style="font-size: 0.75rem; color: #64748B; font-weight: 800; text-transform: uppercase;">Tổng số phim đã xem</div>
<div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; line-height: 1.2; margin-top: 4px;">{history_count:,} <span style="font-size: 0.85rem; color: #64748B; font-weight: 600;">bộ phim</span></div>
</div>
<div style="background: #F8FAFC; border: 1px solid #CBD1E9; border-radius: 10px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
<div style="font-size: 0.75rem; color: #64748B; font-weight: 800; text-transform: uppercase;">Điểm đánh giá trung bình</div>
<div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; line-height: 1.2; margin-top: 4px;">{avg_user_rating:.2f} <span style="font-size: 0.85rem; color: #64748B; font-weight: 600;">/ 5.0 sao</span></div>
</div>
<div style="background: #F8FAFC; border: 1px solid #CBD1E9; border-radius: 10px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);">
<div style="font-size: 0.75rem; color: #64748B; font-weight: 800; text-transform: uppercase;">Thể loại yêu thích nhất</div>
<div style="font-size: 1.6rem; font-weight: 800; color: #4F46E5; line-height: 1.2; margin-top: 4px;">{top_fav_genre_vi} <span style="font-size: 0.8rem; color: #64748B; font-weight: 600;">({top_fav_genre_en})</span></div>
</div>
</div>
<div style="font-size: 0.8rem; font-weight: 800; color: #0F172A; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.03em;">Top 5 tác phẩm đánh giá cao nhất:</div>
""",
                unsafe_allow_html=True,
            )

            # 5 thẻ con hiển thị 5 phim chấm cao nhất
            hist_cols = st.columns(5)
            for idx, h_item in enumerate(top_5_history):
                m_info = movies_map.get(h_item["movieId"], (f"Phim #{h_item['movieId']}", "", "Chưa phân loại"))
                title_short = m_info[0] if len(m_info[0]) <= 32 else m_info[0][:30] + "..."
                genre_short = m_info[2].split(",")[0]
                with hist_cols[idx]:
                    st.markdown(
                        f"""<div style="background: #ffffff; border: 1px solid #CBD1E9; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04); border-radius: 8px; padding: 10px 12px; height: 100%; min-height: 70px;">
<div style="font-size: 12px; font-weight: 800; color: #0F172A; line-height: 1.3; margin-bottom: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{m_info[0]}">{title_short}</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 11px; color: #4338CA; font-weight: 700;">{genre_short}</span>
<span style="color: #D97706; font-size: 13px; font-weight: 900;">★ {float(h_item['rating']):.1f}</span>
</div>
</div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            if history_count < 5:
                st.warning(f"Cảnh báo Cold-Start: Người dùng #{active_uid} mới chỉ có {history_count} đánh giá (< 5). Không gian tiềm ẩn có thể chưa phản ánh trọn vẹn gu cá nhân.")

            # -----------------------------------------------------------------
            # 4. Điều khiển: Chọn số lượng gợi ý (5/10/20) & Chiến lược lọc
            # -----------------------------------------------------------------
            st.markdown(
                """<div class="pbi-panel" style="padding-bottom: 0.8rem;">
<div class="pbi-panel-title">ĐIỀU KHIỂN CHIẾN LƯỢC GỢI Ý & BỘ LỌC ĐỀ XUẤT</div>""",
                unsafe_allow_html=True,
            )
            ctrl_c1, ctrl_c2 = st.columns([1, 1.8])
            with ctrl_c1:
                tab3_top_k = st.select_slider(
                    "Số lượng phim gợi ý hiển thị:",
                    options=[5, 10, 20],
                    value=top_k_select if top_k_select in [5, 10, 20] else 10,
                    key="tab3_top_k_slider",
                )
            with ctrl_c2:
                filter_strategy = st.radio(
                    "Chiến lược định hướng danh mục:",
                    options=[
                        "Mặc định (Theo điểm ALS)",
                        "Ưu tiên Phim Phổ biến (Nhiều lượt đánh giá)",
                        "Ưu tiên Phim Ít người biết (Khám phá / Niche)",
                    ],
                    index=0,
                    horizontal=True,
                    key="tab3_filter_strategy",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # Mô hình ALS suy diễn & Xử lý lọc / sắp xếp
            # -----------------------------------------------------------------
            user_df = spark.createDataFrame([(int(active_uid),)], ["userId"])
            recs_raw = model.recommendForUserSubset(user_df, max(90, history_count + 45)).collect()

            if not recs_raw or len(recs_raw[0]["recommendations"]) == 0:
                st.info("Không có gợi ý nào phù hợp cho người dùng này.")
            else:
                raw_items = recs_raw[0]["recommendations"]

                # Lọc 100% phim đã xem
                unrated_items = [
                    {"movieId": int(r["movieId"]), "rating": float(r["rating"])}
                    for r in raw_items
                    if int(r["movieId"]) not in rated_ids
                ]

                # Áp dụng bộ lọc thể loại từ Sidebar
                candidate_pool = []
                for item in unrated_items:
                    m_info = movies_map.get(item["movieId"])
                    if not m_info:
                        continue
                    
                    title, genres_en, genres_vi, year = m_info

                    if selected_genre_filter != "Tất cả thể loại (All)":
                        if selected_genre_filter not in genres_vi:
                            continue
                            
                    mood_genres = MOOD_MAPPING.get(selected_mood)
                    if mood_genres:
                        if not any(g in genres_en for g in mood_genres):
                            continue
                            
                    if selected_era == "Phim Mới (Sau 2010)" and not (year >= 2010): continue
                    if selected_era == "Thập niên 2000s" and not (2000 <= year < 2010): continue
                    if selected_era == "Thập niên 90s" and not (1990 <= year < 2000): continue
                    if selected_era == "Phim Cổ Điển (Trước 1990)" and not (year < 1990): continue

                    m_stat = pop_stats.get(str(item["movieId"]), {})
                    p_count = int(m_stat.get("count", 0))
                    p_avg = float(m_stat.get("avg", 0.0))

                    candidate_pool.append({
                        "movieId": item["movieId"],
                        "als_score": min(float(item["rating"]), 5.0),
                        "title": title,
                        "genres_en": genres_en,
                        "genres_vi": genres_vi,
                        "pop_count": p_count,
                        "pop_avg": p_avg,
                    })

                # Sắp xếp lại danh sách dựa trên chiến lược đã chọn
                if filter_strategy == "Ưu tiên Phim Phổ biến (Nhiều lượt đánh giá)":
                    sorted_pool = sorted(candidate_pool, key=lambda x: (x["pop_count"], x["als_score"]), reverse=True)
                elif filter_strategy == "Ưu tiên Phim Ít người biết (Khám phá / Niche)":
                    sorted_pool = sorted(candidate_pool, key=lambda x: (x["pop_count"], -x["als_score"]))
                else:
                    sorted_pool = sorted(candidate_pool, key=lambda x: x["als_score"], reverse=True)

                final_displayed_recs = sorted_pool[:tab3_top_k]

                # -------------------------------------------------------------
                # 2. Kết quả gợi ý hiển thị dạng CARD (Grid 5 cột)
                # 3. Dòng giải thích giao thoa thể loại
                # 6. Cột/thông số độ phổ biến
                # -------------------------------------------------------------
                st.markdown(
                    f"""<div class="pbi-panel">
<div class="pbi-panel-title" style="display: flex; justify-content: space-between; align-items: center;">
<span>KẾT QUẢ TOP {len(final_displayed_recs)} BỘ PHIM GỢI Ý (DẠNG THẺ CARD THEO LƯỚI 5 CỘT)</span>
<span style="font-size: 0.78rem; color: #166534; font-weight: 700; background: #DCFCE7; padding: 4px 10px; border-radius: 6px;">ĐÃ LOẠI TRỪ 100% ({len(rated_ids)} PHIM) ĐÃ XEM</span>
</div>""",
                    unsafe_allow_html=True,
                )

                # Hiển thị từng hàng 5 cột chuẩn Power BI Card Grid
                for row_start in range(0, len(final_displayed_recs), 5):
                    row_items = final_displayed_recs[row_start : row_start + 5]
                    grid_cols = st.columns(5)
                    for col_idx, item in enumerate(row_items):
                        overall_rank = row_start + col_idx + 1
                        explanation = compute_genre_explanation(item["genres_en"], user_liked_movies, movies_map)
                        card_html = render_movie_rec_card(
                            rank=overall_rank,
                            movie_id=item["movieId"],
                            title=item["title"],
                            genres_vi=item["genres_vi"],
                            score=item["als_score"],
                            pop_count=item["pop_count"],
                            explanation=explanation,
                        )
                        with grid_cols[col_idx]:
                            st.markdown(card_html, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # 5. Chế độ so sánh: 2 cột song song ALS (Cá nhân hóa) vs Popularity (Đại trà)
                # -------------------------------------------------------------
                st.markdown(
                    f"""<div class="pbi-panel">
<div class="pbi-panel-title" style="display: flex; justify-content: space-between; align-items: center;">
<span>ĐỐI SÁNH TRỰC QUAN: TOP 5 GỢI Ý CÁ NHÂN HÓA (ALS) VS TOP 5 GỢI Ý ĐẠI TRÀ (POPULARITY)</span>
<span style="font-size: 0.74rem; font-weight: 700; color: #5E48E8; background: #EDE9FE; padding: 3px 8px; border-radius: 4px;">BENCHMARK SO SÁNH CHO USER #{active_uid}</span>
</div>""",
                    unsafe_allow_html=True,
                )

                col_als, col_pop = st.columns(2)

                with col_als:
                    st.markdown(
                        f"""<div style="background: #F8FAFC; border: 1px solid #CBD1E9; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px;">
<div style="font-size: 0.86rem; font-weight: 800; color: #5E48E8; text-transform: uppercase;">CỘT 1: TOP 5 GỢI Ý TỪ ALS (CÁ NHÂN HÓA)</div>
<div style="font-size: 0.76rem; color: #64748B; margin-top: 2px;">Dựa trên không gian tiềm ẩn (Latent Factor Rank=50) tối ưu riêng cho User #{active_uid}:</div>
</div>""",
                        unsafe_allow_html=True,
                    )
                    top_5_als = sorted(candidate_pool, key=lambda x: x["als_score"], reverse=True)[:5]
                    for idx, a_item in enumerate(top_5_als, 1):
                        a_exp = compute_genre_explanation(a_item["genres_en"], user_liked_movies, movies_map)
                        pop_txt = f"{a_item['pop_count']:,} lượt" if a_item['pop_count'] < 10000 else f"{a_item['pop_count']/1000:.1f}K lượt"
                        st.markdown(
                            f"""<div style="background: #ffffff; border: 1px solid #E6E8F4; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; box-shadow: 0 2px 5px rgba(94, 72, 232, 0.03);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
<span style="font-size: 0.74rem; font-weight: 800; color: #5E48E8; background: #EDE9FE; padding: 1px 6px; border-radius: 4px;">ALS #{idx}</span>
<span style="color: #D97706; font-size: 0.82rem; font-weight: 800;">★ {a_item['als_score']:.2f} <span style="font-size: 0.7rem; color: #64748B;">/ 5</span></span>
</div>
<div style="font-size: 0.84rem; font-weight: 700; color: #1E1B4B; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{a_item['title']}</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 3px; font-size: 0.74rem;">
<span style="color: #6D28D9; font-weight: 600;">{a_item['genres_vi']}</span>
<span style="color: #64748B;">Độ phổ biến: <b>{pop_txt}</b></span>
</div>
<div style="font-size: 0.72rem; color: #5E48E8; margin-top: 4px; background: #F8FAFC; padding: 3px 6px; border-radius: 4px;">{a_exp}</div>
</div>""",
                            unsafe_allow_html=True,
                        )

                with col_pop:
                    st.markdown(
                        f"""<div style="background: #F8FAFC; border: 1px solid #CBD1E9; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px;">
<div style="font-size: 0.86rem; font-weight: 800; color: #1E1B4B; text-transform: uppercase;">CỘT 2: TOP 5 GỢI Ý TỪ POPULARITY (ĐẠI TRÀ)</div>
<div style="font-size: 0.76rem; color: #64748B; margin-top: 2px;">Các tác phẩm có số lượt xem và điểm trung bình cao nhất toàn hệ thống mà User chưa xem:</div>
</div>""",
                        unsafe_allow_html=True,
                    )
                    # Lấy Top 5 phim phổ biến nhất toàn hệ thống mà user chưa xem
                    unwatched_popular = [mid for mid in top_pop_ids if mid not in rated_ids][:5]
                    for idx, p_mid in enumerate(unwatched_popular, 1):
                        p_info = movies_map.get(p_mid, (f"Phim #{p_mid}", "", "Chưa phân loại"))
                        p_stat = pop_stats.get(str(p_mid), {})
                        p_cnt = int(p_stat.get("count", 0))
                        p_avg = float(p_stat.get("avg", 0.0))
                        p_txt = f"{p_cnt:,} lượt" if p_cnt < 10000 else f"{p_cnt/1000:.1f}K lượt"
                        st.markdown(
                            f"""<div style="background: #ffffff; border: 1px solid #E6E8F4; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
<span style="font-size: 0.74rem; font-weight: 800; color: #1E1B4B; background: #E2E8F0; padding: 1px 6px; border-radius: 4px;">POP #{idx}</span>
<span style="color: #64748B; font-size: 0.82rem; font-weight: 700;">★ {p_avg:.2f} <span style="font-size: 0.7rem; color: #64748B;">/ 5</span></span>
</div>
<div style="font-size: 0.84rem; font-weight: 700; color: #1E1B4B; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{p_info[0]}</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 3px; font-size: 0.74rem;">
<span style="color: #6D28D9; font-weight: 600;">{p_info[2]}</span>
<span style="color: #64748B;">Độ phổ biến: <b>{p_txt}</b></span>
</div>
<div style="font-size: 0.72rem; color: #64748B; margin-top: 4px; background: #F1F5F9; padding: 3px 6px; border-radius: 4px;">Gợi ý chung cho mọi người dùng (Không cá nhân hóa)</div>
</div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f"""<div style="background: #F8FAFC; border-radius: 8px; padding: 10px 14px; font-size: 0.81rem; color: #1E1B4B; border-left: 3px solid #5E48E8; margin-top: 10px; line-height: 1.5;">
<b>Ý nghĩa phân tích:</b> Popularity Baseline luôn đưa ra các bộ phim quốc dân quen thuộc (như <i>The Shawshank Redemption</i>, <i>The Godfather</i>). Trong khi đó, mô hình ALS đào sâu vào 50 chiều tiềm ẩn để phát hiện những bộ phim phù hợp chính xác với gu thể loại yêu thích (<b>{top_fav_genre_vi}</b>) của User #{active_uid}, ngay cả khi phim đó ít phổ biến hơn trên toàn mạng lưới.
</div>
</div>""",
                    unsafe_allow_html=True,
                )

                # Bảng chi tiết bổ trợ trong Expander nếu cần kiểm tra số liệu
                with st.expander("Xem chi tiết danh sách dưới dạng Bảng Dữ liệu (Data Table View)"):
                    table_rows = []
                    for idx, item in enumerate(final_displayed_recs, 1):
                        table_rows.append({
                            "Thứ hạng": f"Top {idx}",
                            "Mã phim": item["movieId"],
                            "Tên tác phẩm điện ảnh": item["title"],
                            "Thể loại (Tiếng Việt)": item["genres_vi"],
                            "Điểm dự đoán ALS": round(item["als_score"], 2),
                            "Độ phổ biến (Ratings)": item["pop_count"],
                            "Lý do gợi ý": compute_genre_explanation(item["genres_en"], user_liked_movies, movies_map),
                        })
                    st.dataframe(
                        pd.DataFrame(table_rows),
                        column_config={
                            "Điểm dự đoán ALS": st.column_config.ProgressColumn(
                                "Điểm dự đoán ALS",
                                format="%.2f",
                                min_value=0.0,
                                max_value=5.0,
                            ),
                            "Độ phổ biến (Ratings)": st.column_config.NumberColumn(
                                "Độ phổ biến (Ratings)",
                                format="%d",
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

    # =========================================================================
    # TAB 4: LỊCH SỬ ĐÁNH GIÁ CỦA NGƯỜI DÙNG (USER HISTORY)
    # =========================================================================
    with tabs[3]:
        active_uid = st.session_state.get("selected_user_id", 1)
        user_history = fetch_user_history(spark, active_uid)

        if not user_history:
            st.info(f"Người dùng #{active_uid} chưa có lịch sử đánh giá.")
        else:
            ratings_series = [h["rating"] for h in user_history]
            avg_rating = np.mean(ratings_series)
            max_rating = np.max(ratings_series)
            min_rating = np.min(ratings_series)

            fav_genres_list = []
            for h in user_history:
                if h["rating"] >= 4.0:
                    m_info = movies_map.get(h["movieId"])
                    if m_info and m_info[1] != "(no genres listed)":
                        fav_genres_list.extend(m_info[1].split("|"))

            fav_genre_str = "Chưa xác định"
            if fav_genres_list:
                top_genre = pd.Series(fav_genres_list).value_counts().index[0]
                fav_genre_str = f"{top_genre} ({GENRE_TRANSLATIONS.get(top_genre, top_genre)})"

            # 4 Thẻ chỉ số User
            hu1, hu2, hu3, hu4 = st.columns(4)
            with hu1:
                st.markdown(render_kpi_card("TỔNG PHIM ĐÃ XEM", f"{len(user_history)} phim", sp_users, "100% History", True, "Parquet 25M"), unsafe_allow_html=True)
            with hu2:
                st.markdown(render_kpi_card("ĐIỂM TRUNG BÌNH", f"{avg_rating:.2f} sao", sp_ratings, "Thang 5.0", False, "Mức độ bình thường"), unsafe_allow_html=True)
            with hu3:
                st.markdown(render_kpi_card("CAO NHẤT / THẤP NHẤT", f"{max_rating:.1f} / {min_rating:.1f}", sp_rmse, f"Biên độ {(max_rating - min_rating):.1f}", True, "Phân hóa rõ ràng"), unsafe_allow_html=True)
            with hu4:
                st.markdown(render_kpi_card("THỂ LOẠI TÂM ĐẮC", fav_genre_str.split(" ")[0], sp_prec, "Top 1 Gu", True, fav_genre_str), unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

            # Bảng chi tiết toàn bộ phim user đã xem
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">DANH SÁCH TÁC PHẨM ĐÃ ĐÁNH GIÁ (SẮP XẾP THEO ĐIỂM GIẢM DẦN)</div>""",
                unsafe_allow_html=True,
            )

            hist_rows = []
            sorted_history = sorted(user_history, key=lambda x: x["rating"], reverse=True)
            for h in sorted_history:
                m_info = movies_map.get(h["movieId"], (f"Movie #{h['movieId']}", "Unknown", "Chưa phân loại"))
                date_str = time.strftime("%d/%m/%Y", time.localtime(h["timestamp"])) if h.get("timestamp") else "N/A"
                hist_rows.append({
                    "Mã phim": h["movieId"],
                    "Tên tác phẩm điện ảnh": m_info[0],
                    "Thể loại gốc": m_info[1],
                    "Thể loại tiếng Việt": m_info[2],
                    "Điểm đã chấm": float(h["rating"]),
                    "Thời gian đánh giá": date_str,
                })

            hist_df = pd.DataFrame(hist_rows)
            st.dataframe(
                hist_df,
                column_config={
                    "Mã phim": st.column_config.NumberColumn(
                        "Mã phim",
                        format="%d",
                        width="small",
                    ),
                    "Điểm đã chấm": st.column_config.NumberColumn(
                        "Điểm đã chấm (sao)",
                        format="%.1f",
                        width="small",
                    ),
                },
                use_container_width=True,
                height=350,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 5: KHẢO SÁT MỞ RỘNG (SCALABILITY)
    # =========================================================================
    with tabs[4]:
        scalability_df = static_results.get("scalability_df")
        if scalability_df is not None and not scalability_df.empty:
            st.markdown(
                """<div class="pbi-panel">
<div class="pbi-panel-title">HIỆU NĂNG MỞ RỘNG TRÊN CỤM PHÂN TÁN SPARK & HDFS (100K → 10M → 25M)</div>""",
                unsafe_allow_html=True,
            )
            disp_sc = scalability_df.copy()
            disp_sc.columns = ["Quy mô Dataset", "Số lượng Ratings", "Thời gian huấn luyện", "Test RMSE", "Test MAE"]
            disp_sc["Quy mô Dataset"] = disp_sc["Quy mô Dataset"].str.upper()

            st.dataframe(
                disp_sc.style.format({
                    "Số lượng Ratings": "{:,}",
                    "Thời gian huấn luyện": "{:.1f}s",
                    "Test RMSE": "{:.4f}",
                    "Test MAE": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        sc_p1, sc_p2 = st.columns(2)
        time_img_path = RESULTS_DIR / "scalability_time.png"
        rmse_img_path = RESULTS_DIR / "scalability_rmse.png"

        with sc_p1:
            if time_img_path.exists():
                st.markdown(
                    """<div class="pbi-panel">
<div class="pbi-panel-title">BIỂU ĐỒ THỜI GIAN HUẤN LUYỆN (SUB-LINEAR SCALING)</div>""",
                    unsafe_allow_html=True,
                )
                st.image(str(time_img_path), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with sc_p2:
            if rmse_img_path.exists():
                st.markdown(
                    """<div class="pbi-panel">
<div class="pbi-panel-title">BIỂU ĐỒ SAI SỐ DỰ ĐOÁN RMSE THEO QUY MÔ</div>""",
                    unsafe_allow_html=True,
                )
                st.image(str(rmse_img_path), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """<div style="background: #ffffff; border-radius: 12px; padding: 1.15rem 1.35rem; border-left: 4px solid #5E48E8; border: 1px solid #CBD1E9; margin-top: 0.6rem;">
<div style="font-weight: 700; font-size: 0.92rem; color: #1E1B4B; margin-bottom: 0.45rem;">
BÁO CÁO PHÂN TÍCH HIỆU NĂNG MỞ RỘNG (EXECUTIVE SCALABILITY SUMMARY)
</div>
<div style="font-size: 0.82rem; color: #1E1B4B; line-height: 1.6;">
<b>1. Tốc độ Tăng trưởng Thời gian Dưới Tuyến tính (Sub-linear Compute Scalability):</b><br>
Khi dữ liệu tương tác mở rộng từ 100,000 lên 25,000,095 ratings (tăng gấp <b>250 lần</b>), thời gian huấn luyện ALS phân tán trên cụm Spark chỉ tăng từ 11.7 giây lên 761.6 giây (chỉ tăng <b>65.3 lần</b>). Cơ chế phân vùng khối Block Matrix Partitioning đã phân bổ bộ nhớ đồng đều giữa các worker và tránh được hiện tượng nghẽn I/O.<br><br>
<b>2. Lý giải Hiện tượng RMSE ở 10M (0.7788) thấp hơn ở 25M (0.8455):</b><br>
Ở mốc 10M, dữ liệu tập trung chủ yếu vào ~10,000 bộ phim phổ biến với mật độ tương tác trung bình ~1,000 ratings/phim (ma trận dày hơn, bài toán dự đoán ít bị phân tán). Khi mở rộng lên 25M, danh mục phim tăng vọt lên <b>62,423 bộ phim</b> (gấp hơn 6 lần). Phần lớn là các phim nghệ thuật, phim độc lập (Long-tail) có lượng rating rất thưa thớt, đưa độ thưa ma trận lên tới <b>99.75%</b>. Sai số RMSE ở 25M tăng nhẹ lên 0.8455 nhưng vẫn vượt trội hoàn toàn so với Baseline (1.0097) và mang lại giá trị thực tiễn cao hơn nhờ độ bao phủ danh mục phim toàn diện.
</div>
</div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
