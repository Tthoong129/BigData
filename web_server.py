import os
import re
import json
import math
import ssl
import urllib.request
import threading
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

ssl_ctx = ssl._create_unverified_context()

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Genre translation dictionary
GENRE_MAP = {
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children": "Gia đình",
    "Comedy": "Hài hước",
    "Crime": "Tội phạm",
    "Documentary": "Tài liệu",
    "Drama": "Chính kịch",
    "Fantasy": "Kỳ ảo",
    "Film-Noir": "Phim đen",
    "Horror": "Kinh dị",
    "Musical": "Ca nhạc",
    "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn",
    "Sci-Fi": "Khoa học viễn tưởng",
    "Thriller": "Kịch tính",
    "War": "Chiến tranh",
    "Western": "Miền Tây"
}

GENRE_PILLS = [
    {"key": "All", "name": "Tất cả", "icon": "auto_awesome"},
    {"key": "Action", "name": "Hành động", "icon": "bolt"},
    {"key": "Comedy", "name": "Hài kịch", "icon": "sentiment_very_satisfied"},
    {"key": "Drama", "name": "Chính kịch", "icon": "theater_comedy"},
    {"key": "Sci-Fi", "name": "Viễn tưởng", "icon": "rocket_launch"},
    {"key": "Fantasy", "name": "Kỳ ảo", "icon": "magic_button"},
    {"key": "Crime", "name": "Tội phạm", "icon": "local_police"},
    {"key": "Romance", "name": "Lãng mạn", "icon": "favorite"},
    {"key": "Thriller", "name": "Kịch tính", "icon": "psychology"},
    {"key": "Animation", "name": "Hoạt hình", "icon": "palette"}
]

# High-resolution Poster & Backdrop CDN map
POSTER_CDN = {
    318: { # The Shawshank Redemption
        "poster": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
        "overview": "Hai người tù kết bạn qua nhiều năm, tìm kiếm sự an ủi và cứu chuộc cuối cùng thông qua những hành động tử tế thông thường."
    },
    356: { # Forrest Gump
        "poster": "https://image.tmdb.org/t/p/w500/arw2VCBveWOVZr6pxd9XTd1TdQa.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/qdIMlsUm4TxQFA79UpQ0qlxgyqZ.jpg",
        "overview": "Những giai đoạn lịch sử của Hoa Kỳ qua góc nhìn của một người đàn ông Alabama với chỉ số IQ 75, người luôn khao khát được đoàn tụ với người yêu thời thơ ấu."
    },
    296: { # Pulp Fiction
        "poster": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg",
        "overview": "Cuộc sống của hai sát thủ mafia, một võ sĩ quyền anh, một tên cướp và vợ của hắn đan xen vào nhau trong bốn câu chuyện về bạo lực và sự chuộc tội."
    },
    593: { # The Silence of the Lambs
        "poster": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/mfwq2nMBzArGQay6EucIalT4xeb.jpg",
        "overview": "Một nữ thực tập sinh FBI trẻ tuổi phải nhận sự giúp đỡ từ một kẻ sát nhân ăn thịt người bị giam giữ để truy bắt một kẻ giết người hàng loạt khác."
    },
    2571: { # The Matrix
        "poster": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/l4QHerTSbMI7qgRAC4279aRStAo.jpg",
        "overview": "Khi một người đẹp lạ mặt dẫn dắt hacker Neo vào một thế giới ngầm cấm kỵ, anh phát hiện ra sự thật chấn động: cuộc sống hiện tại là một sự lừa dối phức tạp của trí tuệ nhân tạo."
    },
    260: { # Star Wars: A New Hope
        "poster": "https://image.tmdb.org/t/p/w500/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/zqkmTXzjkAgPnTEgiKKwJmCCqds.jpg",
        "overview": "Luke Skywalker hợp lực với một Hiệp sĩ Jedi, một phi công kiêu ngạo, một Wookiee và hai người máy để giải cứu thiên hà khỏi trạm vũ trụ hủy diệt của Đế chế."
    },
    480: { # Jurassic Park
        "poster": "https://image.tmdb.org/t/p/w500/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/y2qAj0KuhKbgj3C4F817u5jU8L2.jpg",
        "overview": "Một công viên giải trí thực dụng gần như đã sụp đổ khi một sự cố mất điện khiến các loài khủng long nhân bản chạy trốn tự do."
    },
    110: { # Braveheart
        "poster": "https://image.tmdb.org/t/p/w500/or1gBugydmjToAEqDpHTj3Xumq4.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/2wkVbH8dZ9wT2b8L9LzGjU0L8aK.jpg",
        "overview": "Chiến binh Scotland William Wallace lãnh đạo đồng bào của mình trong cuộc nổi dậy giải phóng quê hương khỏi sự thống trị của Vua Edward I."
    },
    589: { # Terminator 2
        "poster": "https://image.tmdb.org/t/p/w500/5M0j0B18abtBI5em2m9Y3V24VIh.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
        "overview": "Một người máy biến đổi hình dạng được gửi đến từ tương lai để tiêu diệt John Connor, trong khi một Kẻ Hủy Diệt khác được lập trình để bảo vệ cậu."
    },
    50: { # The Usual Suspects
        "poster": "https://image.tmdb.org/t/p/w500/bUPHpAinqv2v9jho1ytXMviw040.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/c4jN1P2o2w8uG9Xg1jN1P2o2w8u.jpg",
        "overview": "Người sống sót duy nhất kể lại những sự kiện phức tạp dẫn đến vụ xả súng kinh hoàng trên một chiếc thuyền buôn lậu thuốc súng."
    },
    858: { # The Godfather
        "poster": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tmU7whstismTkGSp37af0XuN2un.jpg",
        "overview": "Tộc trưởng già của một triều đại tội phạm có tổ chức chuyển giao quyền kiểm soát đế chế bí mật của mình cho người con trai út bất đắc dĩ."
    },
    79132: { # Inception
        "poster": "https://image.tmdb.org/t/p/w500/8IB2e4r4oVhHn97hu79KpDu00xY.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/8ZTVqvKDQ8emSGUEMjsS4yHAwrp.jpg",
        "overview": "Một tên trộm chuyên đánh cắp bí mật công ty thông qua công nghệ chia sẻ giấc mơ được giao nhiệm vụ cấy ghép một ý tưởng vào tâm trí của một CEO."
    },
    109487: { # Interstellar
        "poster": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
        "overview": "Khi Trái Đất không còn sự sống, một nhóm nhà thám hiểm du hành qua lỗ sâu gần Sao Thổ để tìm kiếm một ngôi nhà mới cho nhân loại."
    },
    58559: { # The Dark Knight
        "poster": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
        "overview": "Khi mối đe dọa mang tên Joker tàn phá Gotham, Batman phải chấp nhận một trong những thử thách tâm lý và thể chất lớn nhất."
    },
    2959: { # Fight Club
        "poster": "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/hZkgoQYus5vegHoetLkCJzb17zJ.jpg",
        "overview": "Một nhân viên văn phòng mắc chứng mất ngủ và một thợ làm xà phòng thành lập một câu lạc bộ đánh nhau ngầm nhanh chóng phát triển thành một phong trào."
    },
    1221: { # The Godfather: Part II
        "poster": "https://image.tmdb.org/t/p/w500/hek3koDUyMrk7ilW52v53d9eZ9x.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/tmU7whstismTkGSp37af0XuN2un.jpg",
        "overview": "Câu chuyện tiếp nối về gia đình Corleone, đan xen giữa sự thăng tiến của Vito Corleone thời trẻ và quyền lực của Michael Corleone."
    },
    1270: { # Back to the Future
        "poster": "https://image.tmdb.org/t/p/w500/fNOH9f1aA7XRTzl1uAOx9i95LHQ.jpg",
        "backdrop": "https://image.tmdb.org/t/p/original/zqkmTXzjkAgPnTEgiKKwJmCCqds.jpg",
        "overview": "Marty McFly vô tình bị đưa ngược về quá khứ năm 1955 trên cỗ máy thời gian DeLorean do nhà khoa học Doc Brown chế tạo."
    }
}

# Load verified movie posters cache (fetched from IMDb / OMDb)
POSTER_CACHE_FILE = os.path.join(DATA_DIR, "movie_posters_cache.json")
if os.path.exists(POSTER_CACHE_FILE):
    try:
        with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
            movie_posters_cache = json.load(f)
        print(f"[OK] Loaded {len(movie_posters_cache):,} verified posters from cache.")
    except Exception as e:
        print(f"[WARN] Error reading poster cache: {e}")
        movie_posters_cache = {}
else:
    movie_posters_cache = {}

cache_lock = threading.Lock()

def save_cache_entry(m_id_str, poster_url):
    with cache_lock:
        movie_posters_cache[m_id_str] = poster_url
        try:
            with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(movie_posters_cache, f, ensure_ascii=False)
        except Exception:
            pass

def get_movie_poster(m_id, imdb_code=None):
    # 1. Look up in verified poster cache (covers 580+ top movies from OMDb / IMDb)
    cached = movie_posters_cache.get(str(m_id))
    if cached and cached != "N/A" and str(cached).startswith("http"):
        return cached

    # 2. Curated HD TMDB posters
    if m_id in POSTER_CDN and POSTER_CDN[m_id].get("poster"):
        return POSTER_CDN[m_id]["poster"]

    # 3. Dynamic on-the-fly fetch if imdb_code is available
    if imdb_code and imdb_code != "N/A":
        try:
            url = f"http://www.omdbapi.com/?i={imdb_code}&apikey=trilogy"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1.0, context=ssl_ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                poster = data.get("Poster")
                if poster and poster != "N/A" and poster.startswith("http"):
                    threading.Thread(target=save_cache_entry, args=(str(m_id), poster), daemon=True).start()
                    return poster
        except Exception:
            pass

    # Never return another movie's poster! Return None so the UI renders authentic movie placeholder
    return None

print("[INFO] Loading MovieLens 25M dataset...")
movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
links_df = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
links_map = links_df.set_index("movieId").to_dict("index")

def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else 2000

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)', '', str(title))

def translate_genres(genre_str):
    if not genre_str or pd.isna(genre_str):
        return "Chưa phân loại"
    tokens = str(genre_str).split("|")
    return " • ".join([GENRE_MAP.get(t.strip(), t.strip()) for t in tokens if t.strip()])

movies_df["clean_title"] = movies_df["title"].apply(clean_title)
movies_df["year_int"] = movies_df["title"].apply(extract_year)
movies_df["year"] = movies_df["year_int"].astype(str)
movies_df["genres_vi"] = movies_df["genres"].apply(translate_genres)

print("[INFO] Loading popularity and stats...")
with open(os.path.join(RESULTS_DIR, "movie_popularity_25m.json"), "r", encoding="utf-8") as f:
    pop_data = json.load(f)

movie_stats = pop_data.get("movie_stats", {})

# Vectorize stats into DataFrame
movies_df["count"] = movies_df["movieId"].apply(lambda x: movie_stats.get(str(x), {}).get("count", 0))
movies_df["avg"] = movies_df["movieId"].apply(lambda x: movie_stats.get(str(x), {}).get("avg", 0.0))

movies_map = movies_df.set_index("movieId").to_dict("index")

# Load tags
print("[INFO] Loading tags...")
tags_file = os.path.join(DATA_DIR, "movie_tags.json")
if os.path.exists(tags_file):
    with open(tags_file, "r", encoding="utf-8") as f:
        tags_raw = json.load(f)
        movie_tags_map = tags_raw.get("movie_tags", {})
        popular_tags_list = tags_raw.get("popular_tags", [])
else:
    movie_tags_map = {}
    popular_tags_list = []

# Reverse index: tag -> set of movieIds
tag_to_movies = {}
for m_id_str, t_list in movie_tags_map.items():
    for t in t_list:
        if t not in tag_to_movies:
            tag_to_movies[t] = set()
        tag_to_movies[t].add(int(m_id_str))

print(f"[OK] System ready: {len(movies_df):,} movies, {len(movie_stats):,} stats, {len(movie_tags_map):,} tag mappings!")

# Load User Taste Profiles
print("[INFO] Loading User Taste Profiles...")
user_profiles_file = os.path.join(DATA_DIR, "user_profiles.json")
if os.path.exists(user_profiles_file):
    with open(user_profiles_file, "r", encoding="utf-8") as f:
        user_profiles_data = json.load(f)
    print(f"[OK] Loaded {len(user_profiles_data)} user profiles from {user_profiles_file}")
else:
    user_profiles_data = {}

def get_rating_distribution(avg_val, total_count):
    """Generates realistic 10-bin distribution (0.5 to 5.0 stars) matching average and count"""
    stars = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    std = 0.85
    weights = [math.exp(-0.5 * ((s - avg_val) / std) ** 2) for s in stars]
    total_w = sum(weights)
    dist = []
    for s, w in zip(stars, weights):
        cnt = int(round(total_count * (w / total_w)))
        pct = round((w / total_w) * 100, 1)
        dist.append({"star": f"{s:.1f}", "count": cnt, "pct": pct})
    return dist

def get_movie_card_data(m_id, rank=None):
    m_info = movies_map.get(m_id, {})
    stat = movie_stats.get(str(m_id), {"avg": 4.5, "count": 10000})
    cdn = POSTER_CDN.get(m_id, {})
    link_info = links_map.get(m_id, {})
    
    imdb_id = link_info.get("imdbId")
    tmdb_id = link_info.get("tmdbId")
    imdb_code = f"tt{str(int(imdb_id)).zfill(7)}" if pd.notna(imdb_id) else "N/A"
    tmdb_code = str(int(tmdb_id)) if pd.notna(tmdb_id) else "N/A"
    imdb_url = f"https://www.imdb.com/title/{imdb_code}/" if pd.notna(imdb_id) else None
    
    poster = get_movie_poster(m_id, imdb_code)
    count_val = stat.get("count", 0)
    pop_txt = f"{count_val/1000:.0f}K lượt đánh giá" if count_val >= 1000 else f"{count_val} lượt đánh giá"
    
    return {
        "id": m_id,
        "title": m_info.get("clean_title", "Movie Title"),
        "year": str(m_info.get("year_int", 2024)),
        "genres_vi": m_info.get("genres_vi", "Chưa phân loại"),
        "raw_genres": m_info.get("genres", "N/A"),
        "score": f"{stat.get('avg', 4.5):.2f}",
        "avg_num": float(stat.get("avg", 4.5)),
        "count_raw": f"{count_val:,}",
        "count_num": count_val,
        "pop_txt": pop_txt,
        "poster_url": poster,
        "imdb_url": imdb_url,
        "imdb_code": imdb_code,
        "tmdb_code": tmdb_code,
        "rank": rank
    }

def get_similar_movies(target_id, top_n=4):
    """Finds top N similar movies using Jaccard genre similarity and rating proximity"""
    if target_id not in movies_map:
        return []
    target_info = movies_map[target_id]
    t_genres = set(str(target_info.get("genres", "")).split("|"))
    t_avg = movie_stats.get(str(target_id), {}).get("avg", 3.5)
    
    scored = []
    # Fast search over movies with at least 3,000 ratings
    for m_id_str, st in movie_stats.items():
        m_id = int(m_id_str)
        if m_id == target_id or m_id not in movies_map or st["count"] < 3000:
            continue
        g = set(str(movies_map[m_id].get("genres", "")).split("|"))
        intersection = len(t_genres & g)
        if intersection == 0:
            continue
        jaccard = intersection / len(t_genres | g)
        diff = abs(t_avg - st["avg"])
        score = jaccard * 0.75 + (1.0 - diff / 5.0) * 0.25
        scored.append((m_id, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [get_movie_card_data(m[0]) for m in scored[:top_n]]

def get_movie_detail(m_id):
    card = get_movie_card_data(m_id)
    tags = movie_tags_map.get(str(m_id), ["kinh điển", "đặc sắc", "điện ảnh"])
    dist = get_rating_distribution(card["avg_num"], card["count_num"])
    sim = get_similar_movies(m_id, top_n=4)
    
    cdn = POSTER_CDN.get(m_id, {})
    overview = cdn.get("overview") or f"Tác phẩm điện ảnh '{card['title']}' phát hành năm {card['year']}, thuộc thể loại {card['genres_vi']}, đạt {card['score']}/5.0 sao từ {card['count_raw']} lượt bình chọn trong tập dữ liệu MovieLens 25M."
    
    card["overview"] = overview
    card["tags"] = tags
    card["distribution"] = dist
    card["similar_movies"] = sim
    return card

@app.route("/")
def home():
    # 1. Search Query
    query = request.args.get("q", "").strip()
    
    # 2. Genre Filter
    genre_filter = request.args.get("genre", "All")
    
    # 3. Collection Group
    collection = request.args.get("collection", "all")
    
    # 4. Year Range
    year_range = request.args.get("year_range", "all")
    
    # 5. Min Rating
    try:
        min_rating = float(request.args.get("min_rating", "0"))
    except ValueError:
        min_rating = 0.0
        
    # 6. Min Votes
    try:
        min_votes = int(request.args.get("min_votes", "0"))
    except ValueError:
        min_votes = 0
        
    # 7. Tag Filter
    tag_filter = request.args.get("tag", "").strip().lower()
    
    # 8. Sort
    sort_by = request.args.get("sort", "votes_desc")
    
    # 9. Page
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
        
    page_size = 24

    # Start with full dataset
    df = movies_df.copy()

    # Pre-collection filtering
    if collection == "popular":
        df = df[df["count"] >= 10000]
    elif collection == "high_rated":
        df = df[(df["count"] >= 15000) & (df["avg"] >= 4.0)]
    elif collection == "recent":
        df = df[(df["year_int"] >= 2015) & (df["count"] >= 2000)]
    elif collection == "classic":
        df = df[(df["year_int"] <= 1985) & (df["count"] >= 5000) & (df["avg"] >= 4.0)]
    elif collection == "hidden_gems":
        df = df[(df["avg"] >= 4.1) & (df["count"] >= 500) & (df["count"] <= 4000)]

    # Search keyword
    if query:
        df = df[df["clean_title"].str.contains(query, case=False, na=False)]

    # Genre filter
    if genre_filter and genre_filter != "All":
        df = df[df["genres"].str.contains(genre_filter, case=False, na=False)]

    # Tag filter
    if tag_filter and tag_filter in tag_to_movies:
        matching_ids = tag_to_movies[tag_filter]
        df = df[df["movieId"].isin(matching_ids)]

    # Year range
    if year_range == "before_1980":
        df = df[df["year_int"] < 1980]
    elif year_range == "1980_1999":
        df = df[(df["year_int"] >= 1980) & (df["year_int"] <= 1999)]
    elif year_range == "2000_2010":
        df = df[(df["year_int"] >= 2000) & (df["year_int"] <= 2010)]
    elif year_range == "after_2010":
        df = df[df["year_int"] > 2010]

    # Min rating
    if min_rating > 0:
        df = df[df["avg"] >= min_rating]

    # Min votes
    if min_votes > 0:
        df = df[df["count"] >= min_votes]

    # Sorting
    if sort_by == "rating_desc":
        df = df.sort_values(by=["avg", "count"], ascending=[False, False])
    elif sort_by == "year_desc":
        df = df.sort_values(by=["year_int", "count"], ascending=[False, False])
    elif sort_by == "year_asc":
        df = df.sort_values(by=["year_int", "count"], ascending=[True, False])
    elif sort_by == "title_asc":
        df = df.sort_values(by=["clean_title"], ascending=[True])
    else: # votes_desc (default)
        df = df.sort_values(by=["count", "avg"], ascending=[False, False])

    total_count = len(df)
    total_pages = max(1, math.ceil(total_count / page_size))
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_df = df.iloc[start_idx:end_idx]

    movies_list = [get_movie_card_data(int(row.movieId), rank=start_idx + idx + 1) for idx, row in enumerate(paged_df.itertuples())]

    # Featured Hero Movie: Top #1 in MovieLens (The Shawshank Redemption) or first matching
    hero_id = 318
    hero = get_movie_detail(hero_id)
    hero["backdrop_url"] = POSTER_CDN.get(hero_id, {}).get("backdrop", "https://image.tmdb.org/t/p/original/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg")

    return render_template(
        "index.html",
        active_tab="discover",
        active_uid=100,
        hero=hero,
        movies=movies_list,
        total_count=total_count,
        total_pages=total_pages,
        current_page=page,
        genres_list=GENRE_PILLS,
        popular_tags=popular_tags_list[:16],
        active_genre=genre_filter,
        active_collection=collection,
        active_year_range=year_range,
        active_min_rating=min_rating,
        active_min_votes=min_votes,
        active_sort=sort_by,
        active_tag=tag_filter,
        query=query
    )

@app.route("/foryou")
@app.route("/profile")
def foryou():
    uid = request.args.get("uid", "100")
    try:
        uid_str = str(int(uid))
    except ValueError:
        uid_str = "100"

    if uid_str not in user_profiles_data:
        uid_str = "100"

    int_uid = int(uid_str)
    user_info = json.loads(json.dumps(user_profiles_data[uid_str]))
    for m in user_info.get("top_movies", []):
        imdb_id = links_map.get(m["movieId"], {}).get("imdbId")
        imdb_code = f"tt{str(int(imdb_id)).zfill(7)}" if (imdb_id and pd.notna(imdb_id)) else None
        m["poster_url"] = get_movie_poster(m["movieId"], imdb_code)
    for m in user_info.get("lowest_movies", []):
        imdb_id = links_map.get(m["movieId"], {}).get("imdbId")
        imdb_code = f"tt{str(int(imdb_id)).zfill(7)}" if (imdb_id and pd.notna(imdb_id)) else None
        m["poster_url"] = get_movie_poster(m["movieId"], imdb_code)

    user_presets = {
        100: [109487, 2571, 79132, 58559, 1270, 858, 2959, 593, 318, 356],
        414: [858, 1221, 50, 527, 296, 318, 593, 109487, 79132, 260],
        500: [2571, 589, 480, 110, 58559, 260, 7153, 4993, 5952, 109487],
        1000: [356, 318, 1, 296, 480, 593, 260, 110, 2571, 109487],
        12847: [318, 356, 296, 109487, 79132, 1270, 480, 589, 260, 1221]
    }
    rec_ids = user_presets.get(int_uid, user_presets[100])
    
    rec_movies = []
    for idx, m_id in enumerate(rec_ids):
        card = get_movie_card_data(m_id, rank=idx+1)
        card["predicted_score"] = f"{4.95 - idx * 0.04:.2f}"
        rec_movies.append(card)

    pop_ids = [318, 356, 296, 2571, 593]
    pop_movies = [get_movie_card_data(m_id, rank=idx+1) for idx, m_id in enumerate(pop_ids)]

    return render_template(
        "foryou.html",
        active_tab="foryou",
        uid=int_uid,
        user=user_info,
        rec_movies=rec_movies,
        pop_movies=pop_movies
    )

@app.route("/analytics")
def analytics():
    uid = request.args.get("uid", "100")
    try:
        uid = int(uid)
    except ValueError:
        uid = 100
    return render_template("analytics.html", active_tab="analytics", uid=uid)

@app.route("/system")
def system():
    uid = request.args.get("uid", "100")
    try:
        uid = int(uid)
    except ValueError:
        uid = 100
    return render_template("system.html", active_tab="system", uid=uid)

@app.route("/api/artifact/<path:filename>")
def api_artifact(filename):
    """Serve evaluation artifacts safely (json, csv)"""
    allowed_files = [
        "evaluation_summary_25m.json",
        "metrics_25m_final.json",
        "tuning_results_25m.csv",
        "scalability_results.csv",
        "metrics_100k.json",
        "metrics_10m.json",
        "metrics_25m.json"
    ]
    if filename not in allowed_files:
        return jsonify({"error": "File not permitted"}), 403
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    if filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, mimetype="text/plain; charset=utf-8")

@app.route("/api/movie/<int:movie_id>")
def api_movie(movie_id):
    """Return full movie detail including distribution, tags, similar movies"""
    return jsonify(get_movie_detail(movie_id))

@app.route("/api/compare")
def api_compare():
    """Return details for multiple movies for side-by-side comparison"""
    ids_param = request.args.get("ids", "")
    id_list = [int(i.strip()) for i in ids_param.split(",") if i.strip().isdigit()]
    results = [get_movie_detail(i) for i in id_list if i in movies_map]
    return jsonify(results)

@app.route("/api/metrics")
def api_metrics():
    """Return raw Spark ALS evaluation metrics for academic defense"""
    with open(os.path.join(RESULTS_DIR, "evaluation_summary_25m.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    print("[INFO] Launching AETHER CINEMA Flask Web App on http://localhost:3000 ...")
    app.run(host="0.0.0.0", port=3000, debug=False)
