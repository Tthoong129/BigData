import os
import re
import json
import ssl
import urllib.request
import concurrent.futures
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

cache_file = os.path.join(DATA_DIR, "movie_posters_cache.json")
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

print(f"[INFO] Current cache has {len(poster_cache)} posters.")

# Load links and stats
links_df = pd.read_csv(os.path.join(DATA_DIR, "links.csv")).set_index("movieId")
movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

with open(os.path.join(RESULTS_DIR, "movie_popularity_25m.json"), "r", encoding="utf-8") as f:
    pop_data = json.load(f)

movie_stats = pop_data.get("movie_stats", {})
movies_df["count"] = movies_df["movieId"].apply(lambda x: movie_stats.get(str(x), {}).get("count", 0))
movies_df["avg"] = movies_df["movieId"].apply(lambda x: movie_stats.get(str(x), {}).get("avg", 0.0))

def extract_year(t):
    m = re.search(r'\((\d{4})\)', str(t))
    return int(m.group(1)) if m else 2000

movies_df["year_int"] = movies_df["title"].apply(extract_year)

# 1. Top 400 popular
top_pop = movies_df.sort_values(by="count", ascending=False).head(400)["movieId"].tolist()

# 2. Top 100 high rated (>= 5000 ratings)
top_rated = movies_df[movies_df["count"] >= 5000].sort_values(by=["avg", "count"], ascending=[False, False]).head(100)["movieId"].tolist()

# 3. Top 60 recent (>= 2015, count >= 1000)
recent = movies_df[(movies_df["year_int"] >= 2015) & (movies_df["count"] >= 1000)].sort_values(by="count", ascending=False).head(60)["movieId"].tolist()

# 4. Top 60 classic (<= 1985, count >= 2000, avg >= 4.0)
classics = movies_df[(movies_df["year_int"] <= 1985) & (movies_df["count"] >= 2000) & (movies_df["avg"] >= 4.0)].sort_values(by="count", ascending=False).head(60)["movieId"].tolist()

# 5. Top 60 hidden gems
hidden = movies_df[(movies_df["avg"] >= 4.1) & (movies_df["count"] >= 500) & (movies_df["count"] <= 4000)].sort_values(by="count", ascending=False).head(60)["movieId"].tolist()

# 6. Top 20 for each genre
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
genre_tops = []
for g in genres:
    gt = movies_df[movies_df["genres"].str.contains(g, na=False)].sort_values(by="count", ascending=False).head(20)["movieId"].tolist()
    genre_tops.extend(gt)

all_ids = list(dict.fromkeys(top_pop + top_rated + recent + classics + hidden + genre_tops))
missing_ids = [m_id for m_id in all_ids if str(m_id) not in poster_cache and m_id in links_df.index]
print(f"[INFO] Fetching real posters for {len(missing_ids)} additional movies (out of {len(all_ids)} key movies)...")

ctx = ssl._create_unverified_context()

def fetch_poster(m_id):
    try:
        imdb_val = links_df.loc[m_id, "imdbId"]
        if pd.isna(imdb_val):
            return m_id, None
        imdb_code = f"tt{str(int(imdb_val)).zfill(7)}"
        url = f"http://www.omdbapi.com/?i={imdb_code}&apikey=trilogy"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            poster = data.get("Poster")
            if poster and poster != "N/A" and poster.startswith("http"):
                return m_id, poster
    except Exception:
        pass
    return m_id, None

success_count = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    results = executor.map(fetch_poster, missing_ids)
    for m_id, poster_url in results:
        if poster_url:
            poster_cache[str(m_id)] = poster_url
            success_count += 1

print(f"[OK] Fetched {success_count} new posters. Total cache now: {len(poster_cache)} movies.")

with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(poster_cache, f, ensure_ascii=False, indent=2)

print(f"[OK] Successfully saved poster cache to {cache_file}.")
