"""Loading, cleaning and caching of the Maven Analytics Airbnb dataset (with a synthetic fallback)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import os

ROOT = Path(__file__).resolve().parents[2]  # project root (parent of backend/)
DATA_DIR = Path(os.environ.get("AIRBNB_DATA_DIR", ROOT / "data"))
CACHE = DATA_DIR / ".cache.pkl"
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"))

WANTED = {
    "listing_id", "city", "room_type", "property_type", "price", "accommodates", "bedrooms", "minimum_nights",
    "amenities", "review_scores_rating", "review_scores_value", "host_is_superhost", "instant_bookable",
    "latitude", "longitude", "host_total_listings_count", "host_response_rate",
}


def _find(prefix: str) -> Path | None:
    files = sorted(p for p in DATA_DIR.glob("*.csv") if p.name.lower().startswith(prefix)) if DATA_DIR.is_dir() else []
    return files[0] if files else None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce").astype("float64")


def _bool(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower().isin(["t", "true", "1", "yes"])


def _scale01(x: pd.Series) -> pd.Series:
    """Ratings come as 0-5, 0-10 or 0-100 depending on the export: detect from the max and map to 0..1."""
    m = x.max()
    return x / (100 if m > 10 else 10 if m > 5 else 5)


def _derive(L: pd.DataFrame) -> pd.DataFrame:
    L["log_price"] = np.log(L["price"])
    L["price_per_guest"] = L["price"] / L["accommodates"]
    L["log_min_nights"] = np.log1p(L["min_nights"])
    L["log_host_listings"] = np.log1p(L["host_listings"])
    return L


def load_data(max_price: float = 10_000.0):
    """Returns (listings, monthly_reviews, source). monthly_reviews has columns city, month, reviews."""
    lp, rp = _find("listings"), _find("reviews")
    if lp is None:
        L, monthly = synthetic()
        return _derive(L), monthly, "synthetic"
    key = (lp.name, lp.stat().st_mtime, rp.name if rp else None, rp.stat().st_mtime if rp else None, max_price)
    if CACHE.exists():
        try:
            k, L, monthly = pd.read_pickle(CACHE)
            if k == key:
                return L, monthly, "csv"
        except Exception:
            pass
    L = _load_listings(lp, max_price)
    monthly = pd.DataFrame(columns=["city", "month", "reviews"])
    if rp is not None:
        monthly, per = _load_reviews(rp, L.set_index("listing_id")["city"])
        L["n_reviews"] = L["listing_id"].map(per).fillna(0)
    L = _derive(L)
    pd.to_pickle((key, L, monthly), CACHE)
    return L, monthly, "csv"


def _load_listings(path: Path, max_price: float) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=lambda c: c.strip().lower() in WANTED, low_memory=False, encoding_errors="replace")
    raw.columns = [c.strip().lower() for c in raw.columns]
    for c in WANTED:
        if c not in raw:
            raw[c] = np.nan
    raw["price"] = _num(raw["price"])
    raw["listing_id"] = _num(raw["listing_id"])
    raw = raw[(raw["price"] > 0) & (raw["price"] <= max_price) & raw["city"].notna() & raw["listing_id"].notna()]
    am = raw["amenities"].fillna("").astype(str)
    n_am = am.str.count(",") + 1
    n_am[am.str.strip().isin(["", "[]", "{}"])] = 0
    resp = _num(raw["host_response_rate"])
    resp = resp.where(resp <= 1, resp / 100)
    prop = raw["property_type"].fillna("Unknown").astype(str)
    top = prop.value_counts().index[:8]
    L = pd.DataFrame({
        "listing_id": raw["listing_id"].astype("int64"),
        "city": raw["city"].astype(str).str.strip(),
        "room_type": raw["room_type"].fillna("Unknown").astype(str),
        "property_type": prop.where(prop.isin(top), "Other"),
        "price": raw["price"],
        "accommodates": _num(raw["accommodates"]).fillna(1).clip(lower=1),
        "bedrooms": _num(raw["bedrooms"]),
        "min_nights": _num(raw["minimum_nights"]).fillna(1),
        "amenities": n_am,
        "rating": _scale01(_num(raw["review_scores_rating"])),
        "value": _scale01(_num(raw["review_scores_value"])),
        "superhost": _bool(raw["host_is_superhost"]),
        "instant": _bool(raw["instant_bookable"]),
        "lat": _num(raw["latitude"]),
        "lon": _num(raw["longitude"]),
        "host_listings": _num(raw["host_total_listings_count"]).fillna(1),
        "response_rate": resp,
        "n_reviews": 0.0,
    })
    return L.reset_index(drop=True)


def _load_reviews(path: Path, city_by_id: pd.Series):
    """Stream the (multi-million row) reviews file in chunks and keep only monthly counts."""
    parts, per = [], pd.Series(dtype="float64")
    for ch in pd.read_csv(path, usecols=lambda c: c.strip().lower() in {"listing_id", "date"},
                          chunksize=1_000_000, low_memory=False, encoding_errors="replace"):
        ch.columns = [c.strip().lower() for c in ch.columns]
        ch["listing_id"] = pd.to_numeric(ch["listing_id"], errors="coerce")
        ch = ch.dropna(subset=["listing_id"])
        ch["listing_id"] = ch["listing_id"].astype("int64")
        ch["city"] = ch["listing_id"].map(city_by_id)
        ch["month"] = pd.to_datetime(ch["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        ch = ch.dropna(subset=["city", "month"])
        parts.append(ch.groupby(["city", "month"]).size())
        per = per.add(ch.groupby("listing_id").size(), fill_value=0)
    if not parts:
        return pd.DataFrame(columns=["city", "month", "reviews"]), per
    monthly = pd.concat(parts).groupby(level=[0, 1]).sum().rename("reviews").reset_index()
    return monthly, per


def synthetic(seed: int = 42, n: int = 1200):
    """Generated demo data with realistic structure (price drivers, seasonality, COVID dip)."""
    rng = np.random.default_rng(seed)
    cities = [("Bangkok", 13.75, 100.5, 45, 12), ("Cape Town", -33.92, 18.42, 60, 1), ("Hong Kong", 22.3, 114.17, 80, 10),
              ("Istanbul", 41.01, 28.97, 55, 7), ("Mexico City", 19.43, -99.13, 50, 3), ("New York", 40.71, -74.0, 150, 8),
              ("Paris", 48.86, 2.35, 130, 6), ("Rio de Janeiro", -22.9, -43.2, 70, 2), ("Rome", 41.9, 12.5, 110, 5),
              ("Sydney", -33.87, 151.2, 135, 12)]
    rooms = np.array(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    mult = {"Entire home/apt": 1.0, "Private room": .55, "Shared room": .3, "Hotel room": 1.1}
    props = np.array(["Apartment", "House", "Condominium", "Loft", "Guest suite"])
    months = pd.date_range("2016-01-01", "2024-06-01", freq="MS")
    frames, monthly = [], []
    for name, lat, lon, base, peak in cities:
        room = rng.choice(rooms, n, p=[.62, .31, .04, .03])
        entire = room == "Entire home/apt"
        acc = np.where(entire, rng.integers(2, 8, n), rng.integers(1, 4, n))
        am = rng.integers(5, 45, n)
        sh = rng.random(n) < .2
        dlat, dlon = rng.normal(0, .08, n), rng.normal(0, .08, n)
        price = (base * pd.Series(room).map(mult).to_numpy() * (1 + .22 * (acc - 1)) * (1 + .004 * am)
                 * np.exp(.35 * rng.normal(size=n)) * (1 + .6 * np.exp(-np.hypot(dlat, dlon) * 12)) * np.where(sh, 1.05, 1))
        rating = np.minimum(1, .85 + .1 * rng.random(n) + np.where(sh, .03, 0))
        frames.append(pd.DataFrame({
            "city": name, "room_type": room, "property_type": rng.choice(props, n), "price": np.round(price),
            "accommodates": acc.astype(float), "bedrooms": np.where(entire, np.maximum(1, acc // 2), 1).astype(float),
            "min_nights": 1 + np.floor(np.exp(rng.random(n) * 2.5)), "amenities": am,
            "rating": rating, "value": np.clip(rating - .04 + .05 * rng.normal(size=n), .5, 1),
            "superhost": sh, "instant": rng.random(n) < .35, "lat": lat + dlat, "lon": lon + dlon,
            "host_listings": 1 + np.floor(np.exp(rng.normal(size=n))), "response_rate": .6 + .4 * rng.random(n),
            "n_reviews": rng.poisson(30, n).astype(float)}))
        yrs = months.year.to_numpy()
        growth = 1 + .35 * (yrs - 2016)
        season = 1 + .45 * np.cos(2 * np.pi * (months.month.to_numpy() - peak) / 12)
        covid = (months >= COVID[0]) & (months <= COVID[1])
        cnt = (720 * growth * season * np.where(covid, .15, 1) * (.9 + .2 * rng.random(len(months)))).astype(int)
        monthly.append(pd.DataFrame({"city": name, "month": months, "reviews": cnt}))
    L = pd.concat(frames, ignore_index=True)
    L.insert(0, "listing_id", np.arange(1, len(L) + 1))
    return L, pd.concat(monthly, ignore_index=True)
