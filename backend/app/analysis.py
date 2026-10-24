"""All analytics: pandas/numpy for aggregation, scipy for tests, scikit-learn for ML-based importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .data import COVID

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def sel(L: pd.DataFrame, city: str = "ALL", room: str = "ALL") -> pd.DataFrame:
    m = pd.Series(True, index=L.index)
    if city != "ALL":
        m &= L["city"] == city
    if room != "ALL":
        m &= L["room_type"] == room
    return L[m]


# ------------------------------------------------------------------ Q1: city differences
def city_summary(L: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    g = L.groupby("city")
    out = pd.DataFrame({
        "listings": g.size(),
        "avg_price": g["price"].mean(),
        "median_price": g["price"].median(),
        "median_price_per_guest": g["price_per_guest"].median(),
        "avg_rating": g["rating"].mean() * 100,
        "superhost_pct": g["superhost"].mean() * 100,
        "entire_home_pct": g["room_type"].apply(lambda s: s.str.startswith("Entire").mean() * 100),
        "avg_accommodates": g["accommodates"].mean(),
        "avg_min_nights": g["min_nights"].mean(),
        "instant_pct": g["instant"].mean() * 100,
    })
    out["total_reviews"] = monthly.groupby("city")["reviews"].sum().reindex(out.index).fillna(0) if len(monthly) else 0
    return out.round(2)


def room_mix(L: pd.DataFrame) -> pd.DataFrame:
    return (pd.crosstab(L["city"], L["room_type"], normalize="index") * 100).round(1)


def city_price_test(L: pd.DataFrame) -> dict:
    """Kruskal-Wallis: do price distributions differ between cities? epsilon^2 is the effect size."""
    groups = [g["price"].to_numpy() for _, g in L.groupby("city")]
    h, p = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return {"H": float(h), "p": float(p), "epsilon2": float((h - k + 1) / (n - k))}


# ------------------------------------------------------------------ Q2: price drivers
NICE = {"city": "City", "room_type": "Room type", "property_type": "Property type", "accommodates": "Accommodates",
        "bedrooms": "Bedrooms", "amenities": "Amenities count", "min_nights": "Minimum nights", "rating": "Review rating",
        "value": "Value rating", "host_listings": "Host listings", "response_rate": "Host response rate",
        "superhost": "Superhost", "instant": "Instant bookable", "lat": "Latitude", "lon": "Longitude",
        "n_reviews": "Number of reviews"}


def ml_importance(L: pd.DataFrame, city: str = "ALL", sample: int = 60_000, seed: int = 0):
    """Gradient boosting on ln(price); permutation importance on hold-out data (drop in R^2 when a column is shuffled).
    Handles non-linearities and interactions, and treats categoricals/NaNs natively."""
    D = sel(L, city)
    D = D.sample(min(len(D), sample), random_state=seed)
    cats = ["room_type", "property_type"] + (["city"] if city == "ALL" else [])
    nums = ["accommodates", "bedrooms", "amenities", "min_nights", "rating", "value", "host_listings",
            "response_rate", "superhost", "instant", "lat", "lon", "n_reviews"]
    X = D[cats + nums].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    X[["superhost", "instant"]] = X[["superhost", "instant"]].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, D["log_price"], test_size=0.2, random_state=seed)
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=250, learning_rate=0.08,
                                          random_state=seed).fit(Xtr, ytr)
    r2 = float(model.score(Xte, yte))
    pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=seed, n_jobs=-1)
    df = pd.DataFrame({"feature": [NICE[c] for c in X.columns], "importance": pi.importances_mean, "std": pi.importances_std})
    df["share_pct"] = 100 * df["importance"].clip(lower=0) / max(df["importance"].clip(lower=0).sum(), 1e-12)
    return df.sort_values("importance", ascending=False).reset_index(drop=True), r2


def price_premiums(L: pd.DataFrame, city: str = "ALL") -> tuple[pd.DataFrame, float]:
    """OLS on ln(price) with heteroskedasticity-robust (HC1) errors, in numpy. Coefficients are shown as % price change,
    holding the other attributes constant."""
    D = sel(L, city)
    cols: dict[str, pd.Series] = {
        "Accommodates (+1 guest)": D["accommodates"],
        "Bedrooms (+1)": D["bedrooms"].fillna(D["bedrooms"].median()),
        "Amenities (+10)": D["amenities"] / 10,
        "Minimum nights (x2)": np.log2(1 + D["min_nights"]),
        "Review rating (+1 of 10)": D["rating"].fillna(D["rating"].median()) * 10,
        "Value rating (+1 of 10)": D["value"].fillna(D["value"].median()) * 10,
        "Host listings (x2)": np.log2(1 + D["host_listings"]),
        "Response rate (+10 pts)": D["response_rate"].fillna(D["response_rate"].median()) * 10,
        "Superhost": D["superhost"].astype(float),
        "Instant bookable": D["instant"].astype(float),
    }
    cat_cols = ["room_type", "property_type"] + (["city"] if city == "ALL" else [])
    for c in cat_cols:  # dummies, most common level is the reference
        ref = D[c].value_counts().index[0]
        for lvl in sorted(D[c].unique()):
            if lvl != ref:
                cols[f"{NICE[c]}: {lvl} (vs {ref})"] = (D[c] == lvl).astype(float)
    X = pd.DataFrame(cols)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    A, y = X.to_numpy(float), D["log_price"].to_numpy(float)
    n, k = A.shape
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    Ar = A * resid[:, None]
    cov = n / (n - k) * bread @ (Ar.T @ Ar) @ bread
    se = np.sqrt(np.diag(cov))
    adj_r2 = 1 - (1 - (1 - resid.var() / y.var())) * (n - 1) / (n - k)
    out = pd.DataFrame({"term": X.columns, "coef": beta, "se": se})
    out["pct_effect"] = (np.exp(out["coef"]) - 1) * 100
    out["ci_low"] = (np.exp(out["coef"] - 1.96 * out["se"]) - 1) * 100
    out["ci_high"] = (np.exp(out["coef"] + 1.96 * out["se"]) - 1) * 100
    out["p_value"] = 2 * stats.norm.sf(np.abs(out["coef"] / out["se"]))
    return out[out["term"] != "const"].round(4).reset_index(drop=True), float(adj_r2)


def price_by_group(L: pd.DataFrame, feature: str, city: str = "ALL") -> pd.DataFrame:
    D = sel(L, city).copy()
    if feature in ("accommodates", "bedrooms"):
        D[feature] = D[feature].clip(upper=10 if feature == "accommodates" else 6)
        D = D.dropna(subset=[feature])
        D[feature] = D[feature].astype(int)
    g = D.groupby(feature)["price"].agg(count="size", median_price="median", mean_price="mean").reset_index()
    if feature not in ("accommodates", "bedrooms"):
        g = g.sort_values("count", ascending=False).head(12)
    return g.rename(columns={feature: "group"}).round(2)


# ------------------------------------------------------------------ Q3: trends & seasonality
def monthly_wide(monthly: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    w = monthly[monthly["city"].isin(cities)].pivot_table(index="month", columns="city", values="reviews", aggfunc="sum")
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="MS")).fillna(0)


def _covid_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((idx >= COVID[0]) & (idx <= COVID[1]))


def seasonality(monthly: pd.DataFrame, cities: list[str], exclude_covid: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seasonal index per calendar month (1.0 = average month): ratio of each month to its centred 2x12 moving average
    (which removes growth), averaged by calendar month and rescaled to mean 1. If exclude_covid, the COVID months are
    interpolated before smoothing and left out of the average. Returns (12 x city table, summary per city)."""
    w = monthly_wide(monthly, cities)
    covid = _covid_mask(w.index) if exclude_covid else np.zeros(len(w), bool)
    idx, rows = {}, []
    for c in w:
        s = w[c].mask(covid).interpolate(limit_direction="both") if exclude_covid else w[c]
        trend = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
        ratio = (s / trend).where(~covid).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 24:
            continue
        m = ratio.groupby(ratio.index.month).mean().reindex(range(1, 13))
        m = m / m.mean()
        idx[c] = m.to_numpy()
        rows.append({"city": c, "peak_month": MONTHS[int(m.argmax()) ], "quietest_month": MONTHS[int(m.argmin())],
                     "peak_to_trough": round(float(m.max() / m.min()), 2), "months_used": len(ratio),
                     "seasonal_strength": round(decompose(monthly, c, exclude_covid)[1], 2)})
    return pd.DataFrame(idx, index=MONTHS), pd.DataFrame(rows).set_index("city") if rows else pd.DataFrame()


def decompose(monthly: pd.DataFrame, city: str, exclude_covid: bool = True) -> tuple[pd.DataFrame, float]:
    """Additive decomposition: trend = centred 2x12 moving average; seasonal = mean detrended value per calendar month.
    seasonal_strength = 1 - Var(remainder)/Var(detrended), clipped to [0, 1] (Hyndman's measure)."""
    s = monthly_wide(monthly, [city])[city]
    if exclude_covid:
        s = s.mask(_covid_mask(s.index)).interpolate(limit_direction="both")
    trend = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    detr = s - trend
    seas = detr.groupby(detr.index.month).transform("mean")
    rem = detr - seas
    strength = float(np.clip(1 - rem.var() / detr.dropna().var(), 0, 1)) if detr.notna().sum() > 24 else float("nan")
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seas, "remainder": rem}), strength


def yearly_growth(monthly: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    w = monthly_wide(monthly, cities)
    w = w[w.index.year.isin([y for y, g in w.groupby(w.index.year) if len(g) == 12])]
    y = w.groupby(w.index.year).sum()
    return y


# ------------------------------------------------------------------ Q4: value
def value_scores(L: pd.DataFrame, quality_weight: float = 0.5, room: str = "ALL") -> pd.DataFrame:
    """score = w * quality + (1-w) * affordability, both relative to the best city (0-100).
    quality = mean(rating, value rating); affordability = cheapest median price per guest / city's median price per guest."""
    D = sel(L, room=room)
    g = D.groupby("city")
    t = pd.DataFrame({"median_price": g["price"].median(), "median_price_per_guest": g["price_per_guest"].median(),
                      "avg_rating": g["rating"].mean() * 100, "avg_value_rating": g["value"].mean() * 10,
                      "listings": g.size()})
    q = (g["rating"].mean().fillna(0) + g["value"].mean().fillna(0)) / 2
    t["quality_index"] = 100 * q / q.max()
    t["affordability_index"] = 100 * t["median_price_per_guest"].min() / t["median_price_per_guest"]
    t["score"] = quality_weight * t["quality_index"] + (1 - quality_weight) * t["affordability_index"]
    return t.sort_values("score", ascending=False).round(2)


# ------------------------------------------------------------------ simple driver strengths (same contract as the old Java API)
def _eta(y: pd.Series, g: pd.Series) -> float:
    """Correlation ratio: share of ln(price) variance explained by a categorical attribute (0..1, as sqrt)."""
    mean = y.mean()
    sst = float(((y - mean) ** 2).sum())
    grp = y.groupby(g.to_numpy()).agg(["size", "mean"])
    ssb = float((grp["size"] * (grp["mean"] - mean) ** 2).sum())
    return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0


def driver_strengths(L: pd.DataFrame, city: str = "ALL") -> pd.DataFrame:
    D = sel(L, city)
    y = D["log_price"]
    num = {"Accommodates": D["accommodates"], "Bedrooms": D["bedrooms"], "Amenities count": D["amenities"],
           "Minimum nights (log)": D["log_min_nights"], "Review rating": D["rating"], "Value rating": D["value"],
           "Host listings (log)": D["log_host_listings"], "Host response rate": D["response_rate"]}
    cat = {"Room type": D["room_type"], "Property type": D["property_type"],
           "Superhost": D["superhost"].map({True: "Superhost", False: "Not superhost"}),
           "Instant bookable": D["instant"].map({True: "Instant", False: "Request"})}
    if city == "ALL":
        cat["City"] = D["city"]
    rows = []
    for name, x in num.items():
        r = x.corr(y)
        if pd.notna(r):
            rows.append({"feature": name, "kind": "numeric", "strength": abs(r), "correlation": r})
    for name, g in cat.items():
        rows.append({"feature": name, "kind": "categorical", "strength": _eta(y, g), "correlation": None})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).round(3).reset_index(drop=True)
