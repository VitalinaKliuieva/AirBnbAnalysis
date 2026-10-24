# Airbnb Market Analytics Platform

Full-stack analytics app for the Maven Analytics Airbnb dataset (~250k listings, 10 cities, 5M+ reviews).
**Python 3.10+ / FastAPI** backend, **Angular 18 / Chart.js** frontend. Runs entirely on localhost.

```
CSV files ──► data.py (pandas) ──► analysis.py ──► service.py ──► main.py (FastAPI) ──HTTP/JSON──► Angular ──► Chart.js
              clean + cache        pure functions   JSON contract   routes, CORS
```

## Quick start

Download “Airbnb” from the Maven Analytics Data Playground(https://mavenanalytics.io/data-playground/airbnb-listings-reviews?_gl=1*17674v5*_gcl_au*NTg5NTgwMTc0LjE3OTI2OTc1NDQ.*_ga*NDY3OTAxNTE1LjE3OTI2OTc0MTY.*_ga_RNVXNX7G3B*czE3OTI2OTc0MTYkbzEkZzEkdDE3OTI2OTc1NDQkajU2JGwwJGgw*_ga_D6PKE9HLBV*czE3OTI2OTc0MTYkbzEkZzEkdDE3OTI2OTc1NDQkajU2JGwwJGgw*_ga_9CBEFJW9QX*czE3OTI2OTc0MTYkbzEkZzEkdDE3OTI2OTc1NDQkajU2JGwwJGgw) and put Listings.csv and Reviews.csv into data/

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8080 --reload                # docs: http://localhost:8080/docs

# frontend (second terminal, Node 20+)
cd frontend && npm install && npm start                  # http://localhost:4200
```
Put `Listings.csv` and `Reviews.csv` into `data/` (project root). If no listings file is found, the backend generates
synthetic demo data (`source: "synthetic"` in `/api/meta`) so the stack is runnable without the dataset.
`GET /` returns 404 by design; the API lives under `/api`, the UI on port 4200.

## Repository layout
```
backend/
  requirements.txt
  app/
    data.py       load + clean CSVs, chunked review streaming, pickle cache, synthetic fallback
    analysis.py   analytics as pure functions over DataFrames (no web code)
    service.py    Service class: calls analysis, returns JSON-safe dicts in the shapes the UI expects
    main.py       FastAPI app, lifespan loader, routes, CORS
frontend/
  proxy.conf.json           /api -> http://localhost:8080 (dev server proxy)
  src/app/api.service.ts    thin HttpClient wrapper
  src/app/chart.component.ts  reusable Chart.js wrapper (input: Chart.js config)
  src/app/app.component.ts  four-tab dashboard (standalone component, builds chart configs from API data)
data/                       CSV input + .cache.pkl (generated)
```

## Backend design

### Data layer (`data.py`)
- `load_data(max_price=10_000)` returns `(listings, monthly, source)`.
- **listings** (one row per listing): `listing_id, city, room_type, property_type, price, accommodates, bedrooms, min_nights,
  amenities (count), rating (0-1), value (0-1), superhost, instant, lat, lon, host_listings, response_rate, n_reviews`
  plus derived `log_price, price_per_guest, log_min_nights, log_host_listings`.
- **monthly** (reviews aggregated): `city, month (month-start Timestamp), reviews`. The multi-million-row reviews file is read
  in 1M-row chunks, mapped to a city via `listing_id`, and reduced to counts immediately, so raw reviews are never held in memory.
- Cleaning: header matching is case-insensitive and missing columns become NaN; price strings like `$1,200.00` are parsed;
  prices <= 0 or > `max_price` are dropped; rating scale (0-5 / 0-10 / 0-100) is detected from the column max and normalised to 0-1;
  `property_type` keeps the 8 most common values (rest -> `Other`); BOM and bad encodings are tolerated.
- **Cache**: result is pickled to `data/.cache.pkl`, keyed on file names, mtimes and `max_price`. Delete the file to force a reload.
  Only load pickles you created yourself (pickle is not safe for untrusted files).
- `AIRBNB_DATA_DIR` env var overrides the data folder.

### Analytics layer (`analysis.py`)
| Function | What it does |
|---|---|
| `city_summary`, `room_mix` | per-city aggregates (mean/median price, price per guest, rating, superhost %, room-type mix, review totals) |
| `driver_strengths` | \|Pearson r\| of numeric attributes and correlation ratio η of categorical ones, against ln(price) (used by the current UI) |
| `ml_importance` | `HistGradientBoostingRegressor` on ln(price) + permutation importance on a hold-out split (native categoricals/NaNs) |
| `price_premiums` | OLS in numpy with HC1 robust errors; effects reported as % price change with 95% CI and p-values |
| `price_by_group` | median/mean price and counts per group |
| `seasonality`, `decompose` | seasonal index = month / centred 2x12 moving average (removes growth), averaged per calendar month; additive decomposition with Hyndman seasonal strength; optional COVID exclusion (Mar 2020 - Jun 2021) |
| `value_scores` | `w * quality + (1-w) * affordability`; quality = mean(rating, value rating), affordability = cheapest median price per guest / city's |
| `city_price_test` | Kruskal-Wallis + epsilon² across cities (implemented, not exposed via API yet) |

### Service and API layer
`service.Service` is framework-free: it takes the DataFrames, calls `analysis`, and converts results with `rec()`
(`DataFrame.to_json` -> `json.loads`) so NaN becomes `null` and numpy types become plain Python. Column names are converted to
camelCase (`median_price` -> `medianPrice`). `main.py` loads data once in the FastAPI **lifespan** hook and keeps one `Service`
in module state. Route handlers are plain `def` functions, so FastAPI runs them in its threadpool.

### Endpoints
| Method & path | Params | Response |
|---|---|---|
| `GET /api/meta` | | `{source, cities[], roomTypes[], listings, reviews}` |
| `GET /api/cities` | | `[{city, listings, avgPrice, medianPrice, medianPricePerGuest, avgRating, superhostPct, entireHomePct, avgAccommodates, avgMinNights, instantPct, totalReviews}]` |
| `GET /api/cities/room-types` | | `[{city, roomType, count, share (0-1), medianPrice}]` |
| `GET /api/price/drivers` | `city=ALL` | `[{feature, kind: numeric\|categorical, strength, correlation\|null}]` sorted by strength |
| `GET /api/price/by` | `feature`, `city=ALL` | `[{group, count, medianPrice}]`; features: Room type, Property type, Superhost, Instant bookable, City, Accommodates, Bedrooms (others -> `[]`) |
| `GET /api/price/importance` | `city=ALL` | `{r2, features:[{feature, importance, std, sharePct}]}` (model fit per request, a few seconds) |
| `GET /api/price/premiums` | `city=ALL` | `{adjR2, terms:[{term, coef, se, pctEffect, ciLow, ciHigh, pValue}]}` |
| `GET /api/reviews/monthly` | `cities=ALL\|A,B` | `{months:["2016-01",...], series:{city:[int]}}` |
| `GET /api/reviews/seasonality` | `cities=ALL\|A,B`, `excludeCovid=true` | `{months:[Jan..Dec], series:{city:[12 floats]}, peak:{city:month}, low:{city:month}}` |
| `GET /api/value` | `qualityWeight=0.5`, `roomType=ALL` | `[{city, score, medianPrice, medianPricePerGuest, avgRating, avgValueRating, qualityIndex, affordabilityIndex}]` |

`city=ALL` means no filter. Interactive docs are generated at `/docs`.

## Frontend design
- Standalone Angular 18 components, no router, no state library: `AppComponent` owns the data, builds Chart.js configs, and passes them
  to `<app-chart [config]>`; `ChartComponent` destroys and recreates the chart when the config object changes.
- Tabs: City comparison, Price drivers (click a bar to load `/price/by` for that attribute), Reviews & seasonality (city chips, COVID toggle),
  Best value (quality-vs-price slider).
- `proxy.conf.json` forwards `/api` to the backend, so the browser sees one origin; CORS is also enabled for `http://localhost:4200`.

## Extending
1. Add a pure function to `analysis.py` (DataFrame in, DataFrame/dict out).
2. Wrap it in a `Service` method that returns JSON-safe data (use `rec()` for DataFrames).
3. Add a route in `main.py`.
4. Call it from the frontend with `Api.get<T>('path', params)` and build a config for `<app-chart>`.

Example UI work not yet done: charts for `/price/importance` and `/price/premiums`.

## Testing (not included yet)
There are no automated tests. A good first test uses the synthetic data and FastAPI's test client:
```python
from fastapi.testclient import TestClient
from app.main import app

def test_meta():
    with TestClient(app) as c:                       # runs the lifespan loader
        r = c.get("/api/meta").json()
        assert r["listings"] > 0 and "Paris" in r["cities"]
```
Run it with `AIRBNB_DATA_DIR` pointing at an empty folder to force the synthetic dataset. Needs `pip install pytest httpx`.

## Known limitations
- State is in-process memory: with `--workers N`, each worker loads its own copy of the data.
- Startup blocks until data is loaded (minutes on the first full-CSV run); the UI shows "Loading..." until then.
- `ml_importance` and `price_premiums` are computed on every request (no caching).
- Prices are not currency-converted; `max_price` is only configurable in code (`load_data`).
- Importance, premiums and correlations are associations, not causal effects.
- Expected input: ISO-parsable review dates and the Maven column names listed in `data.py` (`WANTED`).

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `404 GET /` or `/favicon.ico` in uvicorn log | normal; open `http://localhost:4200` (UI) or `/docs` (API) |
| UI stuck on "Loading..." | backend still reading CSVs, or not running; check `http://localhost:8080/api/meta` |
| `source: "synthetic"` | no `Listings*.csv` found in `data/` (or wrong `AIRBNB_DATA_DIR`) |
| Empty review charts | no reviews file found, or its `listing_id`s don't match the listings file |
| Stale numbers after replacing CSVs | delete `data/.cache.pkl` |