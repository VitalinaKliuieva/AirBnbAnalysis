"""FastAPI backend: same endpoints/JSON as the former Spring Boot service, so the Angular app is unchanged.
Run from /backend:  uvicorn app.main:app --port 8080"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from . import data
from .service import Service

S: dict[str, Service] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    L, monthly, source = data.load_data()   # first run on the full CSVs takes a few minutes, then cached
    S["svc"] = Service(L, monthly, source)
    yield


app = FastAPI(title="Airbnb analysis API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/meta")
def meta(): return S["svc"].meta()

@app.get("/api/cities")
def cities(): return S["svc"].cities()

@app.get("/api/cities/room-types")
def room_types(): return S["svc"].room_types()

@app.get("/api/price/drivers")
def drivers(city: str = "ALL"): return S["svc"].drivers(city)

@app.get("/api/price/by")
def price_by(feature: str, city: str = "ALL"): return S["svc"].price_by(feature, city)

@app.get("/api/price/importance")
def importance(city: str = "ALL"): return S["svc"].importance(city)

@app.get("/api/price/premiums")
def premiums(city: str = "ALL"): return S["svc"].premiums(city)

@app.get("/api/reviews/monthly")
def monthly(cities: str = "ALL"): return S["svc"].monthly(cities)

@app.get("/api/reviews/seasonality")
def seasonality(cities: str = "ALL", exclude_covid: bool = Query(True, alias="excludeCovid")):
    return S["svc"].seasonality(cities, exclude_covid)

@app.get("/api/value")
def value(quality_weight: float = Query(0.5, alias="qualityWeight"), room_type: str = Query("ALL", alias="roomType")):
    return S["svc"].value(quality_weight, room_type)
