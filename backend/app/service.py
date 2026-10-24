"""Turns the pandas results into the exact JSON shapes the Angular frontend expects (framework-free, easy to test)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import analysis as A

NUM_COLS = {"Accommodates": "accommodates", "Bedrooms": "bedrooms"}
CAT_COLS = {"Room type": "room_type", "Property type": "property_type", "Superhost": "superhost",
            "Instant bookable": "instant", "City": "city"}


def camel(s: str) -> str:
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), s)


def rec(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts (NaN -> null, numpy -> python)."""
    return json.loads(df.rename(columns=camel).to_json(orient="records"))


class Service:
    def __init__(self, L: pd.DataFrame, monthly: pd.DataFrame, source: str):
        self.L, self.m, self.source = L, monthly, source

    def _cities(self, cities: str) -> list[str]:
        have = sorted(self.m["city"].unique()) if len(self.m) else []
        return have if cities.upper() == "ALL" else [c for c in cities.split(",") if c in have]

    # ---- endpoints
    def meta(self) -> dict:
        return {"source": self.source, "cities": sorted(self.L["city"].unique()),
                "roomTypes": self.L["room_type"].value_counts().index.tolist(),
                "listings": int(len(self.L)), "reviews": int(self.m["reviews"].sum()) if len(self.m) else 0}

    def cities(self) -> list[dict]:
        return rec(A.city_summary(self.L, self.m).reset_index())

    def room_types(self) -> list[dict]:
        g = self.L.groupby(["city", "room_type"])["price"].agg(count="size", median_price="median").reset_index()
        g["share"] = (g["count"] / g.groupby("city")["count"].transform("sum")).round(3)
        return rec(g.rename(columns={"room_type": "roomType"}))

    def drivers(self, city: str) -> list[dict]:
        return rec(A.driver_strengths(self.L, city))

    def price_by(self, feature: str, city: str) -> list[dict]:
        col = NUM_COLS.get(feature) or CAT_COLS.get(feature)
        if col is None:
            return []
        g = A.price_by_group(self.L, col, city)
        if col == "superhost":
            g["group"] = g["group"].map({True: "Superhost", False: "Not superhost"})
        elif col == "instant":
            g["group"] = g["group"].map({True: "Instant", False: "Request"})
        g["group"] = g["group"].astype(str)
        return rec(g[["group", "count", "median_price"]])

    def monthly(self, cities: str) -> dict:
        cs = self._cities(cities)
        if not cs:
            return {"months": [], "series": {}}
        w = A.monthly_wide(self.m, cs)
        return {"months": w.index.strftime("%Y-%m").tolist(), "series": {c: [int(v) for v in w[c]] for c in w}}

    def seasonality(self, cities: str, exclude_covid: bool) -> dict:
        cs = self._cities(cities)
        if not cs:
            return {"months": A.MONTHS, "series": {}, "peak": {}, "low": {}}
        idx, summ = A.seasonality(self.m, cs, exclude_covid)
        return {"months": A.MONTHS, "series": {c: [round(float(v), 3) for v in idx[c]] for c in idx},
                "peak": summ["peak_month"].to_dict() if len(summ) else {},
                "low": summ["quietest_month"].to_dict() if len(summ) else {}}

    def value(self, quality_weight: float, room: str) -> list[dict]:
        return rec(A.value_scores(self.L, quality_weight, room).drop(columns="listings").reset_index())

    # ---- extras the current UI ignores (ready for richer charts)
    def importance(self, city: str) -> dict:
        imp, r2 = A.ml_importance(self.L, city)
        return {"r2": round(r2, 3), "features": rec(imp)}

    def premiums(self, city: str) -> dict:
        ols, adj = A.price_premiums(self.L, city)
        return {"adjR2": round(adj, 3), "terms": rec(ols)}
