"""Funciones compartidas por el entrenamiento, la evaluación y la aplicación."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDE = {"ASSETID", "PRICE", "UNITPRICE", "geometry", "DISTRITO", "PERIOD"}


def preparar(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Solo atributos disponibles al publicar el anuncio; nunca el precio objetivo."""
    x = df.drop(columns=list(EXCLUDE & set(df.columns))).copy()
    x["ANTIGUEDAD_2018"] = (2018 - x["CADCONSTRUCTIONYEAR"]).clip(lower=0)
    x["AREA_POR_HABITACION"] = x["CONSTRUCTEDAREA"] / x["ROOMNUMBER"].clip(lower=1)
    return x[features] if features is not None else x


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mantiene un registro auditable de los filtros aplicados."""
    inicial = len(df)
    df = df.drop_duplicates(["ASSETID", "PERIOD"], keep="last").copy()
    duplicados = inicial - len(df)
    ok = (df.PRICE.between(30_000, 5_000_000) &
          df.CONSTRUCTEDAREA.between(20, 500) &
          df.ROOMNUMBER.between(0, 15) &
          df.BATHNUMBER.between(0, 10) &
          df.UNITPRICE.between(300, 30_000) &
          df.LATITUDE.notna() & df.LONGITUDE.notna())
    df = df.loc[ok].copy()
    return df, {"inicial": inicial, "duplicados_asset_period": duplicados,
                "excluidos_por_rango_o_ubicacion": int((~ok).sum()), "final": len(df)}


def precio(model, x: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(x)) * x["CONSTRUCTEDAREA"].to_numpy()


def metricas(y, pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y, pred = np.asarray(y), np.asarray(pred)
    return {"n": int(len(y)), "mae_eur": round(float(mean_absolute_error(y, pred)), 2),
            "rmse_eur": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            "mediana_error_pct": round(float(np.median(np.abs(pred - y) / y) * 100), 2),
            "r2": round(float(r2_score(y, pred)), 4)}


def correccion_conformal(y, low, high, alpha=.2) -> float:
    """Cuantil finito split-conformal, reservado a datos de calibración."""
    s = np.maximum(low - np.asarray(y), np.asarray(y) - high)
    k = min(len(s), int(np.ceil((len(s) + 1) * (1 - alpha))))
    return float(max(0, np.partition(s, k - 1)[k - 1]))
