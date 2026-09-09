"""
VALORA · Entrenamiento del motor de valoración
==============================================
Genera, para Madrid, Barcelona y Valencia:
  - tres modelos por ciudad (percentiles 10, 50 y 90) sobre log(€/m²)
  - restricciones de monotonía en las variables de signo inequívoco
  - validación cruzada por bloques espaciales (sin fuga)
  - reindexación por distrito de 2018 a precios corrientes
Artefactos en formato nativo de XGBoost: no dependen de la versión de scikit-learn.
"""
import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rdata
import xgboost as xgb
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
RS = 42
DATA = Path("data")
OUT = Path("artefactos"); OUT.mkdir(exist_ok=True)

CIUDADES = ["Madrid", "Barcelona", "Valencia"]

# Código de distrito -> nombre, extraído del segmento 8 de LOCATIONID
DISTRITOS = {
 "Madrid": {"01":"Centro","02":"Arganzuela","03":"Retiro","04":"Salamanca","05":"Chamartín",
   "06":"Tetuán","07":"Chamberí","08":"Fuencarral","09":"Moncloa","10":"Latina",
   "11":"Carabanchel","12":"Usera","13":"Puente de Vallecas","14":"Moratalaz",
   "15":"Ciudad Lineal","16":"Hortaleza","17":"Villaverde","18":"Villa de Vallecas",
   "19":"Vicálvaro","20":"San Blas","21":"Barajas"},
 "Barcelona": {"01":"Ciutat Vella","02":"Eixample","03":"Sants-Montjuïc","04":"Les Corts",
   "05":"Sarrià-Sant Gervasi","06":"Gràcia","07":"Horta-Guinardó","08":"Nou Barris",
   "09":"Sant Andreu","10":"Sant Martí"},
 "Valencia": {"01":"Ciutat Vella","02":"L'Eixample","03":"Extramurs","04":"Campanar",
   "05":"La Saïdia","06":"El Pla del Real","07":"L'Olivereta","08":"Patraix","09":"Jesús",
   "10":"Quatre Carreres","11":"Poblats Marítims","12":"Camins al Grau","13":"Algirós",
   "14":"Benimaclet","15":"Rascanya","16":"Benicalap","17":"Pobles del Nord",
   "18":"Pobles de l'Oest","19":"Pobles del Sud"},
}

# Nivel de precio publicado por idealista, agosto 2026 (€/m² de oferta)
NIVEL_2026 = {
 "Madrid": {"_ciudad": 6471, "Salamanca":10841, "Chamberí":9209, "Retiro":8467,
   "Chamartín":8222, "Centro":7745, "Arganzuela":6346, "Moncloa":6332, "Fuencarral":5610,
   "Hortaleza":5483, "Ciudad Lineal":5201, "Barajas":5171, "Moratalaz":4563,
   "Latina":4044, "Carabanchel":3842, "Puente de Vallecas":3480},
 "Barcelona": {"_ciudad": 5440, "Sarrià-Sant Gervasi":7167, "Les Corts":6670,
   "Eixample":6648, "Gràcia":5670, "Sant Martí":5352, "Ciutat Vella":4841,
   "Sants-Montjuïc":4592, "Horta-Guinardó":4177, "Sant Andreu":4024, "Nou Barris":3270},
 "Valencia": {"_ciudad": 3469, "L'Eixample":5213, "Ciutat Vella":4840,
   "El Pla del Real":4359, "Extramurs":3792, "Campanar":3488, "Algirós":3449,
   "Camins al Grau":3401, "Benimaclet":3393, "Poblats Marítims":3386,
   "Quatre Carreres":3203},
}

MONOTONAS = {"BATHNUMBER":1, "ROOMNUMBER":1, "HASLIFT":1, "HASPARKINGSPACE":1,
  "HASSWIMMINGPOOL":1, "HASDOORMAN":1, "HASAIRCONDITIONING":1, "HASTERRACE":1,
  "HASGARDEN":1, "HASBOXROOM":1, "HASWARDROBE":1,
  "DISTANCE_TO_CITY_CENTER":-1, "DISTANCE_TO_METRO":-1}

CONV = rdata.conversion.SimpleConverter(default_encoding="utf-8")


def leer_rda(fichero, objeto):
    return CONV.convert(rdata.parser.parse_file(DATA / fichero))[objeto]


def poligono(sfg):
    """Convierte una geometría sf anidada en un polígono de shapely."""
    try:
        a = np.asarray(sfg[0], dtype=float)
        if a.ndim == 2 and a.shape[1] >= 2:
            return Polygon(a[:, :2])
    except Exception:
        pass
    partes = []
    for sub in sfg:
        try:
            b = np.asarray(sub[0], dtype=float)
            if b.ndim == 2 and b.shape[1] >= 2:
                partes.append(Polygon(b[:, :2]))
        except Exception:
            continue
    return MultiPolygon(partes) if partes else None


def asignar_distrito(df, ciudad):
    """Point-in-polygon contra los barrios del paquete; devuelve distrito y centroides."""
    poly = pd.DataFrame(leer_rda(f"{ciudad}_Polygons.rda", f"{ciudad}_Polygons"))
    geoms, codigos, nombres = [], [], []
    for i in range(len(poly)):
        g = poligono(poly["geometry"].values[i])
        if g is None or not g.is_valid:
            continue
        geoms.append(g)
        codigos.append(str(poly["LOCATIONID"].values[i]).split("-")[7])
        nombres.append(str(poly["LOCATIONNAME"].values[i]))

    arbol = STRtree(geoms)
    puntos = [Point(lo, la) for lo, la in zip(df.LONGITUDE.values, df.LATITUDE.values)]
    idx = arbol.query(puntos, predicate="within")
    asign = np.full(len(df), -1, dtype=int)
    asign[idx[0]] = idx[1]

    mapa = DISTRITOS[ciudad]
    df["DISTRITO"] = [mapa.get(codigos[i], "Otros") if i >= 0 else "Otros" for i in asign]

    # centroides de barrio con su distrito: la app los usa para localizar sin shapely
    centroides = [{"lat": round(g.centroid.y, 5), "lon": round(g.centroid.x, 5),
                   "barrio": n, "distrito": mapa.get(c, "Otros")}
                  for g, c, n in zip(geoms, codigos, nombres)]
    return df, centroides


def entrenar_ciudad(ciudad):
    print(f"\n{'='*66}\n  {ciudad.upper()}\n{'='*66}")
    df = pd.DataFrame(leer_rda(f"{ciudad}_Sale.rda", f"{ciudad}_Sale"))
    df = df[[c for c in df.columns if c != "geometry"]].reset_index(drop=True)
    df, centroides = asignar_distrito(df, ciudad)
    print(f"  {len(df):,} anuncios | {df.ASSETID.nunique():,} viviendas | "
          f"{(df.DISTRITO!='Otros').mean()*100:.1f}% con distrito asignado")

    feats = [c for c in df.columns
             if c not in ("ASSETID", "PRICE", "UNITPRICE", "DISTRITO")]
    X = df[feats].astype(float)
    y = np.log(df.UNITPRICE.values.astype(float))
    area = df.CONSTRUCTEDAREA.values.astype(float)
    precio = df.PRICE.values.astype(float)
    mono = "(" + ",".join(str(MONOTONAS.get(c, 0)) for c in feats) + ")"

    # bloques espaciales de ~100 m: agrupan también el mismo inmueble repetido
    bloque = (df.LATITUDE.round(3).astype(str) + "_" +
              df.LONGITUDE.round(3).astype(str)).values

    base = dict(n_estimators=450, learning_rate=0.05, max_depth=7, subsample=0.85,
                colsample_bytree=0.85, min_child_weight=5, n_jobs=-1,
                random_state=RS, tree_method="hist", monotone_constraints=mono)

    # ── validación honesta por bloques ──────────────────────────────────
    r2s, maes = [], []
    pliegues = list(GroupKFold(3).split(X, y, bloque))
    for tr, te in pliegues:
        m50 = xgb.XGBRegressor(**base).fit(X.iloc[tr], y[tr])
        pred = np.exp(m50.predict(X.iloc[te])) * area[te]
        r2s.append(r2_score(precio[te], pred))
        maes.append(mean_absolute_error(precio[te], pred))
    tr, te = pliegues[0]
    lo = xgb.XGBRegressor(**base, objective="reg:quantileerror",
                          quantile_alpha=0.1).fit(X.iloc[tr], y[tr])
    hi = xgb.XGBRegressor(**base, objective="reg:quantileerror",
                          quantile_alpha=0.9).fit(X.iloc[tr], y[tr])
    cob = float((((precio[te] >= np.exp(lo.predict(X.iloc[te])) * area[te]) &
                  (precio[te] <= np.exp(hi.predict(X.iloc[te])) * area[te]))).mean())
    r2, mae = float(np.mean(r2s)), float(np.mean(maes))
    print(f"  R² (bloques espaciales) : {r2:.4f}")
    print(f"  MAE                     : {mae:,.0f} €")
    print(f"  Cobertura del intervalo : {cob*100:.1f} %  (objetivo 80 %)")

    # ── modelos finales sobre todos los datos ───────────────────────────
    modelos = {}
    for nombre, alpha in (("q10", 0.1), ("q50", None), ("q90", 0.9)):
        kw = dict(base)
        if alpha is not None:
            kw.update(objective="reg:quantileerror", quantile_alpha=alpha)
        m = xgb.XGBRegressor(**kw).fit(X, y)
        ruta = OUT / f"{ciudad.lower()}_{nombre}.ubj"
        m.get_booster().save_model(str(ruta))
        modelos[nombre] = ruta.name
        print(f"  guardado {ruta.name}  ({ruta.stat().st_size/1e6:.1f} MB)")

    # ── reindexación por distrito ───────────────────────────────────────
    med18 = df[df.DISTRITO != "Otros"].groupby("DISTRITO").UNITPRICE.median()
    niveles = NIVEL_2026[ciudad]
    ciudad18 = float(df.UNITPRICE.median())
    f_ciudad = niveles["_ciudad"] / ciudad18
    factores = {}
    for d, v18 in med18.items():
        v26 = niveles.get(d)
        factores[d] = {"e2018": round(float(v18), 1),
                       "e2026": float(v26) if v26 else None,
                       "factor": round(float(v26) / float(v18), 4) if v26
                                 else round(f_ciudad, 4),
                       "estimado": v26 is None,
                       "n": int((df.DISTRITO == d).sum())}
    print(f"  factor de ciudad x{f_ciudad:.2f} | "
          f"{sum(1 for f in factores.values() if not f['estimado'])}/{len(factores)} "
          f"distritos con nivel publicado")

    meta = {
        "ciudad": ciudad,
        "features": feats,
        "modelos": modelos,
        "n_anuncios": int(len(df)),
        "n_viviendas": int(df.ASSETID.nunique()),
        "r2_bloques": round(r2, 4),
        "mae": round(mae, 0),
        "cobertura_intervalo": round(cob, 4),
        "unitprice_mediana_2018": round(ciudad18, 1),
        "nivel_ciudad_2026": niveles["_ciudad"],
        "factor_ciudad": round(f_ciudad, 4),
        "distritos": factores,
        "centroides": centroides,
        "dominio": {
            "area_p01": float(df.CONSTRUCTEDAREA.quantile(0.01)),
            "area_p99": float(df.CONSTRUCTEDAREA.quantile(0.99)),
            "area_max": float(df.CONSTRUCTEDAREA.max()),
            "precio_max_2018": float(df.PRICE.max()),
        },
        "medianas": {c: float(df[c].median()) for c in feats},
        "centro": {"lat": float(df.LATITUDE.median()),
                   "lon": float(df.LONGITUDE.median())},
    }
    (OUT / f"{ciudad.lower()}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import sys, gc
    objetivo = sys.argv[1:] or CIUDADES
    resumen = {}
    for c in objetivo:
        m = entrenar_ciudad(c); gc.collect()
        resumen[c] = {k: m[k] for k in
                      ("n_anuncios", "n_viviendas", "r2_bloques", "mae",
                       "cobertura_intervalo", "factor_ciudad", "nivel_ciudad_2026")}
    prev = {}
    if (OUT / "resumen.json").exists():
        prev = json.loads((OUT / "resumen.json").read_text(encoding="utf-8"))
    prev.update(resumen)
    (OUT / "resumen.json").write_text(
        json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    resumen = prev
    print(f"\n{'='*66}\n  RESUMEN\n{'='*66}")
    print(pd.DataFrame(resumen).T.to_string())
