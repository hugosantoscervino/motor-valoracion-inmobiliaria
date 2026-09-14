"""Entrena y evalúa idealista18; todas las métricas corresponden a ofertas de 2018."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from geografia import leer_rda, asignar_distrito
from modelado import preparar, limpiar, precio, metricas, correccion_conformal

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artefactos'
OUT.mkdir(exist_ok=True)
CITIES = ('Madrid', 'Barcelona', 'Valencia')

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def model(alpha=None, n_estimators=260):
    p = dict(n_estimators=n_estimators, max_depth=5, learning_rate=.06,
             min_child_weight=5, subsample=.9, colsample_bytree=.9,
             tree_method='hist', random_state=42, n_jobs=2)
    p.update(objective='reg:squarederror' if alpha is None else 'reg:quantileerror')
    if alpha is not None: p['quantile_alpha'] = alpha
    return xgb.XGBRegressor(**p)

def run(city):
    df = pd.DataFrame(leer_rda(f'{city}_Sale.rda', f'{city}_Sale'))
    df, flow = limpiar(df.drop(columns='geometry', errors='ignore').reset_index(drop=True))
    df, centroids = asignar_distrito(df, city)
    df = df[df.DISTRITO != 'Otros'].reset_index(drop=True)
    flow['sin_distrito'] = flow['final'] - len(df)
    flow['final'] = len(df)
    if len(df) < 1000: raise ValueError(f'Muestra insuficiente en {city}: {flow}')
    X = preparar(df)
    groups = df.ASSETID.astype(str).to_numpy()
    tr_ca, te = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=42).split(X, groups=groups))
    it, ic = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=43).split(X.iloc[tr_ca], groups=groups[tr_ca]))
    tr, ca = tr_ca[it], tr_ca[ic]
    assert not (set(groups[tr]) & set(groups[ca]) or set(groups[tr]) & set(groups[te]) or set(groups[ca]) & set(groups[te]))
    y = np.log(df.UNITPRICE.to_numpy(dtype=float))
    real = df.PRICE.to_numpy(dtype=float)
    area = df.CONSTRUCTEDAREA.to_numpy(dtype=float)
    city_baseline = float(df.UNITPRICE.iloc[tr].median())
    district_baseline = df.iloc[tr].groupby('DISTRITO').UNITPRICE.median().to_dict()
    base_city = city_baseline * area[te]
    base_district = df.DISTRITO.iloc[te].map(district_baseline).fillna(city_baseline).to_numpy() * area[te]
    models = {q:model(a).fit(X.iloc[tr],y[tr]) for q,a in [('q10',.1),('q50',.5),('q90',.9)]}
    def interval(ix):
        lo, mid, hi = (precio(models[q], X.iloc[ix]) for q in ('q10','q50','q90'))
        return np.minimum(lo, mid), mid, np.maximum(hi, mid)
    low_cal,_,high_cal = interval(ca)
    adjustment = correccion_conformal(real[ca], low_cal, high_cal)
    lo, pred, hi = interval(te)
    lo,hi = np.maximum(0,lo-adjustment),hi+adjustment
    evaluation = {'modelo':metricas(real[te], pred),
                  'baseline_ciudad':metricas(real[te],base_city),
                  'baseline_distrito':metricas(real[te],base_district),
                  'cobertura_2018':round(float(((real[te]>=lo)&(real[te]<=hi)).mean()),4),
                  'ancho_mediano_2018_eur':round(float(np.median(hi-lo)),2),
                  'n_train':len(tr),'n_calibracion':len(ca),'n_test':len(te)}
    # Ablación de localización: cuánto aporta frente al mismo algoritmo sin ella.
    geo_columns=['LATITUDE','LONGITUDE','DISTANCE_TO_CITY_CENTER','DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA','DISTANCE_TO_DIAGONAL','DISTANCE_TO_BLASCO']
    without_geo=X.drop(columns=geo_columns,errors='ignore')
    geo_ablation=model(.5,200).fit(without_geo.iloc[tr],y[tr])
    evaluation['ablacion_sin_localizacion']=metricas(real[te],precio(geo_ablation,without_geo.iloc[te]))
    # La interfaz desconoce muchos atributos catastrales. Medir el mismo modelo
    # usando para ellos únicamente las medianas del conjunto de entrenamiento.
    user_fields={'CONSTRUCTEDAREA','ROOMNUMBER','BATHNUMBER','CADCONSTRUCTIONYEAR',
                 'HASLIFT','HASTERRACE','HASPARKINGSPACE','HASAIRCONDITIONING',
                 'LATITUDE','LONGITUDE','DISTANCE_TO_METRO','DISTANCE_TO_CITY_CENTER',
                 'ANTIGUEDAD_2018','AREA_POR_HABITACION'}
    app_x=X.iloc[te].copy()
    hidden=[c for c in app_x if c not in user_fields]
    medians=X.iloc[tr].median(numeric_only=True).fillna(0)
    for c in hidden:app_x[c]=medians[c]
    evaluation['simulacion_campos_app']=metricas(real[te],precio(models['q50'],app_x))
    # Experimento espacial distinto: el modelo NO ha visto estos distritos.
    excluded = sorted(df.DISTRITO.unique())[::5]
    si = np.flatnonzero(df.DISTRITO.isin(excluded).to_numpy())
    st = np.flatnonzero((~df.DISTRITO.isin(excluded) &
                         ~df.ASSETID.astype(str).isin(set(groups[si]))).to_numpy())
    sm = model(.5,200).fit(X.iloc[st],y[st])
    evaluation['espacial_distritos_nuevos'] = metricas(real[si],precio(sm,X.iloc[si])) | {'distritos':excluded}
    # Experimento temporal: anuncios Q4 cuyos activos no existían antes.
    q4 = df.PERIOD.eq(201812).to_numpy()
    fi = np.flatnonzero(q4 & ~pd.Series(groups).isin(set(groups[~q4])).to_numpy())
    pi = np.flatnonzero(~q4)
    if len(fi)>100:
        tm = model(.5,200).fit(X.iloc[pi],y[pi])
        evaluation['temporal_q4_nuevos'] = metricas(real[fi],precio(tm,X.iloc[fi]))
    # Diagnóstico por distrito basado EXCLUSIVAMENTE en test.
    report = pd.DataFrame({'d':df.DISTRITO.iloc[te].to_numpy(),'real':real[te],
                           'pred':pred,'covered':(real[te]>=lo)&(real[te]<=hi)})
    errors = {}
    for d,s in report.groupby('d'):
        if len(s)>=30:
            errors[d] = metricas(s.real,s.pred) | {'cobertura_2018':round(float(s.covered.mean()),4)}
    # Referencias observadas SOLO en train, no inventar ventas ni ofertas actuales.
    sample = df.iloc[tr][['DISTRITO','PRICE','CONSTRUCTEDAREA','ROOMNUMBER','BATHNUMBER','CADCONSTRUCTIONYEAR']]
    sample = pd.concat([z.sample(min(len(z),100),random_state=42)
                        for _,z in sample.groupby('DISTRITO')],ignore_index=True)
    comps = sample.rename(columns={'PRICE':'precio_2018_eur'}).reset_index(drop=True).to_dict('records')
    for q,m in models.items(): m.save_model(str(OUT/f'{city.lower()}_{q}.ubj'))
    mean18 = df.iloc[tr].groupby('DISTRITO').UNITPRICE.mean()
    meta = {'ciudad':city,'features':list(X.columns),
            'medianas_train':X.iloc[tr].median(numeric_only=True).fillna(0).to_dict(),
            'centroides':centroids,'n_anuncios':len(df),'n_activos':int(df.ASSETID.nunique()),
            'distritos':{d:{'media_2018_eur_m2':float(v),'n_train':int((df.DISTRITO.iloc[tr]==d).sum())} for d,v in mean18.items()},
            'ajuste_calibracion_eur':adjustment,'evaluacion':evaluation,'flujo':flow,
            'fecha_base':'2018','validacion_precios_actuales':False}
    save(f'{city.lower()}_meta.json',meta)
    save(f'{city.lower()}_error_distrito.json',errors)
    save(f'{city.lower()}_comparables.json',comps)
    print(city,json.dumps(evaluation,ensure_ascii=False),flush=True)
    return evaluation

if __name__=='__main__':
    cities=sys.argv[1:] or list(CITIES)
    if any(c not in CITIES for c in cities): raise SystemExit(f'Ciudades: {CITIES}')
    results={c:run(c) for c in cities}
    path=OUT/'resumen.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    previous.update(results)
    save('resumen.json',previous)
