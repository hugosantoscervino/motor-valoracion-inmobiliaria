import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Valorador Inmobiliario · Madrid y Barcelona",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(BASE / "modelo_valoracion.joblib")
    meta = json.loads((BASE / "resultados_tfm.json").read_text(encoding="utf-8"))
    return modelo, meta


model, meta = cargar_modelo()

# ──────────────────────────────── ESTILOS ────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer {visibility: hidden;}

.hero {
    background: linear-gradient(120deg, #0D1B2A 0%, #12324A 55%, #0D9488 140%);
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 26px;
    color: #fff;
    box-shadow: 0 12px 32px rgba(13,27,42,.28);
}
.hero h1 {
    font-size: 2.15rem; font-weight: 800; margin: 0 0 6px 0;
    letter-spacing: -.02em; line-height: 1.15; color: #fff;
}
.hero p { font-size: 1rem; margin: 0; color: #9FD8CE; }
.hero .chips { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
    border-radius: 40px; padding: 5px 14px; font-size: .78rem; color: #E4F3F0;
    backdrop-filter: blur(4px);
}

.price-card {
    background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
    border-radius: 20px; padding: 30px 34px; color: #fff; text-align: center;
    box-shadow: 0 12px 30px rgba(13,148,136,.32);
}
.price-card .label {
    font-size: .8rem; text-transform: uppercase; letter-spacing: .14em;
    color: #B6E6DF; margin-bottom: 6px;
}
.price-card .value { font-size: 3.1rem; font-weight: 800; line-height: 1.05; letter-spacing: -.03em; }
.price-card .band { font-size: .9rem; color: #CFEDE8; margin-top: 10px; }

.stat {
    background: #F6FAFA; border: 1px solid #E2ECEC; border-radius: 14px;
    padding: 16px 18px; height: 100%;
}
.stat .k { font-size: 1.5rem; font-weight: 700; color: #0D1B2A; line-height: 1.1; }
.stat .l { font-size: .76rem; color: #6B7C7B; margin-top: 3px; }

.section-title {
    font-size: 1.05rem; font-weight: 600; color: #0D1B2A;
    margin: 26px 0 10px 0; padding-bottom: 7px; border-bottom: 2px solid #E2ECEC;
}
.note { font-size: .8rem; color: #7A8A89; line-height: 1.55; }

div.stButton > button {
    background: #0D9488; color: #fff; border: none; border-radius: 11px;
    padding: .62rem 1rem; font-weight: 600; font-size: .95rem; width: 100%;
    transition: all .18s ease;
}
div.stButton > button:hover {
    background: #0F766E; color: #fff; transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(13,148,136,.3);
}
section[data-testid="stSidebar"] { background: #FAFCFC; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────── CABECERA ────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <h1>Valorador Inmobiliario</h1>
  <p>Estimación de precio de vivienda en Madrid y Barcelona mediante Machine Learning</p>
  <div class="chips">
    <span class="chip">{meta['modelo_final']}</span>
    <span class="chip">R² = {meta['r2_test']:.3f}</span>
    <span class="chip">{meta['n_final']:,} viviendas</span>
    <span class="chip">{len(meta['features'])} variables</span>
  </div>
</div>
""".replace(",", "."),
    unsafe_allow_html=True,
)

# ──────────────────────────────── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.markdown("### Características del inmueble")

    city = st.selectbox("Ciudad", ["Madrid", "Barcelona"])

    st.markdown("**Dimensiones**")
    area = st.slider("Superficie construida (m²)", 20, 500, 90)
    c1, c2 = st.columns(2)
    rooms = c1.number_input("Habitaciones", 0, 15, 3)
    baths = c2.number_input("Baños", 0, 10, 2)
    year = st.slider("Año de construcción", 1900, 2018, 1970)

    st.markdown("**Ubicación**")
    d_center = st.slider("Distancia al centro (km)", 0.0, 30.0, 3.0, 0.1)
    d_metro = st.slider("Distancia al metro (km)", 0.0, 10.0, 0.4, 0.1)

    lat_def, lon_def = (40.4168, -3.7038) if city == "Madrid" else (41.3874, 2.1686)
    with st.expander("Coordenadas exactas"):
        lat = st.number_input("Latitud", value=float(lat_def), format="%.5f")
        lon = st.number_input("Longitud", value=float(lon_def), format="%.5f")

    st.markdown("**Extras**")
    e1, e2 = st.columns(2)
    lift = e1.checkbox("Ascensor", value=True)
    terrace = e2.checkbox("Terraza")
    parking = e1.checkbox("Parking")
    air = e2.checkbox("Aire acond.")
    pool = e1.checkbox("Piscina")
    doorman = e2.checkbox("Portero")

    st.markdown("")
    calcular = st.button("Estimar precio")


# ──────────────────────────────── LÓGICA ────────────────────────────────
def construir_fila(area_m2):
    d = {c: 0 for c in meta["features"]}
    d.update({
        "CITY": city, "CONSTRUCTEDAREA": area_m2, "ROOMNUMBER": rooms,
        "BATHNUMBER": baths, "CADCONSTRUCTIONYEAR": year,
        "DISTANCE_TO_CITY_CENTER": d_center, "DISTANCE_TO_METRO": d_metro,
        "LATITUDE": lat, "LONGITUDE": lon,
        "HASLIFT": int(lift), "HASTERRACE": int(terrace),
        "HASPARKINGSPACE": int(parking), "HASAIRCONDITIONING": int(air),
        "HASSWIMMINGPOOL": int(pool), "HASDOORMAN": int(doorman),
        "AMENITYID": 3, "CADMAXBUILDINGFLOOR": 6, "CADDWELLINGCOUNT": 20,
        "CADASTRALQUALITYID": 5, "BUILTTYPEID_3": 1, "FLOORCLEAN": 2,
        "FLATLOCATIONID": 1,
    })
    d["BUILDING_AGE"] = max(0, 2018 - year)
    d["AREA_PER_ROOM"] = area_m2 / max(rooms, 1)
    d["FLOOR_RATIO"] = d["FLOORCLEAN"] / max(d["CADMAXBUILDINGFLOOR"], 1)
    d["NEAR_METRO_500M"] = int(d_metro <= 0.5)
    d["AMENITIES_COUNT"] = sum([lift, terrace, parking, air, pool, doorman])
    return {c: d.get(c) for c in meta["features"]}


def predecir(area_m2):
    X = pd.DataFrame([construir_fila(area_m2)])
    return float(np.expm1(model.predict(X))[0])


def eur(x):
    return f"{x:,.0f} €".replace(",", ".")


# ──────────────────────────────── RESULTADO ────────────────────────────────
if calcular:
    precio = predecir(area)
    q = float(meta["p90_abs_error_eur"])
    bajo, alto = max(0, precio - q), precio + q

    izq, der = st.columns([1.05, 1])

    with izq:
        st.markdown(
            f"""
        <div class="price-card">
          <div class="label">Precio estimado</div>
          <div class="value">{eur(precio)}</div>
          <div class="band">Banda de confianza del 90%<br>{eur(bajo)} — {eur(alto)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)
        a.markdown(
            f'<div class="stat"><div class="k">{eur(precio/area)}</div>'
            f'<div class="l">por m²</div></div>', unsafe_allow_html=True)
        b.markdown(
            f'<div class="stat"><div class="k">{area} m²</div>'
            f'<div class="l">superficie</div></div>', unsafe_allow_html=True)
        c.markdown(
            f'<div class="stat"><div class="k">{2018-year}</div>'
            f'<div class="l">años de antigüedad</div></div>', unsafe_allow_html=True)

    with der:
        st.markdown('<div class="section-title">Ubicación</div>', unsafe_allow_html=True)
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=12, size=90)

    # Sensibilidad a la superficie
    st.markdown('<div class="section-title">Cómo evoluciona el precio con la superficie</div>',
                unsafe_allow_html=True)
    rango = np.linspace(max(20, area - 60), min(500, area + 60), 25)
    curva = pd.DataFrame({
        "Superficie (m²)": rango,
        "Precio estimado (€)": [predecir(float(a_)) for a_ in rango],
    }).set_index("Superficie (m²)")
    st.line_chart(curva, height=280, color="#0D9488")
    st.markdown(
        f'<p class="note">Manteniendo el resto de características fijas, '
        f'cada m² adicional en torno a los {area} m² añade aproximadamente '
        f'{eur((predecir(min(500.0, area+10.0)) - precio)/10)} al precio estimado.</p>',
        unsafe_allow_html=True)

else:
    st.info("Ajusta las características en el panel de la izquierda y pulsa **Estimar precio**.")

# ──────────────────────────────── EL MODELO ────────────────────────────────
st.markdown('<div class="section-title">El modelo</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
for col, valor, etiqueta in [
    (m1, f"{meta['r2_test']:.3f}", "R² en test"),
    (m2, eur(meta["mae_test"]), "Error medio absoluto"),
    (m3, eur(meta["rmse_test"]), "RMSE"),
    (m4, f"{meta['n_final']:,}".replace(",", "."), "Viviendas analizadas"),
]:
    col.markdown(
        f'<div class="stat"><div class="k">{valor}</div><div class="l">{etiqueta}</div></div>',
        unsafe_allow_html=True)

g1, g2 = st.columns([1.4, 1])

with g1:
    st.markdown('<div class="section-title">Variables más influyentes (SHAP)</div>',
                unsafe_allow_html=True)
    shap_df = pd.DataFrame(meta["top_shap"])
    shap_df["feature"] = (shap_df["feature"]
                          .str.replace("num__", "", regex=False)
                          .str.replace("cat__", "", regex=False)
                          .str.replace("_", " ", regex=False)
                          .str.title())
    shap_df = shap_df.set_index("feature").rename(
        columns={"mean_abs_shap": "Impacto medio"})
    st.bar_chart(shap_df, height=330, horizontal=True, color="#0D9488")

with g2:
    st.markdown('<div class="section-title">Composición de la muestra</div>',
                unsafe_allow_html=True)
    ciudades = pd.DataFrame(
        {"Viviendas": [meta["n_madrid"], meta["n_barcelona"]]},
        index=["Madrid", "Barcelona"])
    st.bar_chart(ciudades, height=200, color="#0D1B2A")
    st.markdown(
        f'<p class="note">Madrid aporta {meta["n_madrid"]/meta["n_final"]:.0%} '
        f'de la muestra y Barcelona el {meta["n_barcelona"]/meta["n_final"]:.0%}.</p>',
        unsafe_allow_html=True)

with st.expander("Metodología y limitaciones"):
    st.markdown(f"""
**Modelo.** {meta['modelo_final']}, seleccionado tras comparar Regresión Lineal,
Random Forest, XGBoost y LightGBM. Los hiperparámetros se optimizaron con Optuna
mediante validación cruzada. La variable objetivo se modela en escala logarítmica
y se revierte con `expm1` al presentar el resultado.

**Precisión.** R² de {meta['r2_test']:.3f} sobre el conjunto de test, con un error
medio absoluto de {eur(meta['mae_test'])}. La banda mostrada corresponde al
percentil 90 del error absoluto ({eur(meta['p90_abs_error_eur'])}), es decir, 9 de
cada 10 viviendas del test caen dentro de ese margen.

**Limitaciones.** Los datos proceden de anuncios de 2018, por lo que las
estimaciones reflejan el mercado de ese año y no los precios actuales. El modelo
cubre únicamente Madrid y Barcelona. No sustituye a una tasación oficial.
""")

st.markdown(
    '<p class="note" style="text-align:center;margin-top:26px">'
    'TFM · Máster en Big Data, Ciencia de Datos y Business Analytics · '
    'Universidad Complutense de Madrid</p>',
    unsafe_allow_html=True)
