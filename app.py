import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, math
from pathlib import Path

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="VALORA · Valoración automatizada de activos residenciales",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TOKENS ────────────────────────────────────────────────────────────────
CANVAS = "#F2F4F3"
SURFACE = "#FFFFFF"
INK = "#12211E"
MUTED = "#6B7A76"
LINE = "#D3DAD7"
PINE = "#0B4F3F"
PINE_MID = "#2E6B5A"
PINE_SOFT = "#7FA69A"
PINE_WASH = "#C7D6D0"


@st.cache_resource
def cargar():
    return (
        joblib.load(BASE / "modelo_valoracion.joblib"),
        json.loads((BASE / "resultados_tfm.json").read_text(encoding="utf-8")),
    )


model, meta = cargar()

# ── ESTILOS ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Spectral:wght@300;400;600&display=swap');

header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}

.stApp {{ background: {CANVAS}; }}
.block-container {{ padding: 2.2rem 3rem 4rem 3rem !important; max-width: 1280px; }}

html, body, [class*="css"] {{ font-family: 'Archivo', sans-serif; color: {INK}; }}

/* Regla tipográfica: toda magnitud medida en monoespaciada */
.m {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
.s {{ font-family: 'Spectral', serif; }}

/* — Masthead — */
.masthead {{
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 1px solid {INK}; padding-bottom: 12px; margin-bottom: 34px;
}}
.wordmark {{
    font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 1.02rem;
    letter-spacing: .34em; color: {INK}; text-transform: uppercase;
}}
.masthead .sub {{
    font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
    letter-spacing: .12em; color: {MUTED}; text-transform: uppercase;
}}

/* — Etiquetas de sección — */
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: .66rem; letter-spacing: .2em;
    text-transform: uppercase; color: {MUTED}; display: flex; align-items: center;
    gap: 12px; margin: 0 0 14px 0;
}}
.eyebrow::after {{ content: ""; flex: 1; height: 1px; background: {LINE}; }}

/* — Lectura principal — */
.readout {{ animation: rise .5s cubic-bezier(.2,.7,.3,1) both; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(7px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{ .readout {{ animation: none; }} }}

.figure {{
    font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
    font-size: 4.4rem; font-weight: 500; letter-spacing: -.035em;
    color: {PINE}; line-height: .96;
}}
.figure .cur {{ font-size: 2.1rem; font-weight: 400; color: {PINE_MID}; margin-left: 6px; letter-spacing: 0; }}
.derived {{ margin-top: 16px; display: flex; gap: 26px; flex-wrap: wrap; }}
.derived div {{ border-left: 2px solid {PINE_WASH}; padding-left: 11px; }}
.derived .v {{
    font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
    font-size: 1.02rem; color: {INK};
}}
.derived .k {{
    font-family: 'IBM Plex Mono', monospace; font-size: .62rem; letter-spacing: .14em;
    text-transform: uppercase; color: {MUTED}; margin-top: 3px;
}}

/* — Superficies — */
.panel {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 3px;
    padding: 20px 22px; height: 100%;
}}
.lede {{
    font-family: 'Spectral', serif; font-size: .93rem; line-height: 1.62;
    color: {MUTED}; font-weight: 300;
}}
.lede strong {{ color: {INK}; font-weight: 600; }}

/* — Barra de diagnóstico — */
.diag {{ display: flex; border-top: 1px solid {INK}; border-bottom: 1px solid {LINE}; }}
.diag > div {{ flex: 1; padding: 15px 0 14px 0; border-right: 1px solid {LINE}; }}
.diag > div:last-child {{ border-right: none; }}
.diag .v {{
    font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
    font-size: 1.32rem; color: {INK}; letter-spacing: -.02em;
}}
.diag .k {{
    font-family: 'IBM Plex Mono', monospace; font-size: .62rem; letter-spacing: .14em;
    text-transform: uppercase; color: {MUTED}; margin-top: 5px;
}}

.foot {{
    border-top: 1px solid {LINE}; margin-top: 42px; padding-top: 16px;
    display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace; font-size: .64rem;
    letter-spacing: .1em; text-transform: uppercase; color: {MUTED};
}}

.idle {{
    border: 1px dashed {LINE}; border-radius: 3px; padding: 44px 30px; text-align: center;
    background: rgba(255,255,255,.5);
}}
.idle .s {{ font-size: 1.12rem; color: {MUTED}; font-weight: 300; }}

/* — Panel lateral — */
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem !important; }}
.sb-title {{
    font-family: 'IBM Plex Mono', monospace; font-size: .66rem; letter-spacing: .2em;
    text-transform: uppercase; color: {INK}; padding-bottom: 9px;
    border-bottom: 1px solid {INK}; margin-bottom: 4px;
}}
.sb-group {{
    font-family: 'IBM Plex Mono', monospace; font-size: .61rem; letter-spacing: .16em;
    text-transform: uppercase; color: {MUTED}; margin: 20px 0 2px 0;
}}
[data-testid="stWidgetLabel"] p {{
    font-family: 'Archivo', sans-serif !important; font-size: .78rem !important;
    color: {INK} !important; font-weight: 400 !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0; }}

div.stButton > button {{
    background: {PINE}; color: {SURFACE}; border: 1px solid {PINE}; border-radius: 3px;
    font-family: 'Archivo', sans-serif; font-weight: 600; font-size: .74rem;
    letter-spacing: .16em; text-transform: uppercase; padding: .72rem 1rem; width: 100%;
    transition: background .16s ease;
}}
div.stButton > button:hover {{ background: {INK}; border-color: {INK}; color: {SURFACE}; }}
div.stButton > button:focus-visible {{ outline: 2px solid {PINE}; outline-offset: 2px; }}

[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 3px; background: {SURFACE}; }}
[data-testid="stExpander"] summary p {{
    font-family: 'IBM Plex Mono', monospace !important; font-size: .68rem !important;
    letter-spacing: .14em; text-transform: uppercase; color: {MUTED} !important;
}}

@media (max-width: 900px) {{
    .block-container {{ padding: 1.4rem 1.2rem 3rem 1.2rem !important; }}
    .figure {{ font-size: 2.9rem; }}
    .diag {{ flex-wrap: wrap; }}
    .diag > div {{ flex: 1 0 50%; border-bottom: 1px solid {LINE}; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ── FORMATO ───────────────────────────────────────────────────────────────
def num(x, dec=0):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "·").replace(".", ",").replace("·", ".")


def eur(x, dec=0):
    return f"{num(x, dec)} €"


def compact(v):
    if v >= 1_000_000:
        return f"{num(v/1_000_000, 1)}M"
    return f"{num(v/1000, 0)}k"


# ── PANEL LATERAL ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Parámetros del activo</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-group">Localización</div>', unsafe_allow_html=True)
    city = st.selectbox("Mercado", ["Madrid", "Barcelona"], label_visibility="visible")
    d_center = st.slider("Distancia al centro (km)", 0.0, 30.0, 3.0, 0.1)
    d_metro = st.slider("Distancia al metro (km)", 0.0, 10.0, 0.4, 0.1)

    lat_def, lon_def = (40.4168, -3.7038) if city == "Madrid" else (41.3874, 2.1686)
    with st.expander("Coordenadas"):
        lat = st.number_input("Latitud", value=float(lat_def), format="%.5f")
        lon = st.number_input("Longitud", value=float(lon_def), format="%.5f")

    st.markdown('<div class="sb-group">Superficie y distribución</div>', unsafe_allow_html=True)
    area = st.slider("Superficie construida (m²)", 20, 500, 90)
    c1, c2 = st.columns(2)
    rooms = c1.number_input("Habitaciones", 0, 15, 3)
    baths = c2.number_input("Baños", 0, 10, 2)
    year = st.slider("Año de construcción", 1900, 2018, 1970)

    st.markdown('<div class="sb-group">Dotaciones</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    lift = e1.checkbox("Ascensor", value=True)
    terrace = e2.checkbox("Terraza")
    parking = e1.checkbox("Parking")
    air = e2.checkbox("Climatización")
    pool = e1.checkbox("Piscina")
    doorman = e2.checkbox("Portería")

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    calcular = st.button("Calcular valoración")


# ── MODELO ────────────────────────────────────────────────────────────────
def fila(area_m2, dist_centro):
    d = {c: 0 for c in meta["features"]}
    d.update({
        "CITY": city, "CONSTRUCTEDAREA": area_m2, "ROOMNUMBER": rooms,
        "BATHNUMBER": baths, "CADCONSTRUCTIONYEAR": year,
        "DISTANCE_TO_CITY_CENTER": dist_centro, "DISTANCE_TO_METRO": d_metro,
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


def predecir_lote(filas):
    X = pd.DataFrame(filas)
    return np.expm1(model.predict(X))


# ── GRÁFICOS SVG ──────────────────────────────────────────────────────────
def escala_svg(pred, bajo, alto):
    """Escala calibrada: la banda se ve proporcional al valor estimado."""
    W, H = 1000, 96
    x0, x1, base = 8, W - 8, 52
    top = max(alto * 1.12, 1)
    axis_max = math.ceil(top / 50_000) * 50_000

    def px(v):
        return x0 + (v / axis_max) * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none" '
         f'style="display:block;overflow:visible">']
    # banda
    p.append(f'<rect x="{px(bajo):.1f}" y="{base-19}" width="{max(px(alto)-px(bajo),1):.1f}" '
             f'height="38" fill="{PINE}" opacity="0.09"/>')
    for v, lab in ((bajo, "inferior"), (alto, "superior")):
        p.append(f'<line x1="{px(v):.1f}" y1="{base-19}" x2="{px(v):.1f}" y2="{base+19}" '
                 f'stroke="{PINE_SOFT}" stroke-width="1.5"/>')
    # eje
    p.append(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="{LINE}" stroke-width="1"/>')
    # ticks
    step = axis_max / 10
    for i in range(11):
        v = step * i
        h = 8 if i % 2 == 0 else 4
        p.append(f'<line x1="{px(v):.1f}" y1="{base}" x2="{px(v):.1f}" y2="{base+h}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
        if i % 2 == 0:
            p.append(f'<text x="{px(v):.1f}" y="{base+23}" fill="{MUTED}" font-size="10.5" '
                     f'font-family="IBM Plex Mono, monospace" text-anchor="middle">{compact(v)}</text>')
    # marcador
    mx = px(pred)
    p.append(f'<line x1="{mx:.1f}" y1="{base-30}" x2="{mx:.1f}" y2="{base+12}" '
             f'stroke="{PINE}" stroke-width="2.5"/>')
    p.append(f'<polygon points="{mx-6:.1f},{base-30} {mx+6:.1f},{base-30} {mx:.1f},{base-21}" fill="{PINE}"/>')
    p.append("</svg>")
    return "".join(p)


def curva_svg(xs, ys, x_now, y_now, unidad):
    W, H = 520, 190
    L, R, T, B = 10, W - 10, 20, 44
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    if xmax == xmin:
        xmax = xmin + 1
    pad = (ymax - ymin) * 0.16 or 1
    ymin, ymax = ymin - pad, ymax + pad

    def px(v):
        return L + (v - xmin) / (xmax - xmin) * (R - L)

    def py(v):
        return T + (1 - (v - ymin) / (ymax - ymin)) * (H - B - T)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;overflow:visible">']
    for i in range(1, 4):
        gy = T + (H - B - T) * i / 4
        p.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{R}" y2="{gy:.1f}" stroke="{LINE}" '
                 f'stroke-width="1" stroke-dasharray="2 4"/>')
    p.append(f'<polygon points="{px(xmin):.1f},{H-B} {pts} {px(xmax):.1f},{H-B}" '
             f'fill="{PINE}" opacity="0.07"/>')
    p.append(f'<polyline points="{pts}" fill="none" stroke="{PINE}" stroke-width="2" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    p.append(f'<line x1="{px(x_now):.1f}" y1="{T-6}" x2="{px(x_now):.1f}" y2="{H-B}" '
             f'stroke="{PINE_MID}" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(f'<circle cx="{px(x_now):.1f}" cy="{py(y_now):.1f}" r="4.5" fill="{PINE}" '
             f'stroke="{SURFACE}" stroke-width="2"/>')
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{R}" y2="{H-B}" stroke="{INK}" stroke-width="1"/>')
    for v, anchor in ((xmin, "start"), (xmax, "end")):
        p.append(f'<text x="{px(v):.1f}" y="{H-B+17}" fill="{MUTED}" font-size="11" '
                 f'font-family="IBM Plex Mono, monospace" text-anchor="{anchor}">'
                 f'{num(v,1).rstrip("0").rstrip(",") if unidad=="km" else num(v)}{unidad}</text>')
    p.append(f'<text x="{L}" y="{T-6}" fill="{MUTED}" font-size="11" '
             f'font-family="IBM Plex Mono, monospace">{compact(max(ys))} €</text>')
    p.append("</svg>")
    return "".join(p)


def barras_svg(items):
    W = 520
    fila_h, gap = 30, 8
    H = len(items) * (fila_h + gap)
    top = max(v for _, v in items)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;overflow:visible">']
    for i, (nombre, val) in enumerate(items):
        y = i * (fila_h + gap)
        w = max((val / top) * (W - 250), 2)
        op = 0.30 + 0.70 * (val / top)
        p.append(f'<text x="0" y="{y+15}" fill="{INK}" font-size="12.5" '
                 f'font-family="Archivo, sans-serif">{nombre}</text>')
        p.append(f'<rect x="196" y="{y+4}" width="{w:.1f}" height="15" fill="{PINE}" '
                 f'opacity="{op:.2f}" rx="1"/>')
        p.append(f'<text x="{196+w+9:.1f}" y="{y+16}" fill="{MUTED}" font-size="11" '
                 f'font-family="IBM Plex Mono, monospace">{val:.3f}</text>')
        p.append(f'<line x1="0" y1="{y+fila_h+2}" x2="{W}" y2="{y+fila_h+2}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
    p.append("</svg>")
    return "".join(p)


ETIQUETAS = {
    "CONSTRUCTEDAREA": "Superficie construida", "LATITUDE": "Latitud",
    "LONGITUDE": "Longitud", "DISTANCE_TO_CITY_CENTER": "Distancia al centro",
    "DISTANCE_TO_METRO": "Distancia al metro", "BATHNUMBER": "Número de baños",
    "ROOMNUMBER": "Número de habitaciones", "HASLIFT": "Ascensor",
    "CADCONSTRUCTIONYEAR": "Año de construcción", "BUILDING_AGE": "Antigüedad",
    "AREA_PER_ROOM": "Superficie por habitación", "CADASTRALQUALITYID": "Calidad catastral",
    "FLOORCLEAN": "Planta", "AMENITIES_COUNT": "Dotaciones", "CITY": "Mercado",
    "FLOOR_RATIO": "Posición en el edificio", "CADDWELLINGCOUNT": "Viviendas del edificio",
    "CADMAXBUILDINGFLOOR": "Altura del edificio", "HASTERRACE": "Terraza",
    "HASPARKINGSPACE": "Plaza de garaje",
}


def bonito(f):
    limpio = f.replace("num__", "").replace("cat__", "").split("_Madrid")[0]
    return ETIQUETAS.get(limpio, limpio.replace("_", " ").capitalize())


# ── CABECERA ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="masthead"><div class="wordmark">Valora</div>'
    '<div class="sub">Valoración automatizada · Mercado residencial español</div></div>',
    unsafe_allow_html=True,
)

# ── VALORACIÓN ────────────────────────────────────────────────────────────
if calcular:
    precio = float(predecir_lote([fila(area, d_center)])[0])
    q = float(meta["p90_abs_error_eur"])
    bajo, alto = max(0.0, precio - q), precio + q
    holgura = q / precio * 100

    st.markdown('<div class="eyebrow">Valoración estimada</div>', unsafe_allow_html=True)

    izq, der = st.columns([1.15, 1], gap="large")
    with izq:
        st.markdown(
            f'<div class="readout"><div class="figure">{num(precio)}<span class="cur">€</span></div>'
            f'<div class="derived">'
            f'<div><div class="v">{eur(precio/area)}</div><div class="k">por m²</div></div>'
            f'<div><div class="v">{num(area)} m²</div><div class="k">superficie</div></div>'
            f'<div><div class="v">{2018-year}</div><div class="k">años</div></div>'
            f'<div><div class="v">{city}</div><div class="k">mercado</div></div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with der:
        st.markdown(
            f'<div class="lede" style="padding-top:6px">La estimación se acompaña de una '
            f'banda de <strong>±{eur(q)}</strong>, equivalente al <strong>{holgura:.0f}%</strong> '
            f'del valor. Nueve de cada diez viviendas del conjunto de contraste caen dentro de '
            f'ese margen.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Intervalo sobre escala absoluta</div>', unsafe_allow_html=True)
    st.markdown(escala_svg(precio, bajo, alto), unsafe_allow_html=True)
    st.markdown(
        f'<div class="m" style="font-size:.72rem;color:{MUTED};letter-spacing:.1em;'
        f'text-transform:uppercase;margin-top:8px">'
        f'{eur(bajo)} &nbsp;—&nbsp; {eur(alto)}</div>',
        unsafe_allow_html=True,
    )

    # Curvas de respuesta
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")

    xs_a = np.linspace(max(20, area - 55), min(500, area + 55), 22)
    ys_a = predecir_lote([fila(float(a), d_center) for a in xs_a])
    with g1:
        st.markdown('<div class="eyebrow">Respuesta a la superficie</div>', unsafe_allow_html=True)
        st.markdown(curva_svg(list(xs_a), list(ys_a), area, precio, " m²"), unsafe_allow_html=True)
        pend = (float(predecir_lote([fila(min(500.0, area + 10.0), d_center)])[0]) - precio) / 10
        st.markdown(
            f'<div class="lede" style="font-size:.85rem;margin-top:6px">Cada m² adicional en '
            f'torno a los {area} m² añade <strong>{eur(pend)}</strong>.</div>',
            unsafe_allow_html=True,
        )

    xs_d = np.linspace(0.2, min(20.0, max(8.0, d_center + 6)), 22)
    ys_d = predecir_lote([fila(area, float(d)) for d in xs_d])
    with g2:
        st.markdown('<div class="eyebrow">Respuesta a la centralidad</div>', unsafe_allow_html=True)
        st.markdown(curva_svg(list(xs_d), list(ys_d), d_center, precio, " km"), unsafe_allow_html=True)
        caida = (float(ys_d[0]) - float(ys_d[-1])) / float(ys_d[0]) * 100
        st.markdown(
            f'<div class="lede" style="font-size:.85rem;margin-top:6px">Alejarse del centro hasta '
            f'{num(xs_d[-1],1)} km reduce el valor un <strong>{caida:.0f}%</strong>.</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="idle"><div class="s" style="font-family:Spectral,serif">'
        "Define los parámetros del activo en el panel lateral para obtener una valoración."
        "</div></div>",
        unsafe_allow_html=True,
    )

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────
st.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Rendimiento del modelo</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="diag">'
    f'<div><div class="v">{meta["r2_test"]:.3f}</div><div class="k">R² fuera de muestra</div></div>'
    f'<div><div class="v">{eur(meta["mae_test"])}</div><div class="k">Error medio absoluto</div></div>'
    f'<div><div class="v">{eur(meta["rmse_test"])}</div><div class="k">RMSE</div></div>'
    f'<div><div class="v">{num(meta["n_final"])}</div><div class="k">Observaciones</div></div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
b1, b2 = st.columns([1.5, 1], gap="large")

with b1:
    st.markdown('<div class="eyebrow">Factores determinantes del precio</div>', unsafe_allow_html=True)
    items = [(bonito(d["feature"]), float(d["mean_abs_shap"])) for d in meta["top_shap"][:8]]
    st.markdown(barras_svg(items), unsafe_allow_html=True)
    st.markdown(
        '<div class="lede" style="font-size:.85rem;margin-top:10px">Contribución media absoluta '
        "de cada variable a la predicción, medida con valores de Shapley sobre el conjunto de "
        "contraste.</div>",
        unsafe_allow_html=True,
    )

with b2:
    st.markdown('<div class="eyebrow">Cobertura</div>', unsafe_allow_html=True)
    tot = meta["n_final"]
    pm = meta["n_madrid"] / tot
    st.markdown(
        f'<svg viewBox="0 0 400 26" width="100%" style="display:block">'
        f'<rect x="0" y="0" width="{pm*400:.1f}" height="26" fill="{PINE}"/>'
        f'<rect x="{pm*400:.1f}" y="0" width="{(1-pm)*400:.1f}" height="26" fill="{PINE_SOFT}"/>'
        f"</svg>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;margin-top:10px">'
        f'<div><div class="m" style="font-size:1.02rem">{num(meta["n_madrid"])}</div>'
        f'<div class="m" style="font-size:.62rem;letter-spacing:.14em;color:{MUTED};'
        f'text-transform:uppercase;margin-top:2px">Madrid · {pm:.0%}</div></div>'
        f'<div style="text-align:right"><div class="m" style="font-size:1.02rem">'
        f'{num(meta["n_barcelona"])}</div>'
        f'<div class="m" style="font-size:.62rem;letter-spacing:.14em;color:{MUTED};'
        f'text-transform:uppercase;margin-top:2px">Barcelona · {1-pm:.0%}</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("Metodología"):
        st.markdown(
            f'<div class="lede" style="font-size:.86rem">'
            f"Modelo de gradient boosting sobre {len(meta['features'])} variables de producto, "
            f"localización y registro catastral, con hiperparámetros optimizados por búsqueda "
            f"bayesiana. La variable objetivo se modela en escala logarítmica. El intervalo "
            f"mostrado corresponde al percentil 90 del error absoluto fuera de muestra."
            f"</div>",
            unsafe_allow_html=True,
        )

# ── PIE ───────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="foot"><div>Valora · Motor de valoración automatizada</div>'
    f"<div>Estimación orientativa · No sustituye a una tasación homologada</div></div>",
    unsafe_allow_html=True,
)
