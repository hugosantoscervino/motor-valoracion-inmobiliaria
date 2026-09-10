import json, math, base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb
import pydeck as pdk
from PIL import Image

BASE = Path(__file__).resolve().parent
ART = BASE / "artefactos"
if not (ART / "madrid_meta.json").exists() and (ART / "artefactos").exists():
    ART = ART / "artefactos"

st.set_page_config(page_title="© 2026 Aldaba",
                   page_icon="🚪",
                   layout="wide", initial_sidebar_state="collapsed")

# ── preferencias de accesibilidad (leídas antes de pintar los estilos) ──
_qp = st.query_params
def _qs(k, d):
    v = _qp.get(k)
    return v if v is not None else d
AC = st.session_state.get("alto_contraste", False)
TG = st.session_state.get("texto_grande", False)

if AC:
    BG, SURF, LINE = "#FFFFFF", "#FFFFFF", "#000000"
    INK, MUTE, SOFT = "#000000", "#1F1F1F", "#F0F0F0"
    ACC, ACC_DIM, BANDA, CREMA = "#004030", "#004030", "#00281E", "#FFFFFF"
    VERDE, ROJO, AMBAR = "#0B5C31", "#8C1D06", "#5C3B00"
    STOPS = [(0.00, (255, 255, 255)), (0.30, (186, 214, 200)),
             (0.65, (74, 140, 112)), (1.00, (0, 51, 38))]
else:
    BG, SURF, LINE = "#F4F2ED", "#FBFAF7", "#CFC8B9"
    INK, MUTE, SOFT = "#1A1D1B", "#575549", "#EAE9E1"
    ACC, ACC_DIM, BANDA, CREMA = "#0E5D4A", "#7FA697", "#0E5D4A", "#E9E4D8"
    VERDE, ROJO, AMBAR = "#0F6B3C", "#A33A22", "#7A5310"
    STOPS = [(0.00, (238, 234, 226)), (0.25, (198, 214, 202)), (0.50, (137, 180, 158)),
             (0.75, (52, 130, 104)), (1.00, (11, 77, 61))]

FS = 1.16 if TG else 1.0
CIUDADES = ["Madrid", "Barcelona", "Valencia"]
MAPA_W, MAPA_H, DLA, NY = 700, 480, 0.075, 48
NX = int(round(NY * MAPA_W / MAPA_H))
KM = 111.0


@st.cache_resource(show_spinner=False)
def cargar(ciudad):
    meta = json.loads((ART / f"{ciudad.lower()}_meta.json").read_text(encoding="utf-8"))
    mods = {q: xgb.Booster(model_file=str(ART / f"{ciudad.lower()}_{q}.ubj"))
            for q in ("q10", "q50", "q90")}
    return meta, mods, pd.DataFrame(meta["centroides"])


@st.cache_data(show_spinner=False)
def extra(nombre):
    p = ART / nombre
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def indice():
    p = BASE / "indice.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def resumen():
    p = ART / "resumen.json"
    if not p.exists():
        return {c: json.loads((ART / f"{c.lower()}_meta.json").read_text(encoding='utf-8'))
                for c in ['Madrid','Barcelona','Valencia'] if (ART / f"{c.lower()}_meta.json").exists()}
    raw = json.loads(p.read_text(encoding='utf-8'))
    # resumen.json puede contener los meta completos o solo un subconjunto de claves
    for c in list(raw.keys()):
        if 'r2_bloques' not in raw[c]:
            mp = ART / f"{c.lower()}_meta.json"
            if mp.exists(): raw[c] = json.loads(mp.read_text(encoding='utf-8'))
    return raw


IDX = indice()


def aplicar_indice(meta, ciudad):
    """Recalcula los factores por distrito desde indice.json, si existe."""
    if not IDX or ciudad not in IDX.get("ciudades", {}):
        return meta
    niv = IDX["ciudades"][ciudad]
    fc = niv["_ciudad"] / meta["unitprice_mediana_2018"]
    for d, v in meta["distritos"].items():
        n26 = niv.get(d)
        v["e2026"] = float(n26) if n26 else None
        v["factor"] = round(n26 / v["e2018"], 4) if n26 else round(fc, 4)
        v["estimado"] = n26 is None
    meta["nivel_ciudad_2026"] = niv["_ciudad"]
    meta["factor_ciudad"] = round(fc, 4)
    return meta


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def eur(x, dec=0):
    return f"{num(x, dec)} €"


def compact(v):
    return f"{num(v/1_000_000, 1)} M" if v >= 1_000_000 else f"{num(v/1000)} k"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"]{{display:none!important}}
.stApp{{background:{BG}}}
.block-container,[data-testid="stMainBlockContainer"],
section.main>div.block-container{{
 padding:0 2.6rem 5rem!important;max-width:min(1860px,95vw)!important;
 width:100%!important}}
html,body,[class*="css"]{{font-family:'Instrument Sans',sans-serif;color:{INK};
 font-size:{FS}rem}}
.m{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;
 color:{INK}}}

::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-thumb{{background:{LINE};border-radius:3px}}
::selection{{background:{ACC};color:#FFFFFF}}
a:focus-visible,button:focus-visible,input:focus-visible,
[role="tab"]:focus-visible,summary:focus-visible{{outline:3px solid {ACC};outline-offset:2px}}
.saltar{{position:absolute;left:-9999px}}
.saltar:focus{{position:static;display:inline-block;background:{ACC};color:#fff;
 padding:8px 14px;margin:8px 0}}

.cab{{background:{BANDA};color:#FFFFFF;margin:0 -2.6rem 0;padding:30px 2.6rem 26px;
 display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap}}
.cab-badge{{font-family:'IBM Plex Mono',monospace;font-size:{.66*FS}rem;
 letter-spacing:.14em;text-transform:uppercase;color:{CREMA};opacity:.85}}
.marca{{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}}
.tag{{font-size:{.9*FS}rem;color:{CREMA};font-style:italic}}
.navlinea{{border-bottom:2px solid {LINE};margin:0 -2.6rem 30px}}
.sep{{border-top:1px solid {LINE};margin:46px 0 34px}}
.pulso{{background:{SOFT};border:1px solid {LINE};border-radius:4px;padding:18px 22px;margin-top:8px}}
.pulso-t{{font-family:'IBM Plex Mono',monospace;font-size:{.64*FS}rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTE};margin-bottom:14px}}
.pulso-fila{{display:flex;gap:22px;flex-wrap:wrap}}
.pulso-v{{font-family:'IBM Plex Mono',monospace;font-size:{1.18*FS}rem;font-weight:500;color:{INK}}}
.pulso-k{{font-size:{.82*FS}rem;color:{MUTE};margin-top:3px;line-height:1.4}}
.portada h1{{font-size:{2.5*FS}rem;font-weight:600;line-height:1.15;
 letter-spacing:-.02em;margin:14px 0 16px;color:{INK}}}
.portada h1 span{{color:{ACC}}}
.portada p{{font-size:{1.06*FS}rem;line-height:1.6;color:{MUTE};max-width:62ch}}
.wm{{font-size:{1.5*FS}rem;font-weight:600;letter-spacing:.34em;text-transform:uppercase;
 color:#FFFFFF}}
.cab .sub{{font-family:'IBM Plex Mono',monospace;font-size:{.68*FS}rem;letter-spacing:.16em;
 text-transform:uppercase;color:{CREMA}}}

.lbl{{font-family:'IBM Plex Mono',monospace;font-size:{.66*FS}rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE};margin-bottom:10px}}
h2.sec{{font-size:{1.18*FS}rem;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;
 color:{INK}}}
p.sub{{font-size:{.92*FS}rem;color:{MUTE};margin:0 0 18px;line-height:1.55}}

.figura{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;
 font-size:{3.4*FS}rem;font-weight:500;letter-spacing:-.045em;color:{INK};line-height:1;
 white-space:nowrap}}
.figura em{{font-style:normal;font-size:{1.5*FS}rem;color:{ACC};margin-left:6px;font-weight:400}}
.banda{{font-family:'IBM Plex Mono',monospace;font-size:{.86*FS}rem;color:{INK};
 margin-top:12px}}
.banda span{{color:{MUTE}}}
.cifras{{display:flex;gap:28px;flex-wrap:wrap;margin-top:22px}}
.cifras .v{{font-family:'IBM Plex Mono',monospace;font-size:{1.08*FS}rem;color:{INK}}}
.cifras .k{{font-family:'IBM Plex Mono',monospace;font-size:{.62*FS}rem;letter-spacing:.12em;
 text-transform:uppercase;color:{MUTE};margin-top:4px}}

.mapa{{position:relative;border:1px solid {LINE};border-radius:4px;overflow:hidden;
 background:{SURF};background-size:100% 100%;background-repeat:no-repeat;
 aspect-ratio:{MAPA_W}/{MAPA_H}}}
.mapa .ov{{position:absolute;inset:0;width:100%;height:100%}}
.iso{{fill:none;stroke:#FFFFFF;stroke-width:.8;opacity:.55}}
.iso.hi{{stroke:{INK};opacity:.85;stroke-width:1.4}}
.anot{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.06em;
 fill:{INK};paint-order:stroke;stroke:#FFFFFF;stroke-width:3px;stroke-linejoin:round}}
.dis{{font-family:'Instrument Sans',sans-serif;font-size:12.5px;font-weight:500;
 fill:{INK};paint-order:stroke;stroke:#FFFFFF;stroke-width:3.2px;stroke-linejoin:round}}

.esc{{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap;
 font-family:'IBM Plex Mono',monospace;font-size:{.66*FS}rem;color:{MUTE}}}
.esc .bar{{flex:0 0 170px;height:10px;border-radius:2px;border:1px solid {LINE}}}
.esc-bar{{flex:0 0 170px;height:10px;border-radius:2px;border:1px solid {LINE};
 background:linear-gradient(90deg,#EEE9E2,#8CB8A0 50%,#0B4D3D)}}

.nota{{font-size:{.9*FS}rem;line-height:1.62;color:{MUTE}}}
.nota b{{color:{INK};font-weight:600}}
.chip{{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:{.66*FS}rem;letter-spacing:.10em;color:{ACC};border:1px solid {ACC_DIM};border-radius:20px;padding:4px 11px}}
.aviso{{border-left:3px solid {ACC};background:{SOFT};padding:12px 15px;
 font-size:{.87*FS}rem;color:{INK};margin-top:16px;border-radius:0 4px 4px 0}}

.tarj{{border:1px solid {LINE};border-radius:4px;background:{SURF};padding:24px 26px}}
.tarj .t{{font-size:{1.6*FS}rem;font-weight:600;display:flex;align-items:center;gap:10px}}
.tarj .d{{font-size:{.92*FS}rem;color:{MUTE};margin-top:10px;line-height:1.6}}
.sim{{font-size:{1.1*FS}rem;line-height:1}}

.kpi{{display:flex;border-top:2px solid {INK};border-bottom:1px solid {LINE};flex-wrap:wrap}}
.kpi>div{{flex:1 1 160px;padding:16px 14px 14px;border-right:1px solid {LINE}}}
.kpi>div:last-child{{border-right:none}}
.kpi .v{{font-family:'IBM Plex Mono',monospace;font-size:{1.32*FS}rem;color:{INK}}}
.kpi .k{{font-size:{.78*FS}rem;color:{MUTE};margin-top:6px;line-height:1.4}}

table.tb{{width:100%;border-collapse:collapse;font-size:{.92*FS}rem}}
table.tb{{color:{INK}}}
table.tb th{{font-size:{.76*FS}rem;color:{MUTE};font-weight:500;text-align:right;
 padding-bottom:10px;border-bottom:1px solid {LINE}}}
table.tb th:first-child{{text-align:left}}
table.tb td{{padding:12px 0;border-bottom:1px solid {LINE};text-align:right;
 color:{INK}}}
table.tb td .m,table.tb td.m{{color:{INK}}}
table.tb td:first-child{{text-align:left}}
table.tb caption{{caption-side:top;text-align:left;font-size:{.9*FS}rem;color:{MUTE};
 padding-bottom:12px}}

.pie{{border-top:1px solid {LINE};margin-top:46px;padding-top:18px;display:flex;
 justify-content:space-between;gap:16px;flex-wrap:wrap;font-size:{.78*FS}rem;color:{MUTE}}}

[data-testid="stWidgetLabel"] p{{font-family:'Instrument Sans',sans-serif!important;
 font-size:{.84*FS}rem!important;color:{INK}!important;font-weight:500!important}}
[data-baseweb="select"]>div{{background:{SURF}!important;border-color:{LINE}!important;
 border-radius:4px!important;color:{INK}!important}}
[data-baseweb="select"] *{{color:{INK}!important}}
[data-baseweb="select"] svg{{fill:{INK}!important}}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{{
 background:{SURF}!important;border-color:{LINE}!important;border-radius:4px!important;
 color:{INK}!important;-webkit-text-fill-color:{INK}!important}}
[data-testid="stNumberInput"] button{{background:{SURF}!important;color:{INK}!important}}
[data-testid="stNumberInput"] button svg{{fill:{INK}!important}}
[data-baseweb="tag"]{{background:{ACC}!important;color:#FFFFFF!important}}
[data-baseweb="tag"] *{{color:#FFFFFF!important}}
[data-testid="stExpander"]{{border:1px solid {LINE};background:{SURF};border-radius:4px}}
[data-testid="stExpander"] summary p{{font-size:{.86*FS}rem!important;color:{INK}!important;
 font-weight:500}}

div[data-testid="stHorizontalBlock"] div.stButton>button{{background:transparent;
 border:none;color:{MUTE};font-size:{.94*FS}rem;font-weight:500;padding:14px 0;
 border-radius:0;border-bottom:3px solid transparent;transition:none}}
div[data-testid="stHorizontalBlock"] div.stButton>button:hover{{color:{ACC};
 border-bottom-color:{ACC_DIM};background:transparent}}
div.stButton>button[kind="secondary"]{{background:transparent}}
.stTabs [data-baseweb="tab-list"]{{gap:30px;border-bottom:2px solid {LINE};background:transparent}}
.stTabs [data-baseweb="tab"]{{background:transparent;padding:14px 0 12px;height:auto;
 font-size:{.94*FS}rem;font-weight:500;color:{MUTE}}}
.stTabs [aria-selected="true"]{{color:{ACC}!important}}
.stTabs [data-baseweb="tab-highlight"]{{background:{ACC};height:3px}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:30px}}

@media print{{
 .stApp{{background:#FFFFFF!important}}
 .block-container{{padding:0!important;max-width:100%!important}}
 .cab{{background:#FFFFFF!important;color:#000!important;margin:0 0 12px;
  padding:0 0 10px;border-bottom:2px solid #000}}
 .wm,.cab .sub,.tag{{color:#000!important}}
 div.stButton,[data-testid="stFileUploader"],[data-testid="stExpander"],
 .navlinea,iframe,[data-testid="stDownloadButton"]{{display:none!important}}
 .figura{{font-size:2.4rem!important}}
 h2.sec{{page-break-after:avoid}} .sep{{margin:16px 0}}
 table.tb td,table.tb th,.nota,.kpi .k{{color:#000!important}}
}}
@media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
@media(min-width:1500px){{.figura{{font-size:{4.1*FS}rem}}}}
@media(max-width:1100px){{.figura{{font-size:{2.9*FS}rem}}}}
/* Tablet: las columnas de Streamlit se parten en dos por fila */
@media(max-width:1000px){{
 div[data-testid="stHorizontalBlock"]{{flex-wrap:wrap!important;gap:.6rem}}
 div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]{{
  min-width:47%!important;flex:1 1 47%!important}}
 table.tb{{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch}}
}}
/* Móvil: una sola columna y tablas deslizables */
@media(max-width:700px){{
 div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]{{
  min-width:100%!important;flex:1 1 100%!important}}
 .cifras{{gap:14px}} .cifras>div{{flex:1 1 44%}}
 .kpi>div{{flex:1 1 100%;border-right:none;border-bottom:1px solid {LINE}}}
 .figura{{font-size:{2.1*FS}rem}}
 .portada h1{{font-size:{1.7*FS}rem}}
 .cab{{flex-direction:column;align-items:flex-start;gap:6px}}
 .wm{{font-size:{1.05*FS}rem;letter-spacing:.28em}}
 .pie{{flex-direction:column;gap:6px}}
 table.tb td,table.tb th{{padding:9px 8px 9px 0;white-space:nowrap}}
}}
@media(max-width:820px){{
 .block-container,[data-testid="stMainBlockContainer"]{{
  padding:0 1.1rem 3rem!important;max-width:100%!important}}
 .cab{{margin:0 -1.1rem 24px;padding:20px 1.1rem}}
 .navlinea{{margin:0 -1.1rem 24px}}
 .figura{{font-size:{2.3*FS}rem;white-space:normal}} .cifras{{gap:18px}}
 .stTabs [data-baseweb="tab-list"]{{gap:16px;overflow-x:auto;scrollbar-width:none}}
}}
</style>
<a class="saltar" href="#valoracion">Saltar a la valoración</a>
""", unsafe_allow_html=True)

st.markdown('', unsafe_allow_html=True)

# ─────────────────────────── datos del inmueble ───────────────────────
PAGINAS = [("valorar", "Valorar"), ("comparar", "Comparar barrios"),
           ("lote", "Varias a la vez"),
           ("similares", "Viviendas parecidas"),
           ("fiabilidad", "Fiabilidad"), ("metodo", "Cómo funciona")]
if "pagina" not in st.session_state:
    st.session_state.pagina = "inicio"

st.markdown(f'<div class="cab"><div class="marca">'
            f'<span class="wm">Aldaba</span>'
            f'<span class="tag">Llama a cualquier puerta</span></div>'
            f'<div class="cab-badge">Precios · '
            f'{IDX["fecha_corta"] if IDX else "ago-2026"}</div></div>',
           
            unsafe_allow_html=True)

nav = st.columns([1.2] + [1] * len(PAGINAS) + [3.2])
if nav[0].button("Aldaba · Inicio", key="nav_home", use_container_width=True):
    st.session_state.pagina = "inicio"
for _i, (_clave, _etq) in enumerate(PAGINAS):
    if nav[_i + 1].button(_etq, key=f"nav_{_clave}", use_container_width=True):
        st.session_state.pagina = _clave
PAG = st.session_state.pagina
st.markdown('<div class="navlinea"></div>', unsafe_allow_html=True)

if PAG == "inicio":
    R0 = resumen()
    _tot = sum(v.get("n_anuncios",0) for v in R0.values())
    st.markdown(
        f'<div class="portada">'
        f'<h1>Llama a cualquier puerta<br>'
        f'<span>y te decimos lo que hay detrás.</span></h1>'
        f'<p>Aldaba estima lo que vale una vivienda en Madrid, Barcelona y València, '
        f'dice con cuánto margen lo hace y explica de dónde sale cada euro. '
        f'Aprendido de {num(_tot)} viviendas reales.</p></div>'
        f'<div class="kpi" style="margin-top:34px">'
        f'<div><div class="v">3</div><div class="k">ciudades cubiertas</div></div>'
        f'<div><div class="v">{num(_tot)}</div>'
        f'<div class="k">viviendas analizadas</div></div>'
        f'<div><div class="v">{max((v.get("r2_bloques",0) for v in R0.values()), default=0)*100:.0f} %</div>'
        f'<div class="k">de acierto en el mejor mercado</div></div>'
        f'<div><div class="v">{IDX["fecha_corta"] if IDX else "ago-2026"}</div>'
        f'<div class="k">precios actualizados a</div></div></div>',
        unsafe_allow_html=True)
    _p = st.columns(3, gap="large")
    for _col, (_t, _d) in zip(_p, [
        ("Cuánto vale", "Metes los datos del piso y sale una cifra con su margen, más un "
         "mapa de la ciudad con esa misma vivienda en cada barrio."),
        ("¿Piden un precio justo?", "Escribes lo que pide el vendedor y te decimos si está "
         "barata, normal o cara, y por cuánto."),
        ("¿Y si fuera en otro barrio?", "La misma vivienda valorada en todas las zonas, "
         "ordenadas de más cara a más barata.")]):
        _col.markdown(f'<div class="tarj" style="height:100%"><div class="t" '
                      f'style="font-size:1.15rem;color:{INK}">{_t}</div>'
                      f'<div class="d" style="color:{MUTE}">{_d}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="pulso">'
        f'<div class="pulso-t">El mercado en cifras · {IDX["fecha"] if IDX else "agosto de 2026"}</div>'
        f'<div class="pulso-fila">'
        f'<div><div class="pulso-v">+86 %</div><div class="pulso-k">ha subido Madrid desde 2018</div></div>'
        f'<div><div class="pulso-v">3.480 €/m²</div><div class="pulso-k">Vallecas hoy = mediana Madrid 2018</div></div>'
        f'<div><div class="pulso-v">×2,02</div><div class="pulso-k">Salamanca se ha doblado</div></div>'
        f'<div><div class="pulso-v">21 meses</div><div class="pulso-k">consecutivos de máximos históricos</div></div>'
        f'</div></div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button("Valorar una vivienda", key="cta", type="primary"):
        st.session_state.pagina = "valorar"
        st.rerun()
    st.markdown(
        f'<div class="pie"><div>© 2026 Aldaba</div>'
        f'<div>Datos idealista18 (2018) · Precios de  (<a href="https://github.com/paezha/idealista18" target="_blank" style="color:{MUTE};text-decoration:underline">enlace</a>) · '
        f'{IDX["fecha"] if IDX else "agosto de 2026"}</div>'
        f'<div>Estimación orientativa · No sustituye a una tasación oficial</div></div>',
        unsafe_allow_html=True)
    st.stop()

st.markdown('<h2 class="sec">Datos de la vivienda</h2>'
            '<p class="sub">Rellena lo que sepas del inmueble. La valoración se '
            'actualiza al instante.</p>', unsafe_allow_html=True)

c0, c1, c2, c3, c4 = st.columns([1.1, 1.5, 1, 1, 1])
_ci = _qs("c", "Madrid")
ciudad = c0.selectbox("Ciudad", CIUDADES,
                      index=CIUDADES.index(_ci) if _ci in CIUDADES else 0)
meta, mods, cen = cargar(ciudad)
meta = aplicar_indice(meta, ciudad)
dis_ok = [d for d, v in meta["distritos"].items() if v["n"] >= 120]
nivel = {d: meta["distritos"][d]["e2026"] or meta["nivel_ciudad_2026"] for d in dis_ok}
dis_ok = sorted(dis_ok, key=lambda d: -nivel[d])
_db = _qs("b", None)
distrito = c1.selectbox("Barrio o distrito", dis_ok,
                        index=dis_ok.index(_db) if _db in dis_ok else 0)
area = c2.number_input("Metros cuadrados", 25, 600, int(_qs("m", 90)), 5)
rooms = c3.number_input("Habitaciones", 0, 15, int(_qs("h", 3)))
baths = c4.number_input("Baños", 0, 10, int(_qs("ba", 2)))

c5, c6, c7 = st.columns([1, 2, 1.6])
year = c5.number_input("Año de construcción", 1900, 2018, int(_qs("a", 1970)))
DOT = ["Ascensor", "Terraza", "Plaza de garaje", "Climatización", "Piscina", "Portería"]
dots = c6.multiselect("Qué tiene la vivienda", DOT, default=["Ascensor"])
d_metro = c7.slider("Minutos andando al metro (km)", 0.0, 5.0, 0.3, 0.05)

lift, terrace, parking = int(DOT[0] in dots), int(DOT[1] in dots), int(DOT[2] in dots)
air, pool, doorman = int(DOT[3] in dots), int(DOT[4] in dots), int(DOT[5] in dots)
ext = (lift, terrace, parking, air, pool, doorman)

FEAT = meta["features"]
FAC = meta["distritos"][distrito]["factor"]
ESTIMADO = meta["distritos"][distrito]["estimado"]
p99 = int(meta["dominio"]["area_p99"])
sub_c = cen[cen.distrito == distrito]
lat_d = float(sub_c.lat.mean()) if len(sub_c) else meta["centro"]["lat"]
lon_d = float(sub_c.lon.mean()) if len(sub_c) else meta["centro"]["lon"]
clat, clon = meta["centro"]["lat"], meta["centro"]["lon"]

with st.expander("Ubicación exacta · afina la valoración con las coordenadas del portal"):
    st.markdown('<div class="nota">Sin coordenadas usamos el centro del barrio, y dentro '
                'de un mismo barrio el precio varía bastante. Si conoces la dirección, '
                'sácalas de Google Maps: clic derecho sobre el portal y copia los dos '
                'números.</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns([1, 1, 1.4])
    exacta = q3.checkbox("Usar estas coordenadas", key="usar_coord")
    lat_i = q1.number_input("Latitud", value=round(lat_d, 5), format="%.5f", step=0.0005,
                            key=f"lat_{ciudad}_{distrito}")
    lon_i = q2.number_input("Longitud", value=round(lon_d, 5), format="%.5f", step=0.0005,
                            key=f"lon_{ciudad}_{distrito}")
lat0, lon0 = (lat_i, lon_i) if exacta else (lat_d, lon_d)
BARRIO = None
if exacta and len(cen):
    _d = (cen.lat - lat0) ** 2 + (cen.lon - lon0) ** 2
    BARRIO = str(cen.iloc[int(_d.idxmin())]["barrio"])


def fila(a, la, lo, dm=None):
    r = dict(meta["medianas"])
    r.update({"CONSTRUCTEDAREA": a, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
              "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
              "LATITUDE": la, "LONGITUDE": lo,
              "DISTANCE_TO_CITY_CENTER": float(np.sqrt(
                  ((la - clat) * KM) ** 2 +
                  ((lo - clon) * KM * math.cos(math.radians(clat))) ** 2)),
              "DISTANCE_TO_METRO": d_metro if dm is None else dm,
              "HASLIFT": lift, "HASTERRACE": terrace, "HASPARKINGSPACE": parking,
              "HASAIRCONDITIONING": air, "HASSWIMMINGPOOL": pool, "HASDOORMAN": doorman,
              "PERIOD": 201812})
    return {c: r.get(c, 0.0) for c in FEAT}


def bruto(filas, q="q50"):
    return np.exp(mods[q].predict(xgb.DMatrix(pd.DataFrame(filas)[FEAT].astype(float))))


def predecir(filas, q="q50"):
    return bruto(filas, q) * FAC


f0 = [fila(area, lat0, lon0)]
u50 = float(predecir(f0)[0])
u10 = min(float(predecir(f0, "q10")[0]), u50)
u90 = max(float(predecir(f0, "q90")[0]), u50)
p50, p10, p90 = u50 * area, u10 * area, u90 * area

st.markdown('<div id="valoracion"></div>', unsafe_allow_html=True)


class _Seccion:
    def __init__(self, activa): self.activa = activa
    def __enter__(self): return self
    def __exit__(self, *a): return False


tabV = _Seccion(PAG == "valorar")
tabO = _Seccion(PAG == "valorar")
tabZ = _Seccion(PAG == "comparar")
tabM = _Seccion(PAG == "fiabilidad")
tabD = _Seccion(PAG == "metodo")
tabL = _Seccion(PAG == "lote")
tabS = _Seccion(PAG == "similares")


# ───────────────────────────── mapa ───────────────────────────────────
@st.cache_data(show_spinner=False)
def campo(ciudad, area, rooms, baths, year, d_metro, ext):
    mt, mo, ce = cargar(ciudad)
    dlo = DLA * (MAPA_W / MAPA_H) / math.cos(math.radians(mt["centro"]["lat"]))
    la0, lo0 = mt["centro"]["lat"], mt["centro"]["lon"]
    LO, LA = np.meshgrid(np.linspace(lo0 - dlo, lo0 + dlo, NX),
                         np.linspace(la0 + DLA, la0 - DLA, NY))
    b = dict(mt["medianas"])
    b.update({"CONSTRUCTEDAREA": area, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
              "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
              "DISTANCE_TO_METRO": d_metro, "HASLIFT": ext[0], "HASTERRACE": ext[1],
              "HASPARKINGSPACE": ext[2], "HASAIRCONDITIONING": ext[3],
              "HASSWIMMINGPOOL": ext[4], "HASDOORMAN": ext[5], "PERIOD": 201812})
    df = pd.DataFrame([{c: b.get(c, 0.0) for c in mt["features"]}] * LA.size)
    df["LATITUDE"] = LA.ravel(); df["LONGITUDE"] = LO.ravel()
    df["DISTANCE_TO_CITY_CENTER"] = np.sqrt(
        ((LA - la0) * KM) ** 2 +
        ((LO - lo0) * KM * math.cos(math.radians(la0))) ** 2).ravel()
    u = np.exp(mo["q50"].predict(xgb.DMatrix(df[mt["features"]].astype(float))))
    fb = np.array([mt["distritos"].get(d, {}).get("factor", mt["factor_ciudad"])
                   for d in ce.distrito.values])
    dd = ((LA.ravel()[:, None] - ce.lat.values[None, :]) ** 2 +
          (LO.ravel()[:, None] - ce.lon.values[None, :]) ** 2)
    return (u * fb[dd.argmin(1)] * area).reshape(NY, NX).astype(np.float64), dlo


@st.cache_data(show_spinner=False)
def png(zb, shape, lo, hi, stops, alfa=True):
    """Superficie de precio en RGBA: transparente en lo barato, opaca en lo caro,
    para que el callejero se lea por debajo."""
    Z = np.frombuffer(zb, dtype=np.float64).reshape(shape)
    t = np.clip((Z - lo) / max(hi - lo, 1e-9), 0, 1) ** .9
    ps = np.array([s[0] for s in stops])
    rgba = np.zeros(t.shape + (4,), np.uint8)
    for k in range(3):
        rgba[..., k] = np.clip(np.interp(t, ps, [s[1][k] for s in stops]), 0, 255)
    rgba[..., 3] = (np.clip(0.30 + 0.48 * t, 0, 1) * 255).astype(np.uint8) if alfa else 255
    im = Image.fromarray(rgba, "RGBA").resize((shape[1] * 8, shape[0] * 8), Image.BICUBIC)
    b = BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def isolineas(Z, niveles, W, H):
    ny, nx = Z.shape
    sx, sy = W / (nx - 1), H / (ny - 1)
    out = []
    for lv in niveles:
        G = Z > lv; seg = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b_, c_, e = Z[j, i], Z[j, i+1], Z[j+1, i+1], Z[j+1, i]
                idx = G[j, i] | (G[j, i+1] << 1) | (G[j+1, i+1] << 2) | (G[j+1, i] << 3)
                if idx in (0, 15):
                    continue
                T = ((i + (lv-a)/(b_-a))*sx, j*sy) if b_ != a else None
                R = ((i+1)*sx, (j + (lv-b_)/(c_-b_))*sy) if c_ != b_ else None
                B = ((i + (lv-e)/(c_-e))*sx, (j+1)*sy) if c_ != e else None
                L = (i*sx, (j + (lv-a)/(e-a))*sy) if e != a else None
                for p, q in {1:[(L,T)],2:[(T,R)],3:[(L,R)],4:[(R,B)],5:[(L,T),(R,B)],
                             6:[(T,B)],7:[(L,B)],8:[(L,B)],9:[(T,B)],10:[(L,T),(R,B)],
                             11:[(T,R)],12:[(L,R)],13:[(R,B)],14:[(L,T)]}[idx]:
                    if p and q:
                        seg.append(f"M{p[0]:.1f} {p[1]:.1f}L{q[0]:.1f} {q[1]:.1f}")
        out.append("".join(seg))
    return out


@st.cache_data(show_spinner=False)
def por_distrito(ciudad, area, rooms, baths, year, d_metro, ext):
    """El mismo inmueble valorado en cada barrio."""
    mt, mo, ce = cargar(ciudad)
    filas, nombres = [], []
    for d, v in mt["distritos"].items():
        if v["n"] < 120:
            continue
        s = ce[ce.distrito == d]
        if not len(s):
            continue
        la, lo = float(s.lat.mean()), float(s.lon.mean())
        r = dict(mt["medianas"])
        r.update({"CONSTRUCTEDAREA": area, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
                  "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
                  "LATITUDE": la, "LONGITUDE": lo, "DISTANCE_TO_METRO": d_metro,
                  "DISTANCE_TO_CITY_CENTER": float(np.sqrt(
                      ((la - mt["centro"]["lat"]) * KM) ** 2 +
                      ((lo - mt["centro"]["lon"]) * KM *
                       math.cos(math.radians(mt["centro"]["lat"]))) ** 2)),
                  "HASLIFT": ext[0], "HASTERRACE": ext[1], "HASPARKINGSPACE": ext[2],
                  "HASAIRCONDITIONING": ext[3], "HASSWIMMINGPOOL": ext[4],
                  "HASDOORMAN": ext[5], "PERIOD": 201812})
        filas.append({c: r.get(c, 0.0) for c in mt["features"]})
        nombres.append((d, v["factor"], v["estimado"]))
    u = np.exp(mo["q50"].predict(xgb.DMatrix(pd.DataFrame(filas)[mt["features"]].astype(float))))
    out = [(n, float(uu) * f * area, est) for (n, f, est), uu in zip(nombres, u)]
    return sorted(out, key=lambda x: -x[1])


# ───────────────────────── pestaña · valoración ───────────────────────
if tabV.activa:
    st.markdown('<h2 class="sec">Lo que vale esta vivienda hoy</h2>'
                '<p class="sub">Estimación a precios de agosto de 2026. La franja indica '
                'el margen razonable: de cada diez viviendas parecidas, ocho se venden '
                'dentro de él.</p>', unsafe_allow_html=True)

    Z, dlo = campo(ciudad, area, rooms, baths, year, d_metro, ext)
    lo_z, hi_z = float(np.percentile(Z, 2)), float(np.percentile(Z, 98))
    b64 = png(np.ascontiguousarray(Z).tobytes(), Z.shape, lo_z, hi_z, STOPS)

    rank_mapa = por_distrito(ciudad, area, rooms, baths, year, d_metro, ext)
    # rótulos de los barrios principales sobre el mapa
    a1, a2 = st.columns([1, 1.3], gap="large")
    with a1:
        st.markdown(
            f'<div class="lbl">{distrito} · {ciudad}</div>'
            f'<div class="figura">{num(p50)}<em>€</em></div>'
            f'<div class="banda">{eur(p10)} a {eur(p90)} '
            f'<span>· margen razonable</span></div>'
            f'<div class="cifras">'
            f'<div><div class="v">{eur(u50)}</div><div class="k">precio por m²</div></div>'
            f'<div><div class="v">{num(area)} m²</div><div class="k">superficie</div></div>'
            f'<div><div class="v">{2018-year}</div><div class="k">años de antigüedad</div></div>'
            f'</div>'
            f'<div class="chips" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:18px">'
            f'<span class="chip">+{(FAC-1)*100:.0f}\u202f% desde 2018</span>'
            f'<span class="chip">{eur(meta["distritos"][distrito].get("e2026") or meta["nivel_ciudad_2026"])}/m\u00b2 en {distrito}</span>'
            f'<span class="chip">{num(meta["n_anuncios"])} viviendas analizadas</span>'
            f'</div>'
            f'<div class="nota" style="margin-top:24px">En {distrito} el metro cuadrado se '
            f'ofrece hoy a <b>{eur(meta["distritos"][distrito]["e2026"] or meta["nivel_ciudad_2026"])}'
            f'</b> de media. Esta vivienda sale por encima o por debajo según sus '
            f'características concretas.</div>'
            + (f'<div class="aviso"><b>Aviso.</b> Por encima de {p99} m² tenemos pocas '
               f'viviendas parecidas con las que comparar, así que la cifra es orientativa.'
               f'</div>' if area > p99 else '')
            + ('<div class="aviso"><b>Aviso.</b> De este barrio no se publica precio propio, '
               'así que usamos la media de la ciudad.</div>' if ESTIMADO else ''),
            unsafe_allow_html=True)
    with a2:
        # ── Mapa: Leaflet incrustado. La superficie va como imagen que el navegador
        #    interpola (suave, sin celdas) y el precio se lee de la malla en JS.
        _mla = (cen.lat.max() - cen.lat.min()) * 0.06
        _mlo = (cen.lon.max() - cen.lon.min()) * 0.06
        _laA, _laB = cen.lat.min() - _mla, cen.lat.max() + _mla
        _loA, _loB = cen.lon.min() - _mlo, cen.lon.max() + _mlo

        _NY2, _NX2 = 90, 86
        _LA2, _LO2 = np.meshgrid(np.linspace(_laB, _laA, _NY2),
                                 np.linspace(_loA, _loB, _NX2), indexing="ij")
        _b2 = dict(meta["medianas"])
        _b2.update({"CONSTRUCTEDAREA": area, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
                    "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
                    "DISTANCE_TO_METRO": d_metro, "HASLIFT": lift,
                    "HASTERRACE": terrace, "HASPARKINGSPACE": parking,
                    "HASAIRCONDITIONING": air, "HASSWIMMINGPOOL": pool,
                    "HASDOORMAN": doorman, "PERIOD": 201812})
        _df2 = pd.DataFrame([{c: _b2.get(c, 0.0) for c in FEAT}] * _LA2.size)
        _df2["LATITUDE"] = _LA2.ravel()
        _df2["LONGITUDE"] = _LO2.ravel()
        _df2["DISTANCE_TO_CITY_CENTER"] = np.sqrt(
            ((_LA2 - clat) * KM) ** 2 +
            ((_LO2 - clon) * KM * math.cos(math.radians(clat))) ** 2).ravel()
        _fb2 = np.array([meta["distritos"].get(d, {}).get("factor", meta["factor_ciudad"])
                         for d in cen.distrito.values])
        _dd2 = ((_LA2.ravel()[:, None] - cen.lat.values[None, :]) ** 2 +
                (_LO2.ravel()[:, None] - cen.lon.values[None, :]) ** 2)
        _Z2 = (np.exp(mods["q50"].predict(xgb.DMatrix(_df2[FEAT].astype(float))))
               * _fb2[_dd2.argmin(1)] * area).reshape(_NY2, _NX2)
        _lo2 = float(np.percentile(_Z2, 2))
        _hi2 = float(np.percentile(_Z2, 98))

        # Imagen RGBA de la superficie, ampliada con interpolación bicúbica
        _t = np.clip((_Z2 - _lo2) / max(_hi2 - _lo2, 1e-9), 0, 1) ** 0.85
        _stops = [(0.00, (240, 246, 242)), (0.22, (198, 224, 208)),
                  (0.45, (140, 192, 164)), (0.68, (82, 152, 122)),
                  (0.86, (36, 112, 88)), (1.00, (8, 70, 55))]
        _ps = np.array([p[0] for p in _stops])
        _rgba = np.zeros(_t.shape + (4,), np.uint8)
        for _k in range(3):
            _rgba[..., _k] = np.clip(
                np.interp(_t, _ps, [p[1][_k] for p in _stops]), 0, 255)
        # Distancia de cada punto de la malla al barrio más próximo, en km
        _dmin = np.sqrt(_dd2.min(axis=1)).reshape(_NY2, _NX2) * KM
        # Opacidad plena hasta 1,3 km de un barrio; se apaga del todo a 3,2 km
        _mask = np.clip((3.2 - _dmin) / 1.9, 0.0, 1.0)
        _mask = _mask * _mask * (3 - 2 * _mask)          # suavizado
        # y un desvanecido extra en el propio borde de la imagen
        _fy = np.clip(np.minimum(np.arange(_NY2), _NY2 - 1 - np.arange(_NY2)) / 7.0, 0, 1)
        _fx = np.clip(np.minimum(np.arange(_NX2), _NX2 - 1 - np.arange(_NX2)) / 7.0, 0, 1)
        _mask *= _fy[:, None] * _fx[None, :]
        _rgba[..., 3] = (np.clip(0.20 + 0.62 * _t, 0, 1) * _mask * 255).astype(np.uint8)
        _img = Image.fromarray(_rgba, "RGBA").resize(
            (_NX2 * 9, _NY2 * 9), Image.BICUBIC)
        _buf = BytesIO(); _img.save(_buf, "PNG", optimize=True)
        _uri = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()

        _precios = [[int(_Z2[j, i]) for i in range(_NX2)] for j in range(_NY2)]
        _rank_m = por_distrito(ciudad, area, rooms, baths, year, d_metro, ext)
        _marcas = [{"lat": float(cen[cen.distrito == d].lat.mean()),
                    "lon": float(cen[cen.distrito == d].lon.mean()), "t": d}
                   for d, _, _ in _rank_m[:9] if len(cen[cen.distrito == d])]

        _html = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;padding:0;background:__BG__}
 #m{width:100%;height:__H__px;border:1px solid __LINE__;border-radius:4px}
 .leaflet-container{background:__BG__;font-family:'Instrument Sans',sans-serif}
 .leaflet-tile-pane{filter:grayscale(.92) brightness(1.07) contrast(.9)}
 .leaflet-control-attribution{font-size:9px;background:rgba(255,255,255,.7)}
 #tt{position:absolute;z-index:1000;pointer-events:none;display:none;
     background:#FFFFFF;border:1px solid __LINE__;border-radius:4px;
     padding:7px 11px;font-family:'IBM Plex Mono',monospace;font-size:12px;
     color:__INK__;box-shadow:0 2px 10px rgba(0,0,0,.13);white-space:nowrap}
 #tt b{font-size:14px}
 .bq{background:rgba(255,255,255,.82);border:none;border-radius:3px;
     padding:1px 5px;font-size:10.5px;font-weight:500;color:__INK__;
     box-shadow:none;white-space:nowrap}
 .bq:before{display:none}
</style>
<div style="position:relative"><div id="m"></div><div id="tt"></div></div>
<script>
const LA_A=__LAA__, LA_B=__LAB__, LO_A=__LOA__, LO_B=__LOB__;
const NY=__NY__, NX=__NX__, P=__PRECIOS__, MARCAS=__MARCAS__;
const map=L.map('m',{zoomControl:true,attributionControl:true})
  .fitBounds([[LA_A,LO_A],[LA_B,LO_B]]);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(map);
L.imageOverlay('__URI__',[[LA_A,LO_A],[LA_B,LO_B]],{opacity:1,interactive:false})
  .addTo(map);
MARCAS.forEach(function(d){
  L.marker([d.lat,d.lon],{opacity:0,interactive:false})
   .bindTooltip(d.t,{permanent:true,direction:'center',className:'bq'}).addTo(map);
});
L.circleMarker([__PLAT__,__PLON__],{radius:8,color:'#FFFFFF',weight:3,
  fillColor:'__INK__',fillOpacity:1}).addTo(map);
const tt=document.getElementById('tt');
map.on('mousemove',function(e){
  const la=e.latlng.lat, lo=e.latlng.lng;
  if(la<LA_A||la>LA_B||lo<LO_A||lo>LO_B){tt.style.display='none';return;}
  const j=Math.min(NY-1,Math.max(0,Math.round((LA_B-la)/(LA_B-LA_A)*(NY-1))));
  const i=Math.min(NX-1,Math.max(0,Math.round((lo-LO_A)/(LO_B-LO_A)*(NX-1))));
  const v=P[j][i];
  tt.innerHTML='<b>'+v.toLocaleString('es-ES')+' \u20AC</b><br>este piso aqu\u00ed';
  tt.style.left=(e.containerPoint.x+16)+'px';
  tt.style.top=(e.containerPoint.y+16)+'px';
  tt.style.display='block';
});
map.on('mouseout',function(){tt.style.display='none';});
</script>"""
        for _k, _v in [("__BG__", BG), ("__LINE__", LINE), ("__INK__", INK),
                       ("__H__", "500"), ("__LAA__", f"{_laA:.6f}"),
                       ("__LAB__", f"{_laB:.6f}"), ("__LOA__", f"{_loA:.6f}"),
                       ("__LOB__", f"{_loB:.6f}"), ("__NY__", str(_NY2)),
                       ("__NX__", str(_NX2)),
                       ("__PRECIOS__", json.dumps(_precios)),
                       ("__MARCAS__", json.dumps(_marcas, ensure_ascii=False)),
                       ("__URI__", _uri), ("__PLAT__", f"{lat0:.6f}"),
                       ("__PLON__", f"{lon0:.6f}")]:
            _html = _html.replace(_k, _v)
        components.html(_html, height=515, scrolling=False)

        st.markdown(
            f'<figcaption class="nota" style="margin-top:10px">'
            f'<b>Qué estás viendo:</b> cuánto costaría <b>este mismo piso</b> en cada '
            f'punto de {ciudad}. <b>Pasa el cursor</b> por el mapa para ver el precio. '
            f'Cubre el término municipal, que es donde hay datos.</figcaption>'
            f'<div class="esc"><span>{eur(_lo2)}</span>'
            f'<span class="bar" style="flex:0 0 150px;height:10px;border-radius:2px;'
            f'border:1px solid #CFC8B9;background:linear-gradient(90deg,'
            f'#F0F6F2,#C6E0D0 22%,#8CC0A4 45%,#52987A 68%,#247058 86%,#084637)">'
            f'</span><span>{eur(_hi2)}</span><span style="flex:1"></span>'
            f'<span>barato → caro</span></div>',
            unsafe_allow_html=True)

    st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)

    def curva(xs, ys, xn, yn, unidad, etiqueta):
        Wc, Hc, L, R, T, B = 520, 190, 12, 508, 26, 46
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        pad = (ymax - ymin) * .18 or 1
        ymin, ymax = ymin - pad, ymax + pad
        fx = lambda v: L + (v - xmin) / max(xmax - xmin, 1e-9) * (R - L)
        fy = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (Hc - B - T)
        pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
        ex = lambda v: num(v, 1).rstrip("0").rstrip(",") if unidad == " km" else num(v)
        g = "".join(f'<line x1="{L}" y1="{T+(Hc-B-T)*i/3:.1f}" x2="{R}" '
                    f'y2="{T+(Hc-B-T)*i/3:.1f}" stroke="{LINE}" opacity=".55"/>'
                    for i in range(1, 3))
        return (f'<svg viewBox="0 0 {Wc} {Hc}" width="100%" role="img" '
                f'aria-label="{etiqueta}" style="display:block;overflow:visible">{g}'
                f'<polygon points="{L},{Hc-B} {pts} {R},{Hc-B}" fill="{ACC}" opacity=".10"/>'
                f'<polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="2.4" '
                f'stroke-linejoin="round"/>'
                f'<line x1="{fx(xn):.1f}" y1="{T-6}" x2="{fx(xn):.1f}" y2="{Hc-B}" '
                f'stroke="{INK}" stroke-dasharray="3 3" opacity=".5"/>'
                f'<circle cx="{fx(xn):.1f}" cy="{fy(yn):.1f}" r="5.5" fill="{ACC}" '
                f'stroke="#FFFFFF" stroke-width="2.2"/>'
                f'<text x="{fx(xn):.1f}" y="{T-11}" fill="{INK}" font-size="11.5" '
                f'text-anchor="middle" font-family="IBM Plex Mono,monospace">'
                f'esta vivienda</text>'
                f'<line x1="{L}" y1="{Hc-B}" x2="{R}" y2="{Hc-B}" stroke="{LINE}"/>'
                f'<g font-family="IBM Plex Mono,monospace" font-size="11" fill="{MUTE}">'
                f'<text x="{L}" y="{Hc-B+19}">{ex(xmin)}{unidad}</text>'
                f'<text x="{R}" y="{Hc-B+19}" text-anchor="end">{ex(xmax)}{unidad}</text>'
                f'<text x="{R}" y="{T-11}" text-anchor="end">hasta {compact(max(ys))} €</text></g></svg>')

    g1, g2 = st.columns(2, gap="large")
    xa = np.linspace(max(25, area - 55), min(600, area + 55), 22)
    ya = predecir([fila(float(a), lat0, lon0) for a in xa]) * xa
    with g1:
        pend = (float(ya[-1]) - float(ya[0])) / (xa[-1] - xa[0])
        st.markdown('<h2 class="sec">Si fuera más grande o más pequeña</h2>'
                    f'<p class="sub">Manteniendo todo lo demás igual, cada metro cuadrado '
                    f'de más suma unos <b>{eur(pend)}</b>.</p>', unsafe_allow_html=True)
        st.markdown(curva(list(xa), list(ya), area, p50, " m²",
                          f"Precio según superficie, de {num(xa[0])} a {num(xa[-1])} metros "
                          f"cuadrados. Cada metro añade unos {eur(pend)}."),
                    unsafe_allow_html=True)
    xd = np.linspace(0.05, 3.0, 22)
    yd = predecir([fila(area, lat0, lon0, dm=float(v)) for v in xd]) * area
    with g2:
        cai = (float(yd[0]) - float(yd[-1])) / float(yd[0]) * 100
        st.markdown('<h2 class="sec">Si estuviera más lejos del metro</h2>'
                    f'<p class="sub">Alejarse de 50 metros a 3 kilómetros de una boca de '
                    f'metro le resta un <b>{cai:.1f} %</b> del valor.</p>',
                    unsafe_allow_html=True)
        st.markdown(curva(list(xd), list(yd), min(d_metro, 3.0), p50, " km",
                          f"Precio según distancia al metro. Alejarse a 3 kilómetros resta "
                          f"un {cai:.0f} por ciento."), unsafe_allow_html=True)


if tabV.activa:
    st.markdown('<div class="sep"></div>'
                '<h2 class="sec">De dónde sale este precio</h2>'
                '<p class="sub">Cuánto suma o resta cada característica frente a una '
                'vivienda corriente del mismo barrio.</p>', unsafe_allow_html=True)

    # XGBoost calcula la aportación exacta de cada variable (mismo fundamento que SHAP)
    _cb = mods["q50"].predict(
        xgb.DMatrix(pd.DataFrame(f0)[FEAT].astype(float)), pred_contribs=True)[0]
    _NOM = {"CONSTRUCTEDAREA": "Superficie", "LATITUDE": "Ubicación (norte-sur)",
            "LONGITUDE": "Ubicación (este-oeste)",
            "DISTANCE_TO_CITY_CENTER": "Cercanía al centro",
            "DISTANCE_TO_METRO": "Cercanía al metro", "BATHNUMBER": "Número de baños",
            "ROOMNUMBER": "Habitaciones", "HASLIFT": "Ascensor",
            "CADCONSTRUCTIONYEAR": "Año de construcción", "BUILDING_AGE": "Antigüedad",
            "AREA_PER_ROOM": "Amplitud por habitación",
            "CADASTRALQUALITYID": "Calidad de la construcción",
            "FLOORCLEAN": "Planta", "AMENITIES_COUNT": "Equipamiento",
            "HASTERRACE": "Terraza", "HASPARKINGSPACE": "Plaza de garaje",
            "HASSWIMMINGPOOL": "Piscina", "HASDOORMAN": "Portería",
            "HASAIRCONDITIONING": "Climatización",
            "CADMAXBUILDINGFLOOR": "Altura del edificio",
            "CADDWELLINGCOUNT": "Viviendas del edificio",
            "DISTANCE_TO_CASTELLANA": "Cercanía a la Castellana",
            "PERIOD": "Momento del anuncio",
            "FLOOR_RATIO": "Posición en el edificio",
            "SalaryPerYear": "Renta de la zona"}
    _ef = []
    for _k, _f in enumerate(FEAT):
        if _k >= len(_cb) - 1:
            break
        _c = float(_cb[_k])
        if abs(_c) < 0.004:
            continue
        _ef.append((_NOM.get(_f, _f.replace("_", " ").capitalize()),
                    p50 - p50 * math.exp(-_c)))
    _ef.sort(key=lambda x: -abs(x[1]))
    _ef = _ef[:9]

    if _ef:
        _mx = max(abs(v) for _, v in _ef)
        _H = len(_ef) * 30
        _filas = ""
        for _i, (_n, _v) in enumerate(_ef):
            _y = _i * 30
            _w = abs(_v) / _mx * 150
            _x = 250 if _v >= 0 else 250 - _w
            _col = ACC if _v >= 0 else "#A33A22"
            _filas += (f'<text x="0" y="{_y+14}" fill="{INK}" font-size="12.5" '
                       f'font-family="Instrument Sans,sans-serif">{_n}</text>'
                       f'<rect x="{_x:.0f}" y="{_y+3}" width="{_w:.0f}" height="13" '
                       f'fill="{_col}" opacity="0.85" rx="1"/>'
                       f'<text x="{(_x+_w+8) if _v>=0 else (_x-8):.0f}" y="{_y+14}" '
                       f'fill="{_col}" font-size="11.5" '
                       f'text-anchor="{"start" if _v>=0 else "end"}" '
                       f'font-family="IBM Plex Mono,monospace">'
                       f'{"+" if _v>=0 else "−"}{eur(abs(_v))}</text>'
                       f'<line x1="0" y1="{_y+23}" x2="560" y2="{_y+23}" '
                       f'stroke="{LINE}" opacity=".55"/>')
        st.markdown(
            f'<svg viewBox="0 0 560 {_H}" width="100%" role="img" '
            f'aria-label="Desglose: '
            f'{"; ".join(f"{n} {'"'"'suma'"'"' if v>=0 else '"'"'resta'"'"'} {eur(abs(v))}" for n, v in _ef)}." '
            f'style="display:block;max-width:640px">'
            f'<line x1="250" y1="0" x2="250" y2="{_H-8}" stroke="{INK}" '
            f'opacity=".3" stroke-dasharray="2 3"/>{_filas}</svg>'
            f'<div class="nota" style="margin-top:10px;max-width:70ch">Verde suma, rojo '
            f'resta. Los valores salen de descomponer la predicción del propio modelo, '
            f'no de una estimación aparte.</div>', unsafe_allow_html=True)

    # ── Comparación con el método simple ──────────────────────────────
    _niv = meta["distritos"][distrito]["e2026"] or meta["nivel_ciudad_2026"]
    _simple = _niv * area
    _dif = (p50 - _simple) / _simple * 100
    st.markdown(
        f'<div class="sep"></div>'
        f'<h2 class="sec">¿Aporta algo el modelo?</h2>'
        f'<p class="sub">Lo comparamos con la regla de andar por casa: precio medio del '
        f'barrio multiplicado por los metros.</p>'
        f'<div class="kpi">'
        f'<div><div class="v">{eur(_simple)}</div>'
        f'<div class="k">regla simple · {eur(_niv)}/m² × {num(area)} m²</div></div>'
        f'<div><div class="v">{eur(p50)}</div>'
        f'<div class="k">modelo, con las características de esta vivienda</div></div>'
        f'<div><div class="v" style="color:{ACC}">{"+" if _dif>=0 else ""}{_dif:.0f} %</div>'
        f'<div class="k">de diferencia</div></div></div>'
        f'<div class="nota" style="margin-top:12px;max-width:70ch">La regla simple trata '
        f'igual a todos los pisos del barrio. El modelo distingue baños, ascensor, planta, '
        f'antigüedad y distancia al metro, y por eso acierta un '
        f'{meta["r2_bloques"]*100:.0f} % de la variación de precios.</div>',
        unsafe_allow_html=True)

    # ── Enlace compartible ────────────────────────────────────────────
    _p = {"c": ciudad, "b": distrito, "m": str(area), "h": str(rooms),
          "ba": str(baths), "a": str(year)}
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    if st.button("Guardar esta valoración en la dirección web", key="share"):
        st.query_params.update(_p)
        st.rerun()
    st.markdown('<div class="nota" style="margin-top:8px">Al pulsarlo, la dirección del '
                'navegador guarda estos datos: cópiala y quien la abra verá exactamente '
                'esta valoración.</div>', unsafe_allow_html=True)

# ─────────────────────── pestaña · ¿es buen precio? ───────────────────
if tabO.activa:
    st.markdown('<div class="sep"></div>'
                '<h2 class="sec">¿Están pidiendo un precio justo?</h2>'
                '<p class="sub">Escribe el precio del anuncio y lo comparamos con lo que '
                'debería costar una vivienda como esta.</p>', unsafe_allow_html=True)
    o1, o2 = st.columns([1, 1.7], gap="large")
    with o1:
        pedido = st.number_input("Precio que piden (€)", 0, 20_000_000,
                                 int(round(p50 / 1000) * 1000), 5000)
        st.markdown(f'<div class="nota" style="margin-top:12px">Se compara con el margen '
                    f'razonable de esta vivienda: de <b>{eur(p10)}</b> a '
                    f'<b>{eur(p90)}</b>.</div>', unsafe_allow_html=True)
    with o2:
        if pedido <= 0:
            ver, col, sim, det = "Escribe un precio", MUTE, "", ""
        elif pedido < p10:
            ver, col, sim = "Está barata", VERDE, "▼"
            det = (f"Piden {eur(p10-pedido)} menos de lo que marca el margen inferior. "
                   f"Solo una de cada diez viviendas parecidas se ofrece tan barata.")
        elif pedido > p90:
            ver, col, sim = "Está cara", ROJO, "▲"
            det = (f"Piden {eur(pedido-p90)} más de lo que marca el margen superior. "
                   f"Solo una de cada diez viviendas parecidas llega a ese precio.")
        else:
            ver, col, sim = "Precio normal", AMBAR, "●"
            det = (f"Está dentro de lo esperable. La diferencia con nuestra estimación "
                   f"es de {eur(abs(pedido-p50))} "
                   f"{'por encima' if pedido > p50 else 'por debajo'}.")
        st.markdown(f'<div class="tarj"><div class="t" style="color:{col}">'
                    f'<span class="sim" aria-hidden="true">{sim}</span>{ver}</div>'
                    f'<div class="d">{det}</div></div>', unsafe_allow_html=True)

    if pedido > 0:
        amax = max(p90, pedido) * 1.1
        sx_ = lambda v: 10 + v / amax * 980
        st.markdown(
            f'<svg viewBox="0 0 1000 96" width="100%" preserveAspectRatio="none" role="img" '
            f'aria-label="{ver}. Piden {eur(pedido)}. El margen razonable va de {eur(p10)} '
            f'a {eur(p90)}, con estimación central de {eur(p50)}." '
            f'style="display:block;overflow:visible;margin-top:26px">'
            f'<rect x="{sx_(p10):.1f}" y="34" width="{sx_(p90)-sx_(p10):.1f}" height="26" '
            f'fill="{ACC}" opacity=".16"/>'
            f'<line x1="{sx_(p10):.1f}" y1="34" x2="{sx_(p10):.1f}" y2="60" '
            f'stroke="{ACC}" stroke-width="1.8"/>'
            f'<line x1="{sx_(p90):.1f}" y1="34" x2="{sx_(p90):.1f}" y2="60" '
            f'stroke="{ACC}" stroke-width="1.8"/>'
            f'<line x1="10" y1="47" x2="990" y2="47" stroke="{LINE}"/>'
            f'<line x1="{sx_(p50):.1f}" y1="28" x2="{sx_(p50):.1f}" y2="66" '
            f'stroke="{ACC}" stroke-width="2.6"/>'
            f'<text x="{sx_(p50):.1f}" y="20" fill="{INK}" font-size="12" '
            f'text-anchor="middle" font-family="Instrument Sans,sans-serif">'
            f'lo que debería costar</text>'
            f'<circle cx="{sx_(pedido):.1f}" cy="47" r="9" fill="{col}" stroke="#FFFFFF" '
            f'stroke-width="2.5"/>'
            f'<text x="{sx_(pedido):.1f}" y="88" fill="{col}" font-size="12.5" '
            f'font-weight="600" text-anchor="middle" '
            f'font-family="Instrument Sans,sans-serif">piden {eur(pedido)}</text></svg>',
            unsafe_allow_html=True)

# ───────────────────── pestaña · comparar barrios ─────────────────────
if tabZ.activa:
    st.markdown('<h2 class="sec">La misma vivienda, barrio por barrio</h2>'
                f'<p class="sub">Cuánto costaría este piso de {num(area)} m² en cada zona '
                f'de {ciudad}, con las mismas características.</p>', unsafe_allow_html=True)
    rank = por_distrito(ciudad, area, rooms, baths, year, d_metro, ext)
    mx = rank[0][1]
    filas = []
    for d, v, est in rank:
        act = d == distrito
        dif = (v - p50) / p50 * 100
        filas.append(
            f'<tr style="{"font-weight:600" if act else ""}">'
            f'<td>{d}{" · esta vivienda" if act else ""}'
            f'{" *" if est else ""}</td>'
            f'<td class="m">{eur(v)}</td><td class="m">{eur(v/area)}</td>'
            f'<td class="m" style="color:{VERDE if dif>0 else (ROJO if dif<0 else MUTE)}">'
            f'{"+" if dif>0 else ""}{dif:.0f} %</td>'
            f'<td style="width:34%"><svg viewBox="0 0 200 14" width="100%" '
            f'style="display:block" aria-hidden="true">'
            f'<rect x="0" y="2" width="{max(v/mx*200,2):.0f}" height="10" rx="1" '
            f'fill="{ACC}" opacity="{.35+.65*v/mx:.2f}"/></svg></td></tr>')
    st.markdown(
        f'<table class="tb"><caption>Ordenado de más caro a más barato. El porcentaje '
        f'compara con el barrio elegido.</caption><tr><th>Barrio</th>'
        f'<th>Precio total</th><th>Por m²</th><th>Diferencia</th><th></th></tr>'
        f'{"".join(filas)}</table>'
        + f'<div class="nota" style="margin-top:16px">'
        + f'La diferencia entre el barrio más caro y el más barato de {ciudad} '
        + f'es de <b>{eur(rank[0][1]-rank[-1][1])}</b> para esta vivienda concreta. '
        + f'El barrio más caro multiplica por <b>{rank[0][1]/rank[-1][1]:.1f}×</b> '
        + f'al más barato.</div>'
        + ('<div class="nota" style="margin-top:8px">* Sin nivel publicado propio.</div>'
           if any(e for _, _, e in rank) else ''), unsafe_allow_html=True)


if tabL.activa:
    st.markdown('<h2 class="sec">Valorar varias viviendas de golpe</h2>'
                '<p class="sub">Sube un archivo CSV y te devolvemos todas valoradas, '
                'señalando las que se ofrecen por debajo de lo que deberían costar.</p>',
                unsafe_allow_html=True)
    _cols = ["distrito", "m2", "habitaciones", "banos", "anio", "precio_pedido"]
    _ej = pd.DataFrame({"distrito": [dis_ok[0], dis_ok[min(1, len(dis_ok)-1)]],
                        "m2": [90, 65], "habitaciones": [3, 2], "banos": [2, 1],
                        "anio": [1970, 1995], "precio_pedido": [800000, 350000]})
    e1, e2 = st.columns([1, 1.3], gap="large")
    with e1:
        st.markdown('<div class="nota">El archivo necesita estas columnas:</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="m" style="font-size:.82rem;color:{INK};'
                    f'background:{SOFT};padding:10px 13px;border-radius:4px;'
                    f'margin:10px 0">{", ".join(_cols)}</div>', unsafe_allow_html=True)
        st.download_button("Descargar plantilla de ejemplo",
                           _ej.to_csv(index=False).encode("utf-8"),
                           "plantilla_aldaba.csv", "text/csv")
    with e2:
        _up = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if _up is not None:
        try:
            _t = pd.read_csv(_up)
            _falta = [c for c in _cols[:5] if c not in _t.columns]
            if _falta:
                st.markdown(f'<div class="aviso">Faltan columnas: '
                            f'<b>{", ".join(_falta)}</b>.</div>', unsafe_allow_html=True)
            else:
                _t = _t.head(300).copy()
                _rows, _facs = [], []
                for _, r in _t.iterrows():
                    _d = str(r["distrito"]) if str(r["distrito"]) in meta["distritos"] \
                         else distrito
                    _s = cen[cen.distrito == _d]
                    _la = float(_s.lat.mean()) if len(_s) else meta["centro"]["lat"]
                    _lo = float(_s.lon.mean()) if len(_s) else meta["centro"]["lon"]
                    _rr = dict(meta["medianas"])
                    _rr.update({"CONSTRUCTEDAREA": float(r["m2"]),
                                "ROOMNUMBER": float(r["habitaciones"]),
                                "BATHNUMBER": float(r["banos"]),
                                "CADCONSTRUCTIONYEAR": float(r["anio"]),
                                "CONSTRUCTIONYEAR": float(r["anio"]),
                                "LATITUDE": _la, "LONGITUDE": _lo,
                                "DISTANCE_TO_METRO": 0.4, "HASLIFT": 1,
                                "DISTANCE_TO_CITY_CENTER": float(np.sqrt(
                                    ((_la - clat) * KM) ** 2 +
                                    ((_lo - clon) * KM *
                                     math.cos(math.radians(clat))) ** 2)),
                                "PERIOD": 201812})
                    _rows.append({c: _rr.get(c, 0.0) for c in FEAT})
                    _facs.append(meta["distritos"].get(_d, {}).get(
                        "factor", meta["factor_ciudad"]))
                _X = xgb.DMatrix(pd.DataFrame(_rows)[FEAT].astype(float))
                _fa = np.array(_facs)
                _ar = _t["m2"].astype(float).values
                _e50 = np.exp(mods["q50"].predict(_X)) * _fa * _ar
                _e10 = np.exp(mods["q10"].predict(_X)) * _fa * _ar
                _e90 = np.exp(mods["q90"].predict(_X)) * _fa * _ar
                _e10 = np.minimum(_e10, _e50); _e90 = np.maximum(_e90, _e50)
                _t["valoracion"] = _e50.round(0).astype(int)
                _t["minimo"] = _e10.round(0).astype(int)
                _t["maximo"] = _e90.round(0).astype(int)
                if "precio_pedido" in _t.columns:
                    _pp = pd.to_numeric(_t["precio_pedido"], errors="coerce")
                    _t["veredicto"] = np.where(_pp < _e10, "Está barata",
                                       np.where(_pp > _e90, "Está cara", "Precio normal"))
                    _t["margen"] = (_e50 - _pp).round(0)
                    _n_op = int((_t["veredicto"] == "Está barata").sum())
                    st.markdown(
                        f'<div class="kpi" style="margin-top:24px">'
                        f'<div><div class="v">{len(_t)}</div>'
                        f'<div class="k">viviendas valoradas</div></div>'
                        f'<div><div class="v" style="color:{VERDE}">{_n_op}</div>'
                        f'<div class="k">por debajo de su banda</div></div>'
                        f'<div><div class="v">{eur(_e50.sum())}</div>'
                        f'<div class="k">valor total de la cartera</div></div></div>',
                        unsafe_allow_html=True)
                    _t = _t.sort_values("margen", ascending=False)
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.dataframe(_t, use_container_width=True, hide_index=True)
                st.download_button("Descargar resultados",
                                   _t.to_csv(index=False).encode("utf-8"),
                                   "valoraciones_aldaba.csv", "text/csv")
        except Exception as _e:
            st.markdown(f'<div class="aviso">No he podido leer el archivo: {_e}</div>',
                        unsafe_allow_html=True)


if tabS.activa:
    st.markdown('<h2 class="sec">Viviendas reales parecidas a la tuya</h2>'
                '<p class="sub">Anuncios del conjunto de datos con características '
                'similares, con su precio real reindexado a hoy. Sirve para contrastar '
                'que la estimación no se ha inventado nada.</p>', unsafe_allow_html=True)
    _CP = extra("comparables.json").get(ciudad, [])
    if not _CP:
        st.markdown('<div class="aviso">Todavía no hay muestra de comparables cargada '
                    'para esta ciudad.</div>', unsafe_allow_html=True)
    else:
        _c = pd.DataFrame(_CP)
        # distancia en características, normalizada
        _z = np.sqrt(((_c["m2"] - area) / 45) ** 2 + ((_c["hab"] - rooms) / 1.4) ** 2 +
                     ((_c["ban"] - baths) / 1.0) ** 2 +
                     ((_c["anio"] - year) / 30) ** 2 +
                     (((_c["d"] != distrito).astype(float)) * 2.2) ** 2)
        _c = _c.assign(_z=_z).nsmallest(8, "_z")
        _fac = {d: (meta["distritos"].get(d, {}) or {}).get("factor",
                    meta["factor_ciudad"]) for d in _c["d"].unique()}
        _c["hoy"] = [int(p * _fac.get(d, meta["factor_ciudad"]))
                     for p, d in zip(_c["precio"], _c["d"])]
        _fil = "".join(
            f'<tr><td>{r["d"]}</td><td class="m">{num(r["m2"])} m²</td>'
            f'<td class="m">{num(r["hab"])}</td>'
            f'<td class="m">{num(r["ban"])}</td>'
            f'<td class="m">{num(r["anio"])}</td>'
            f'<td class="m">{eur(r["precio"])}</td>'
            f'<td class="m" style="color:{ACC}">{eur(r["hoy"])}</td></tr>'
            for _, r in _c.iterrows())
        st.markdown(
            f'<table class="tb"><caption>Ordenadas por parecido con la vivienda que has '
            f'descrito: {num(area)} m², {rooms} hab., {baths} baños, {year}, '
            f'{distrito}.</caption>'
            f'<tr><th>Barrio</th><th>Superficie</th><th>Hab.</th><th>Baños</th>'
            f'<th>Año</th><th>Precio 2018</th><th>Equivalente hoy</th></tr>{_fil}</table>',
            unsafe_allow_html=True)
        _med = float(_c["hoy"].median())
        _dd = (p50 - _med) / _med * 100
        st.markdown(
            f'<div class="kpi" style="margin-top:26px">'
            f'<div><div class="v">{eur(_med)}</div>'
            f'<div class="k">precio típico de estas ocho, a día de hoy</div></div>'
            f'<div><div class="v">{eur(p50)}</div>'
            f'<div class="k">lo que estima el modelo para la tuya</div></div>'
            f'<div><div class="v" style="color:{ACC}">'
            f'{"+" if _dd>=0 else ""}{_dd:.0f} %</div>'
            f'<div class="k">de diferencia</div></div></div>'
            f'<div class="nota" style="margin-top:12px;max-width:72ch">Los precios de 2018 '
            f'se reindexan con el factor del barrio, el mismo que usa la valoración. Una '
            f'diferencia grande no significa error: tu vivienda puede tener ascensor, '
            f'terraza o mejor planta que estas.</div>', unsafe_allow_html=True)

# ─────────────────────── pestaña · fiabilidad ─────────────────────────
if tabM.activa:
    R = resumen()
    # normalizar: si resumen devuelve metas completos, extraer las claves necesarias
    def _stat(d, k, default=0):
        if k in d: return d[k]
        if 'r2_bloques' in d and k == 'n_anuncios': return d.get('n_anuncios', default)
        return d.get(k, default)
    st.markdown('<h2 class="sec">Hasta qué punto acierta</h2>'
                '<p class="sub">Medido sobre viviendas que el modelo no había visto nunca, '
                'apartando barrios enteros para que no pueda copiar de vecinos.</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi">'
        f'<div><div class="v">{meta["r2_bloques"]*100:.1f} %</div>'
        f'<div class="k">de la variación de precios queda explicada</div></div>'
        f'<div><div class="v">{eur(meta["mae"])}</div>'
        f'<div class="k">de error medio por vivienda</div></div>'
        f'<div><div class="v">{meta["cobertura_intervalo"]*100:.0f} de cada 100</div>'
        f'<div class="k">viviendas caen dentro del margen que damos</div></div>'
        f'<div><div class="v">{num(meta["n_anuncios"])}</div>'
        f'<div class="k">viviendas reales analizadas en {ciudad}</div></div></div>',
        unsafe_allow_html=True)

    _TODOS = extra("error_distrito.json").get(ciudad, {})
    _ED = _TODOS.get(distrito)
    if _ED:
        _ord = sorted(_TODOS.items(), key=lambda x: x[1]["pct"])
        _pos = [k for k, _ in _ord].index(distrito) + 1
        _tot = len(_ord)
        if _pos <= _tot / 3:
            _cal = "de los más fiables"
        elif _pos > 2 * _tot / 3:
            _cal = "de los menos fiables"
        else:
            _cal = "en la media"
        st.markdown(
            f'<div class="aviso" style="margin-top:22px">'
            f'<b>En {distrito} concretamente:</b> el error típico es del '
            f'<b>{_ED["pct"]:.0f} %</b>, unos {eur(_ED["mae"])} de media. '
            f'Es el barrio número {_pos} de {_tot} en precisión, {_cal}. '
            f'Calculado sobre {num(_ED["n"])} viviendas reales del barrio.</div>',
            unsafe_allow_html=True)
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns([1.15, 1], gap="large")
    with m1:
        st.markdown('<h2 class="sec">Las tres ciudades</h2>', unsafe_allow_html=True)
        fl = "".join(
            f'<tr><td>{c}</td><td class="m">{R[c]["r2_bloques"]*100:.1f} %</td>'
            f'<td class="m">{eur(R[c]["mae"])}</td>'
            f'<td class="m">{R[c]["cobertura_intervalo"]*100:.0f} %</td>'
            f'<td class="m">{num(R[c]["n_anuncios"])}</td></tr>' for c in CIUDADES)
        st.markdown(
            f'<table class="tb"><tr><th>Ciudad</th><th>Acierto</th><th>Error medio</th>'
            f'<th>Dentro del margen</th><th>Viviendas</th></tr>{fl}</table>'
            f'<div class="nota" style="margin-top:14px">València cuesta menos de la mitad '
            f'que Madrid y aun así el modelo funciona: pierde '
            f'<b>{(_stat(R.get("Madrid",{}),"r2_bloques",0)-_stat(R.get("Valencia",{}),"r2_bloques",0))*100:.1f} puntos'
            f'</b> de acierto. Es la prueba de que el método sirve en mercados distintos.'
            f'</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<h2 class="sec">Cuánto ha subido cada barrio</h2>'
                    '<p class="sub">Desde 2018 hasta hoy.</p>', unsafe_allow_html=True)
        dd = sorted([(d, v) for d, v in meta["distritos"].items()
                     if v["n"] >= 120 and not v["estimado"]],
                    key=lambda x: -x[1]["factor"])[:10]
        desc_subidas = "; ".join(
            f"{d} más {(v['factor']-1)*100:.0f} por ciento" for d, v in dd)
        if dd:
            mxf = max(v["factor"] for _, v in dd)
            rows = "".join(
                f'<text x="0" y="{i*30+14}" fill="{INK}" font-size="13" '
                f'font-family="Instrument Sans,sans-serif">{d}</text>'
                f'<rect x="190" y="{i*30+3}" width="{max(v["factor"]/mxf*180,2):.1f}" '
                f'height="13" fill="{ACC}" opacity="{.35+.65*v["factor"]/mxf:.2f}" rx="1"/>'
                f'<text x="{190+max(v["factor"]/mxf*180,2)+10:.1f}" y="{i*30+14}" '
                f'fill="{INK}" font-size="12" font-family="IBM Plex Mono,monospace">'
                f'+{(v["factor"]-1)*100:.0f} %</text>'
                f'<line x1="0" y1="{i*30+22}" x2="460" y2="{i*30+22}" stroke="{LINE}" '
                f'opacity=".6"/>' for i, (d, v) in enumerate(dd))
            st.markdown(
                f'<svg viewBox="0 0 460 {len(dd)*30}" width="100%" role="img" '
                f'aria-label="Subida de precios por barrio desde 2018. '
                f'{desc_subidas}." '
                f'style="display:block">{rows}</svg>', unsafe_allow_html=True)
        st.markdown('<div class="nota" style="margin-top:12px">No todos los barrios han '
                    'subido igual, por eso actualizamos barrio a barrio y no con una media '
                    'de ciudad.</div>', unsafe_allow_html=True)

# ─────────────────────── pestaña · cómo funciona ──────────────────────
if tabD.activa:
    st.markdown('<h2 class="sec">Cómo se calcula</h2>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="large")
    bloques = [
        ("De dónde salen los datos",
         f"De {num(meta['n_anuncios'])} viviendas reales puestas a la venta en {ciudad}, "
         f"con sus características y su precio, publicadas por idealista para uso "
         f"académico y completadas con datos del Catastro."),
        ("Qué aprende el modelo",
         "Cuánto suma o resta cada cosa: un baño más, un ascensor, estar a cinco minutos "
         "del metro, ser un bajo o un ático. Aprende proporciones, no precios absolutos."),
        ("Por qué está actualizado",
         "Las proporciones cambian poco con los años, pero los precios suben. Por eso el "
         "nivel de precios se toma del índice publicado cada mes, barrio a barrio. "
         "Comparando el mapa de 2018 con el de hoy, el orden de los barrios es casi "
         "idéntico: lo que ha cambiado es cuánto cuestan, no cuáles son los caros."),
        ("Por qué damos un margen y no una cifra exacta",
         f"Ninguna estimación acierta al euro. En vez de fingir precisión, damos una franja "
         f"y decimos cuánto acierta: {meta['cobertura_intervalo']*100:.0f} de cada 100 "
         f"viviendas caen dentro de ella. En pisos corrientes la franja es estrecha; en "
         f"viviendas raras, ancha."),
        ("Cómo comprobamos que acierta",
         "Apartamos barrios enteros, entrenamos sin ellos y luego pedimos al modelo que los "
         "valore. Así no puede acertar copiando del piso de al lado, que es la trampa más "
         "habitual en este tipo de modelos."),
        ("Qué no puede hacer",
         f"No sabe si la cocina está reformada ni si hay obras enfrente. Si desde 2018 ha "
         f"cambiado lo que la gente valora —por ejemplo la terraza tras la pandemia— tampoco "
         f"lo recoge. Por encima de {p99} m² hay pocas viviendas con las que comparar. "
         f"No sustituye a una tasación oficial."),
    ]
    for i, (t, c) in enumerate(bloques):
        with (d1 if i % 2 == 0 else d2):
            st.markdown(f'<div style="margin-bottom:30px"><h2 class="sec">{t}</h2>'
                        f'<div class="nota">{c}</div></div>', unsafe_allow_html=True)

    with st.expander("Detalle técnico y fuentes"):
        st.markdown(f"""
<div class="nota">
<b>Conjunto de datos.</b> <i>idealista18</i> — Rey-Blanco, Arbués, López y Páez (2024),
<i>Environment and Planning B: Urban Analytics and City Science</i>,
DOI 10.1177/23998083241242844, licencia ODbL. {num(meta['n_anuncios'])} anuncios de
{ciudad} ({num(meta['n_viviendas'])} viviendas únicas) de los cuatro trimestres de 2018.<br><br>
<b>Modelo.</b> Árboles con refuerzo de gradiente sobre el logaritmo del precio por metro
cuadrado, con restricciones de monotonía en las variables de signo inequívoco. Modelar el
precio unitario evita que la estimación se sature en viviendas grandes.<br><br>
<b>Incertidumbre.</b> Regresión cuantílica en los percentiles 10, 50 y 90. Cobertura
empírica {meta['cobertura_intervalo']*100:.1f} % frente al 80 % teórico.<br><br>
<b>Validación.</b> Cruzada por bloques espaciales de unos 100 metros, que agrupan los
anuncios repetidos del mismo inmueble entre trimestres.<br><br>
<b>Actualización.</b> Nivel de precios por distrito de agosto de 2026. Correlación de rangos
entre el mapa de valor de 2018 y el actual: 0,975 en Madrid.
</div>""", unsafe_allow_html=True)

# ───────────────────────────── accesibilidad ──────────────────────────
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
with st.expander("Accesibilidad"):
    x1, x2 = st.columns(2)
    x1.checkbox("Alto contraste", key="alto_contraste",
                help="Blanco y negro puros, bordes marcados y colores más saturados.")
    x2.checkbox("Texto grande", key="texto_grande",
                help="Aumenta un 16 % el tamaño de toda la letra.")
    st.markdown('<div class="nota" style="margin-top:10px">Todos los gráficos llevan '
                'descripción para lectores de pantalla, los avisos no dependen solo del '
                'color y las animaciones se desactivan si tu sistema lo pide.</div>',
                unsafe_allow_html=True)

st.markdown(
    f'<div class="pie"><div>© 2026 Aldaba</div>'
    f'<div>Datos idealista18 (2018) · Precios de {IDX["fecha"]}</div> (<a href=\"https://github.com/paezha/idealista18\" target=\"_blank\" style=\"color:{MUTE};text-decoration:underline\">enlace</a>)'
    f'<div>Estimación orientativa · No sustituye a una tasación oficial</div></div>',
    unsafe_allow_html=True)
