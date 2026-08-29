import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, math, base64
from io import BytesIO
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="VALORA · Cartografía del valor residencial",
                   layout="wide", initial_sidebar_state="collapsed")

VOID, DEEP, RULE = "#06090C", "#0B1116", "#17232A"
BONE, MUTE = "#EDEFEA", "#77878C"
AMBER, AMBER_DIM = "#E39B3C", "#8A5F26"

CENTRO = {"Madrid": (40.4168, -3.7038), "Barcelona": (41.3874, 2.1686)}
STOPS = [(0.00, (10, 20, 32)), (0.30, (16, 64, 63)), (0.55, (28, 122, 98)),
         (0.76, (184, 134, 58)), (0.91, (227, 155, 60)), (1.00, (247, 223, 180))]


@st.cache_resource
def cargar():
    return (joblib.load(BASE / "modelo_valoracion.joblib"),
            json.loads((BASE / "resultados_tfm.json").read_text(encoding="utf-8")))


model, meta = cargar()
FEATS = meta["features"]


# ─────────────────────────────── formato ───────────────────────────────
def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def eur(x, dec=0):
    return f"{num(x, dec)} €"


def compact(v):
    return f"{num(v/1_000_000, 1)}M" if v >= 1_000_000 else f"{num(v/1000)}k"


# ─────────────────────────────── modelo ────────────────────────────────
def base_dict(city, area, rooms, baths, year, d_metro, ext):
    d = {c: 0 for c in FEATS}
    d.update({"CITY": city, "CONSTRUCTEDAREA": area, "ROOMNUMBER": rooms,
              "BATHNUMBER": baths, "CADCONSTRUCTIONYEAR": year,
              "DISTANCE_TO_METRO": d_metro, "AMENITYID": 3,
              "CADMAXBUILDINGFLOOR": 6, "CADDWELLINGCOUNT": 20,
              "CADASTRALQUALITYID": 5, "BUILTTYPEID_3": 1, "FLOORCLEAN": 2,
              "FLATLOCATIONID": 1, "HASLIFT": ext[0], "HASTERRACE": ext[1],
              "HASPARKINGSPACE": ext[2], "HASAIRCONDITIONING": ext[3],
              "HASSWIMMINGPOOL": ext[4], "HASDOORMAN": ext[5]})
    d["BUILDING_AGE"] = max(0, 2018 - year)
    d["AREA_PER_ROOM"] = area / max(rooms, 1)
    d["FLOOR_RATIO"] = 2 / 6
    d["NEAR_METRO_500M"] = int(d_metro <= 0.5)
    d["AMENITIES_COUNT"] = sum(ext)
    return d


def predecir(rows):
    return np.expm1(model.predict(pd.DataFrame(rows)[FEATS]))


@st.cache_data(show_spinner=False)
def campo(city, area, rooms, baths, year, d_metro, ext, nx=76, ny=50):
    """Superficie de precio del activo desplazado por toda la ciudad."""
    la0, lo0 = CENTRO[city]
    las = np.linspace(la0 + 0.075, la0 - 0.075, ny)
    los = np.linspace(lo0 - 0.098, lo0 + 0.098, nx)
    LO, LA = np.meshgrid(los, las)
    dist = np.sqrt(((LA - la0) * 111.0) ** 2 +
                   ((LO - lo0) * 111.0 * math.cos(math.radians(la0))) ** 2)
    df = pd.DataFrame([base_dict(city, area, rooms, baths, year, d_metro, ext)] * LA.size)
    df["LATITUDE"] = LA.ravel()
    df["LONGITUDE"] = LO.ravel()
    df["DISTANCE_TO_CITY_CENTER"] = dist.ravel()
    return np.expm1(model.predict(df[FEATS])).reshape(ny, nx).astype(np.float64)


def ramp(t):
    ps = np.array([s[0] for s in STOPS])
    out = np.zeros(t.shape + (3,), dtype=np.uint8)
    for k in range(3):
        cs = np.array([s[1][k] for s in STOPS], dtype=float)
        out[..., k] = np.clip(np.interp(t, ps, cs), 0, 255).astype(np.uint8)
    return out


@st.cache_data(show_spinner=False)
def campo_png(Z_bytes, shape, lo, hi):
    Z = np.frombuffer(Z_bytes, dtype=np.float64).reshape(shape)
    t = np.clip((Z - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.85
    img = Image.fromarray(ramp(t), "RGB").resize(
        (shape[1] * 7, shape[0] * 7), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── curvas de nivel (marching squares) ──────────────────────────────────
def isolineas(Z, niveles, W, H):
    ny, nx = Z.shape
    sx, sy = W / (nx - 1), H / (ny - 1)
    salida = []
    for lv in niveles:
        d = []
        G = Z > lv
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b = Z[j, i], Z[j, i + 1]
                c, e = Z[j + 1, i + 1], Z[j + 1, i]
                idx = G[j, i] | (G[j, i + 1] << 1) | (G[j + 1, i + 1] << 2) | (G[j + 1, i] << 3)
                if idx in (0, 15):
                    continue
                T = ((i + (lv - a) / (b - a)) * sx, j * sy) if b != a else None
                R = ((i + 1) * sx, (j + (lv - b) / (c - b)) * sy) if c != b else None
                B = ((i + (lv - e) / (c - e)) * sx, (j + 1) * sy) if c != e else None
                L = (i * sx, (j + (lv - a) / (e - a)) * sy) if e != a else None
                pares = {1: [(L, T)], 2: [(T, R)], 3: [(L, R)], 4: [(R, B)],
                         5: [(L, T), (R, B)], 6: [(T, B)], 7: [(L, B)], 8: [(L, B)],
                         9: [(T, B)], 10: [(L, T), (R, B)], 11: [(T, R)],
                         12: [(L, R)], 13: [(R, B)], 14: [(L, T)]}[idx]
                for p, q in pares:
                    if p and q:
                        d.append(f"M{p[0]:.1f} {p[1]:.1f}L{q[0]:.1f} {q[1]:.1f}")
        salida.append("".join(d))
    return salida


# ─────────────────────────────── estilos ───────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,400;6..96,500&family=IBM+Plex+Mono:wght@400;500&family=Jost:wght@300;400;500&display=swap');

header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stSidebarCollapsedControl"]{{display:none!important}}
.stApp{{background:{VOID}}}
.block-container{{padding:2rem 3rem 4rem!important;max-width:1340px}}
html,body,[class*="css"]{{font-family:'Jost',sans-serif;color:{BONE}}}
.m{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}

.mast{{display:flex;align-items:baseline;justify-content:space-between;
 border-bottom:1px solid {RULE};padding-bottom:14px;margin-bottom:26px}}
.wm{{font-family:'Bodoni Moda',serif;font-size:1.7rem;letter-spacing:.42em;
 text-transform:uppercase;color:{BONE};font-weight:400}}
.mast .sub{{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.24em;
 text-transform:uppercase;color:{MUTE}}}

.eb{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.26em;
 text-transform:uppercase;color:{MUTE};display:flex;align-items:center;gap:14px;margin:0 0 14px}}
.eb::after{{content:"";flex:1;height:1px;background:{RULE}}}
.grp{{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.22em;
 text-transform:uppercase;color:{AMBER_DIM};padding-bottom:8px;
 border-bottom:1px solid {RULE};margin-bottom:14px}}

/* ── escena ── */
.scene{{position:relative;border:1px solid {RULE};overflow:hidden;background:{DEEP};
 animation:unveil 1.3s cubic-bezier(.16,.84,.34,1) both}}
@keyframes unveil{{from{{clip-path:inset(0 100% 0 0)}}to{{clip-path:inset(0 0 0 0)}}}}
.scene img{{display:block;width:100%;height:auto;filter:saturate(1.06) contrast(1.04)}}
.scene .ov{{position:absolute;inset:0}}
.scene .scrim{{position:absolute;inset:0;
 background:linear-gradient(100deg,{VOID}F2 0%,{VOID}D8 26%,{VOID}22 56%,transparent 74%)}}
.iso{{fill:none;stroke:{BONE};stroke-width:.6;opacity:.16}}
.iso.hi{{stroke:{AMBER};opacity:.34;stroke-width:.8}}

.plate{{position:absolute;left:3.2%;top:50%;transform:translateY(-50%);max-width:46%}}
.tag{{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.26em;
 text-transform:uppercase;color:{AMBER};margin-bottom:14px}}
.val{{font-family:'Bodoni Moda',serif;font-size:5.6rem;line-height:.88;font-weight:400;
 letter-spacing:-.022em;color:{BONE};text-shadow:0 0 62px rgba(227,155,60,.30)}}
.val sup{{font-size:2.1rem;font-weight:400;color:{AMBER};margin-left:8px;top:-1.9rem}}
.sq{{margin-top:20px;display:flex;gap:22px;flex-wrap:wrap}}
.sq div{{border-left:1px solid {AMBER_DIM};padding-left:11px}}
.sq .v{{font-family:'IBM Plex Mono',monospace;font-size:.95rem;color:{BONE}}}
.sq .k{{font-family:'IBM Plex Mono',monospace;font-size:.56rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE};margin-top:3px}}

@keyframes halo{{0%,100%{{r:7;opacity:.85}}50%{{r:17;opacity:0}}}}
.pin-halo{{animation:halo 3.4s ease-out infinite}}

.lgd{{display:flex;align-items:center;gap:11px;margin-top:11px;
 font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.16em;
 text-transform:uppercase;color:{MUTE}}}
.lgd .bar{{flex:0 0 190px;height:5px;background:linear-gradient(90deg,
 rgb(10,20,32),rgb(16,64,63) 30%,rgb(28,122,98) 55%,rgb(184,134,58) 76%,
 rgb(227,155,60) 91%,rgb(247,223,180))}}

.prose{{font-family:'Bodoni Moda',serif;font-size:1.02rem;line-height:1.66;
 color:{MUTE};font-weight:400}}
.prose b{{color:{BONE};font-weight:500}}

.diag{{display:flex;border-top:1px solid {AMBER_DIM};border-bottom:1px solid {RULE}}}
.diag>div{{flex:1;padding:17px 0 15px;border-right:1px solid {RULE}}}
.diag>div:last-child{{border-right:none}}
.diag .v{{font-family:'IBM Plex Mono',monospace;font-size:1.42rem;letter-spacing:-.02em;color:{BONE}}}
.diag .k{{font-family:'IBM Plex Mono',monospace;font-size:.56rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE};margin-top:6px}}

.foot{{border-top:1px solid {RULE};margin-top:46px;padding-top:18px;display:flex;
 justify-content:space-between;gap:18px;flex-wrap:wrap;
 font-family:'IBM Plex Mono',monospace;font-size:.56rem;letter-spacing:.2em;
 text-transform:uppercase;color:{MUTE}}}

.rise{{animation:rise .8s cubic-bezier(.16,.84,.34,1) both}}
@keyframes rise{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
@media(prefers-reduced-motion:reduce){{.scene,.rise{{animation:none}}.pin-halo{{animation:none}}}}

[data-testid="stWidgetLabel"] p{{font-family:'Jost',sans-serif!important;
 font-size:.78rem!important;color:{MUTE}!important;font-weight:300!important;
 letter-spacing:.03em}}
[data-testid="stExpander"]{{border:1px solid {RULE};background:{DEEP};border-radius:2px}}
[data-testid="stExpander"] summary p{{font-family:'IBM Plex Mono',monospace!important;
 font-size:.6rem!important;letter-spacing:.2em;text-transform:uppercase;color:{MUTE}!important}}
[data-baseweb="select"]>div{{background:{DEEP}!important;border-color:{RULE}!important;
 border-radius:2px!important}}
[data-testid="stNumberInput"] input{{background:{DEEP}!important;color:{BONE}!important;
 border-color:{RULE}!important}}
[data-testid="stCheckbox"] p{{color:{MUTE}!important;font-size:.78rem!important}}

@media(max-width:1000px){{
 .block-container{{padding:1.3rem 1.1rem 3rem!important}}
 .plate{{position:static;transform:none;max-width:100%;padding:20px 18px 4px}}
 .scene .scrim{{background:linear-gradient(180deg,transparent 40%,{VOID}EE)}}
 .val{{font-size:3.1rem}}.val sup{{font-size:1.3rem;top:-1rem}}
 .diag{{flex-wrap:wrap}}.diag>div{{flex:1 0 50%;border-bottom:1px solid {RULE}}}
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────── consola ───────────────────────────────
st.markdown('<div class="mast"><div class="wm">Valora</div>'
            '<div class="sub">Cartografía del valor · Mercado residencial español</div></div>',
            unsafe_allow_html=True)

k1, k2, k3 = st.columns([1, 1, 1.05], gap="large")
with k1:
    st.markdown('<div class="grp">Plaza</div>', unsafe_allow_html=True)
    city = st.selectbox("Mercado", ["Madrid", "Barcelona"], label_visibility="collapsed")
    d_metro = st.slider("Distancia al metro (km)", 0.0, 10.0, 0.4, 0.1)
    d_center = st.slider("Distancia al centro (km)", 0.0, 30.0, 3.0, 0.1)
with k2:
    st.markdown('<div class="grp">Activo</div>', unsafe_allow_html=True)
    area = st.slider("Superficie construida (m²)", 20, 500, 90)
    year = st.slider("Año de construcción", 1900, 2018, 1970)
    r1, r2 = st.columns(2)
    rooms = r1.number_input("Habitaciones", 0, 15, 3)
    baths = r2.number_input("Baños", 0, 10, 2)
with k3:
    st.markdown('<div class="grp">Dotaciones</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    lift = d1.checkbox("Ascensor", value=True)
    terrace = d2.checkbox("Terraza")
    parking = d1.checkbox("Plaza de garaje")
    air = d2.checkbox("Climatización")
    pool = d1.checkbox("Piscina")
    doorman = d2.checkbox("Portería")

ext = (int(lift), int(terrace), int(parking), int(air), int(pool), int(doorman))
la0, lo0 = CENTRO[city]

# ─────────────────────────────── cálculo ───────────────────────────────
d = base_dict(city, area, rooms, baths, year, d_metro, ext)
d.update({"LATITUDE": la0, "LONGITUDE": lo0 + d_center / (111.0 * math.cos(math.radians(la0))),
          "DISTANCE_TO_CITY_CENTER": d_center})
precio = float(predecir([d])[0])
q = float(meta["p90_abs_error_eur"])
bajo, alto = max(0.0, precio - q), precio + q

Z = campo(city, area, rooms, baths, year, d_metro, ext)
lo, hi = float(np.percentile(Z, 1)), float(np.percentile(Z, 99))
b64 = campo_png(np.ascontiguousarray(Z).tobytes(), Z.shape, lo, hi)

W, H = 1000, 340
niveles = list(np.linspace(lo, hi, 11))[1:-1]
paths = isolineas(Z, niveles, W, H)
prox = int(np.argmin([abs(n - precio) for n in niveles]))

px = (d["LONGITUDE"] - (lo0 - 0.098)) / 0.196 * W
py = ((la0 + 0.075) - la0) / 0.15 * H

# ─────────────────────────────── escena ────────────────────────────────
iso_svg = "".join(
    f'<path class="iso{" hi" if i == prox else ""}" d="{p}"/>'
    for i, p in enumerate(paths) if p)

st.markdown(f"""
<div class="scene">
  <img src="data:image/png;base64,{b64}" alt="Superficie de valor de {city}"/>
  <svg class="ov" viewBox="0 0 {W} {H}" preserveAspectRatio="none">{iso_svg}</svg>
  <svg class="ov" viewBox="0 0 {W} {H}">
    <circle class="pin-halo" cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none"
            stroke="{AMBER}" stroke-width="1.4"/>
    <circle cx="{px:.1f}" cy="{py:.1f}" r="4.6" fill="{AMBER}"/>
    <circle cx="{px:.1f}" cy="{py:.1f}" r="4.6" fill="none" stroke="{VOID}" stroke-width="1.4"/>
  </svg>
  <div class="scrim"></div>
  <div class="plate">
    <div class="tag">Valoración · {city}</div>
    <div class="val">{num(precio)}<sup>€</sup></div>
    <div class="sq">
      <div><div class="v">{eur(precio/area)}</div><div class="k">por m²</div></div>
      <div><div class="v">{num(area)} m²</div><div class="k">superficie</div></div>
      <div><div class="v">{2018-year}</div><div class="k">años</div></div>
      <div><div class="v">±{q/precio*100:.0f}%</div><div class="k">holgura</div></div>
    </div>
  </div>
</div>
<div class="lgd"><span class="bar"></span>
 <span>{compact(lo)} €</span><span style="flex:1"></span><span>{compact(hi)} €</span></div>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="rise" style="margin-top:26px">
<div class="prose" style="max-width:72ch">El mismo activo, desplazado por toda la plaza.
La superficie recoge la valoración que el modelo asigna a una vivienda de
<b>{num(area)} m²</b> en cada punto de {city}; las curvas de nivel separan tramos de
<b>{eur((hi-lo)/10)}</b> y la resaltada marca el tramo en el que cae este activo.
El rango va de <b>{eur(lo)}</b> en la corona exterior a <b>{eur(hi)}</b> en el núcleo.</div></div>
""", unsafe_allow_html=True)

# ─────────────────────────── curvas de respuesta ───────────────────────
def curva(xs, ys, xn, yn, unidad, ident):
    Wc, Hc, L, R, T, B = 520, 200, 12, 508, 26, 46
    xmin, xmax, ymin, ymax = float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))
    pad = (ymax - ymin) * .18 or 1
    ymin, ymax = ymin - pad, ymax + pad
    fx = lambda v: L + (v - xmin) / max(xmax - xmin, 1e-9) * (R - L)
    fy = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (Hc - B - T)
    pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
    g = "".join(f'<line x1="{L}" y1="{T+(Hc-B-T)*i/4:.1f}" x2="{R}" '
                f'y2="{T+(Hc-B-T)*i/4:.1f}" stroke="{RULE}"/>' for i in range(1, 4))
    ex = lambda v: num(v, 1).rstrip("0").rstrip(",") if unidad == " km" else num(v)
    return (f'<svg viewBox="0 0 {Wc} {Hc}" width="100%" style="display:block;overflow:visible">'
            f'<defs><linearGradient id="g{ident}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{AMBER}" stop-opacity=".26"/>'
            f'<stop offset="1" stop-color="{AMBER}" stop-opacity="0"/></linearGradient></defs>{g}'
            f'<polygon points="{L},{Hc-B} {pts} {R},{Hc-B}" fill="url(#g{ident})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{AMBER}" stroke-width="1.8" '
            f'stroke-linejoin="round"/>'
            f'<line x1="{fx(xn):.1f}" y1="{T-8}" x2="{fx(xn):.1f}" y2="{Hc-B}" '
            f'stroke="{AMBER_DIM}" stroke-dasharray="2 4"/>'
            f'<circle cx="{fx(xn):.1f}" cy="{fy(yn):.1f}" r="4.4" fill="{AMBER}" '
            f'stroke="{VOID}" stroke-width="1.6"/>'
            f'<line x1="{L}" y1="{Hc-B}" x2="{R}" y2="{Hc-B}" stroke="{RULE}"/>'
            f'<g font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{MUTE}">'
            f'<text x="{L}" y="{Hc-B+18}">{ex(xmin)}{unidad}</text>'
            f'<text x="{R}" y="{Hc-B+18}" text-anchor="end">{ex(xmax)}{unidad}</text>'
            f'<text x="{L}" y="{T-12}">{compact(max(ys))} €</text></g></svg>')


st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
g1, g2 = st.columns(2, gap="large")

xa = np.linspace(max(20, area - 55), min(500, area + 55), 24)
ya = predecir([{**base_dict(city, float(a), rooms, baths, year, d_metro, ext),
                "LATITUDE": la0, "LONGITUDE": d["LONGITUDE"],
                "DISTANCE_TO_CITY_CENTER": d_center} for a in xa])
with g1:
    st.markdown('<div class="eb">Elasticidad · superficie</div>', unsafe_allow_html=True)
    st.markdown(curva(list(xa), list(ya), area, precio, " m²", "a"), unsafe_allow_html=True)
    pend = (float(ya[-1]) - float(ya[0])) / (xa[-1] - xa[0])
    st.markdown(f'<div class="prose" style="font-size:.9rem;margin-top:8px">Cada m² adicional '
                f'en torno a los {area} m² añade <b>{eur(pend)}</b>.</div>', unsafe_allow_html=True)

xd = np.linspace(0.2, min(20.0, max(9.0, d_center + 6)), 24)
yd = predecir([{**base_dict(city, area, rooms, baths, year, d_metro, ext),
                "LATITUDE": la0,
                "LONGITUDE": lo0 + float(v) / (111.0 * math.cos(math.radians(la0))),
                "DISTANCE_TO_CITY_CENTER": float(v)} for v in xd])
with g2:
    st.markdown('<div class="eb">Gradiente · centralidad</div>', unsafe_allow_html=True)
    st.markdown(curva(list(xd), list(yd), d_center, precio, " km", "d"), unsafe_allow_html=True)
    caida = (float(yd[0]) - float(yd[-1])) / float(yd[0]) * 100
    st.markdown(f'<div class="prose" style="font-size:.9rem;margin-top:8px">Alejarse hasta '
                f'{num(xd[-1],1)} km del núcleo resta un <b>{caida:.0f}%</b> del valor.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────── intervalo ─────────────────────────────
st.markdown("<div style='height:42px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eb">Intervalo de confianza sobre escala absoluta</div>',
            unsafe_allow_html=True)
Ws, Hs, x0, x1, bl = 1000, 92, 6, 994, 50
amax = math.ceil(alto * 1.12 / 50_000) * 50_000
sx = lambda v: x0 + v / amax * (x1 - x0)
ticks = "".join(
    f'<line x1="{sx(amax*i/10):.1f}" y1="{bl}" x2="{sx(amax*i/10):.1f}" '
    f'y2="{bl+(8 if i%2==0 else 4)}" stroke="{RULE}"/>' +
    (f'<text x="{sx(amax*i/10):.1f}" y="{bl+23}" fill="{MUTE}" font-size="10" '
     f'font-family="IBM Plex Mono,monospace" text-anchor="middle">{compact(amax*i/10)}</text>'
     if i % 2 == 0 else "") for i in range(11))
st.markdown(
    f'<svg viewBox="0 0 {Ws} {Hs}" width="100%" preserveAspectRatio="none" '
    f'style="display:block;overflow:visible">'
    f'<rect x="{sx(bajo):.1f}" y="{bl-18}" width="{sx(alto)-sx(bajo):.1f}" height="36" '
    f'fill="{AMBER}" opacity=".10"/>'
    f'<line x1="{sx(bajo):.1f}" y1="{bl-18}" x2="{sx(bajo):.1f}" y2="{bl+18}" '
    f'stroke="{AMBER_DIM}" stroke-width="1.4"/>'
    f'<line x1="{sx(alto):.1f}" y1="{bl-18}" x2="{sx(alto):.1f}" y2="{bl+18}" '
    f'stroke="{AMBER_DIM}" stroke-width="1.4"/>'
    f'<line x1="{x0}" y1="{bl}" x2="{x1}" y2="{bl}" stroke="{RULE}"/>{ticks}'
    f'<line x1="{sx(precio):.1f}" y1="{bl-30}" x2="{sx(precio):.1f}" y2="{bl+12}" '
    f'stroke="{AMBER}" stroke-width="2.4"/>'
    f'<polygon points="{sx(precio)-6:.1f},{bl-30} {sx(precio)+6:.1f},{bl-30} '
    f'{sx(precio):.1f},{bl-21}" fill="{AMBER}"/></svg>'
    f'<div class="m" style="font-size:.66rem;color:{MUTE};letter-spacing:.18em;'
    f'text-transform:uppercase;margin-top:10px">{eur(bajo)} &nbsp;—&nbsp; {eur(alto)} · '
    f'nueve de cada diez tasaciones dentro del margen</div>',
    unsafe_allow_html=True)

# ────────────────────────────── diagnóstico ────────────────────────────
st.markdown("<div style='height:46px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eb">Rendimiento del motor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="diag">'
    f'<div><div class="v">{meta["r2_test"]:.3f}</div><div class="k">R² fuera de muestra</div></div>'
    f'<div><div class="v">{eur(meta["mae_test"])}</div><div class="k">Error medio absoluto</div></div>'
    f'<div><div class="v">{eur(meta["rmse_test"])}</div><div class="k">RMSE</div></div>'
    f'<div><div class="v">{num(meta["n_final"])}</div><div class="k">Observaciones</div></div></div>',
    unsafe_allow_html=True)

ETIQ = {"CONSTRUCTEDAREA": "Superficie construida", "LATITUDE": "Latitud",
        "LONGITUDE": "Longitud", "DISTANCE_TO_CITY_CENTER": "Distancia al centro",
        "DISTANCE_TO_METRO": "Distancia al metro", "BATHNUMBER": "Número de baños",
        "ROOMNUMBER": "Número de habitaciones", "HASLIFT": "Ascensor",
        "CADCONSTRUCTIONYEAR": "Año de construcción", "BUILDING_AGE": "Antigüedad",
        "AREA_PER_ROOM": "Superficie por habitación", "CADASTRALQUALITYID": "Calidad catastral",
        "FLOORCLEAN": "Planta", "AMENITIES_COUNT": "Dotaciones", "CITY": "Plaza",
        "FLOOR_RATIO": "Posición en el edificio", "CADDWELLINGCOUNT": "Viviendas del edificio",
        "CADMAXBUILDINGFLOOR": "Altura del edificio", "HASTERRACE": "Terraza",
        "HASPARKINGSPACE": "Plaza de garaje"}

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
b1, b2 = st.columns([1.5, 1], gap="large")
with b1:
    st.markdown('<div class="eb">Factores determinantes</div>', unsafe_allow_html=True)
    it = [(ETIQ.get(x["feature"].replace("num__", "").replace("cat__", "").split("_Madrid")[0],
                    x["feature"].replace("num__", "").replace("_", " ").capitalize()),
           float(x["mean_abs_shap"])) for x in meta["top_shap"][:8]]
    top = max(v for _, v in it)
    rows = "".join(
        f'<text x="0" y="{i*36+15}" fill="{BONE}" font-size="12.5" '
        f'font-family="Jost,sans-serif" font-weight="300">{n}</text>'
        f'<rect x="200" y="{i*36+4}" width="{max(v/top*268,2):.1f}" height="14" '
        f'fill="{AMBER}" opacity="{.22+.78*v/top:.2f}"/>'
        f'<text x="{200+max(v/top*268,2)+10:.1f}" y="{i*36+16}" fill="{MUTE}" font-size="10.5" '
        f'font-family="IBM Plex Mono,monospace">{v:.3f}</text>'
        f'<line x1="0" y1="{i*36+27}" x2="520" y2="{i*36+27}" stroke="{RULE}"/>'
        for i, (n, v) in enumerate(it))
    st.markdown(f'<svg viewBox="0 0 520 {len(it)*36}" width="100%" style="display:block">'
                f'{rows}</svg>', unsafe_allow_html=True)
with b2:
    st.markdown('<div class="eb">Cobertura</div>', unsafe_allow_html=True)
    pm = meta["n_madrid"] / meta["n_final"]
    st.markdown(
        f'<svg viewBox="0 0 400 22" width="100%" style="display:block">'
        f'<rect x="0" y="0" width="{pm*400:.1f}" height="22" fill="{AMBER}"/>'
        f'<rect x="{pm*400:.1f}" y="0" width="{(1-pm)*400:.1f}" height="22" fill="{AMBER_DIM}"/></svg>'
        f'<div style="display:flex;justify-content:space-between;margin-top:11px">'
        f'<div><div class="m" style="font-size:.98rem">{num(meta["n_madrid"])}</div>'
        f'<div class="m" style="font-size:.56rem;letter-spacing:.18em;color:{MUTE};'
        f'text-transform:uppercase;margin-top:3px">Madrid · {pm:.0%}</div></div>'
        f'<div style="text-align:right"><div class="m" style="font-size:.98rem">'
        f'{num(meta["n_barcelona"])}</div><div class="m" style="font-size:.56rem;'
        f'letter-spacing:.18em;color:{MUTE};text-transform:uppercase;margin-top:3px">'
        f'Barcelona · {1-pm:.0%}</div></div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("Metodología"):
        st.markdown(
            f'<div class="prose" style="font-size:.88rem">Ensamblado de árboles con refuerzo '
            f'de gradiente sobre {len(FEATS)} variables de producto, localización y registro '
            f'catastral, con hiperparámetros ajustados por búsqueda bayesiana. La variable '
            f'objetivo se modela en escala logarítmica. La superficie de valor se obtiene '
            f'evaluando el motor sobre una retícula de {Z.shape[0]}×{Z.shape[1]} puntos, '
            f'recalculando la centralidad en cada uno.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="foot"><div>Valora · Motor de valoración automatizada</div>'
    '<div>Estimación orientativa · No sustituye a una tasación homologada</div></div>',
    unsafe_allow_html=True)
