import json, math, base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from PIL import Image

BASE = Path(__file__).resolve().parent
ART = BASE / "artefactos"
if not (ART / "madrid_meta.json").exists() and (ART / "artefactos").exists():
    ART = ART / "artefactos"

st.set_page_config(page_title="VALORA · Motor de valoración residencial",
                   layout="wide", initial_sidebar_state="collapsed")

BG, SURF, LINE = "#F4F2ED", "#FBFAF7", "#D8D2C6"
INK, MUTE, SOFT = "#1A1D1B", "#6E6C63", "#EAE9E1"
ACC, ACC_DIM = "#0E5D4A", "#93B3A7"
BANDA, CREMA = "#0E5D4A", "#E9E4D8"
VERDE, ROJO, AMBAR = "#1A7F4B", "#B5482F", "#9A6B1E"
CIUDADES = ["Madrid", "Barcelona", "Valencia"]
STOPS = [(0.00, (238, 234, 226)), (0.25, (198, 214, 202)), (0.50, (137, 180, 158)),
         (0.75, (52, 130, 104)), (1.00, (11, 77, 61))]
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
def resumen():
    return json.loads((ART / "resumen.json").read_text(encoding="utf-8"))


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
[data-testid="stDecoration"],[data-testid="stSidebarCollapsedControl"]{{display:none!important}}
.stApp{{background:{BG};overflow-x:hidden}}
.block-container{{padding:0 max(1.2rem,calc(50vw - 840px)) 5rem!important;
 max-width:1680px!important}}
.sangre{{margin-left:calc(50% - 50vw);margin-right:calc(50% - 50vw);
 padding:26px max(1.2rem,calc(50vw - 840px))}}
html,body,[class*="css"]{{font-family:'Instrument Sans',sans-serif;color:{INK}}}
.m{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}

.banda{{background:{BANDA};color:#FFFFFF;margin-bottom:30px;
 display:flex;align-items:baseline;justify-content:space-between;
 gap:20px;flex-wrap:wrap}}
.wm{{font-size:1.25rem;font-weight:600;letter-spacing:.36em;text-transform:uppercase;
 color:#FFFFFF}}
.wm span{{color:{CREMA}}}
.banda .sub{{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.18em;
 text-transform:uppercase;color:{CREMA};opacity:.9}}

.lbl{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.2em;
 text-transform:uppercase;color:{MUTE};margin-bottom:10px}}
.eb{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.2em;
 text-transform:uppercase;color:{MUTE};margin:0 0 16px}}

.hero{{display:flex;gap:40px;align-items:center;flex-wrap:wrap}}
.figura{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;
 font-size:3.5rem;font-weight:500;letter-spacing:-.045em;color:{INK};line-height:1}}
.figura em{{font-style:normal;font-size:1.5rem;color:{ACC};margin-left:6px;font-weight:400}}
.banda{{font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:{MUTE};margin-top:10px}}
.cifras{{display:flex;gap:28px;flex-wrap:wrap;margin-top:22px}}
.cifras .v{{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;color:{INK}}}
.cifras .k{{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.14em;
 text-transform:uppercase;color:{MUTE};margin-top:4px}}

.mapa{{position:relative;border:1px solid {LINE};border-radius:4px;overflow:hidden;
 background:{SURF};background-size:100% 100%;background-repeat:no-repeat;
 aspect-ratio:{MAPA_W}/{MAPA_H};animation:fade .55s ease both}}
@keyframes fade{{from{{opacity:0}}to{{opacity:1}}}}
.mapa .ov{{position:absolute;inset:0;width:100%;height:100%}}
.iso{{fill:none;stroke:#FFFFFF;stroke-width:.7;opacity:.5}}
.iso.hi{{stroke:{INK};opacity:.75;stroke-width:1.2}}
.anot{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.08em;
 fill:{INK};paint-order:stroke;stroke:#FFFFFF;stroke-width:2.6px;stroke-linejoin:round}}

.lgd{{display:flex;align-items:center;gap:10px;margin-top:10px;
 font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.12em;
 text-transform:uppercase;color:{MUTE}}}
.lgd .bar{{flex:0 0 150px;height:6px;border-radius:1px;background:linear-gradient(90deg,
 rgb(238,234,226),rgb(198,214,202) 25%,rgb(137,180,158) 50%,rgb(52,130,104) 75%,rgb(11,77,61))}}

.nota{{font-size:.88rem;line-height:1.62;color:{MUTE}}}
.nota b{{color:{INK};font-weight:500}}
.aviso{{border-left:2px solid {ACC};background:{SOFT};padding:12px 15px;
 font-size:.83rem;color:{INK};margin-top:16px;border-radius:0 3px 3px 0}}

.tarj{{border:1px solid {LINE};border-radius:4px;background:{SURF};padding:24px 26px}}
.tarj .t{{font-size:1.5rem;font-weight:600;letter-spacing:-.01em}}
.tarj .d{{font-size:.87rem;color:{MUTE};margin-top:8px;line-height:1.6}}

.kpi{{display:flex;border-top:1px solid {INK};border-bottom:1px solid {LINE}}}
.kpi>div{{flex:1;padding:16px 0 14px;border-right:1px solid {LINE}}}
.kpi>div:last-child{{border-right:none}}
.kpi .v{{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;color:{INK}}}
.kpi .k{{font-family:'IBM Plex Mono',monospace;font-size:.57rem;letter-spacing:.14em;
 text-transform:uppercase;color:{MUTE};margin-top:6px}}

table.tb{{width:100%;border-collapse:collapse;font-size:.88rem}}
table.tb th{{font-family:'IBM Plex Mono',monospace;font-size:.57rem;letter-spacing:.14em;
 text-transform:uppercase;color:{MUTE};font-weight:400;text-align:right;padding-bottom:10px}}
table.tb th:first-child{{text-align:left}}
table.tb td{{padding:11px 0;border-bottom:1px solid {LINE};text-align:right}}
table.tb td:first-child{{text-align:left}}

.pie{{border-top:1px solid {LINE};margin-top:46px;padding-top:18px;display:flex;
 justify-content:space-between;gap:16px;flex-wrap:wrap;
 font-family:'IBM Plex Mono',monospace;font-size:.57rem;letter-spacing:.14em;
 text-transform:uppercase;color:{MUTE}}}

[data-testid="stWidgetLabel"] p{{font-family:'Instrument Sans',sans-serif!important;
 font-size:.76rem!important;color:{MUTE}!important;font-weight:400!important}}
[data-baseweb="select"]>div{{background:{SURF}!important;border-color:{LINE}!important;
 border-radius:3px!important}}
[data-testid="stNumberInput"] input{{background:{SURF}!important;border-color:{LINE}!important;
 border-radius:3px!important}}
[data-testid="stExpander"]{{border:1px solid {LINE};background:{SURF};border-radius:3px}}
[data-testid="stExpander"] summary p{{font-family:'IBM Plex Mono',monospace!important;
 font-size:.6rem!important;letter-spacing:.16em;text-transform:uppercase;color:{MUTE}!important}}

.stTabs [data-baseweb="tab-list"]{{gap:34px;border-bottom:1px solid {LINE};
 background:transparent}}
.stTabs [data-baseweb="tab"]{{background:transparent;padding:14px 0 12px;height:auto;
 font-family:'IBM Plex Mono',monospace;font-size:.63rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE}}}
.stTabs [aria-selected="true"]{{color:{INK}!important}}
.stTabs [data-baseweb="tab-highlight"]{{background:{ACC}}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:30px}}

@media(prefers-reduced-motion:reduce){{.mapa{{animation:none}}}}
@media(min-width:1500px){{ .figura{{font-size:4.2rem}} }}
@media(max-width:1100px){{
 .figura{{font-size:2.9rem}} .kpi{{flex-wrap:wrap}}
 .kpi>div{{flex:1 0 50%;border-bottom:1px solid {LINE}}}
}}
@media(max-width:820px){{
 .block-container{{padding:0 1.1rem 3rem!important}}
 .sangre,.banda{{padding:20px 1.1rem}}
 .figura{{font-size:2.4rem}} .cifras{{gap:18px}}
 .stTabs [data-baseweb="tab-list"]{{gap:16px;overflow-x:auto;scrollbar-width:none}}
 table.tb{{font-size:.8rem}}
}}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────── cabecera ───────────────────────────────
st.markdown('<div class="banda sangre"><div class="wm">Valora<span>.</span></div>'
            '<div class="sub">Motor de valoración residencial · '
            'Madrid · Barcelona · València</div></div>', unsafe_allow_html=True)

st.markdown('<div class="lbl">Activo</div>', unsafe_allow_html=True)
c0, c1, c2, c3, c4 = st.columns([1.1, 1.5, 1, 1, 1])
ciudad = c0.selectbox("Mercado", CIUDADES)
meta, mods, cen = cargar(ciudad)
dis_ok = [d for d, v in meta["distritos"].items() if v["n"] >= 120]
nivel = {d: meta["distritos"][d]["e2026"] or meta["nivel_ciudad_2026"] for d in dis_ok}
dis_ok = sorted(dis_ok, key=lambda d: -nivel[d])
distrito = c1.selectbox("Distrito", dis_ok)
area = c2.number_input("Superficie (m²)", 25, 600, 90, 5)
rooms = c3.number_input("Habitaciones", 0, 15, 3)
baths = c4.number_input("Baños", 0, 10, 2)

c5, c6, c7 = st.columns([1, 2, 1.6])
year = c5.number_input("Año de construcción", 1900, 2018, 1970)
DOT = ["Ascensor", "Terraza", "Plaza de garaje", "Climatización", "Piscina", "Portería"]
dots = c6.multiselect("Dotaciones", DOT, default=["Ascensor"])
d_metro = c7.slider("Distancia al metro (km)", 0.0, 5.0, 0.3, 0.05)

lift, terrace, parking = int(DOT[0] in dots), int(DOT[1] in dots), int(DOT[2] in dots)
air, pool, doorman = int(DOT[3] in dots), int(DOT[4] in dots), int(DOT[5] in dots)
ext = (lift, terrace, parking, air, pool, doorman)

FEAT = meta["features"]
FAC = meta["distritos"][distrito]["factor"]
ESTIMADO = meta["distritos"][distrito]["estimado"]
p99 = int(meta["dominio"]["area_p99"])
sub = cen[cen.distrito == distrito]
lat0 = float(sub.lat.mean()) if len(sub) else meta["centro"]["lat"]
lon0 = float(sub.lon.mean()) if len(sub) else meta["centro"]["lon"]
clat, clon = meta["centro"]["lat"], meta["centro"]["lon"]


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


def predecir(filas, q="q50"):
    return np.exp(mods[q].predict(
        xgb.DMatrix(pd.DataFrame(filas)[FEAT].astype(float)))) * FAC


f0 = [fila(area, lat0, lon0)]
u50 = float(predecir(f0)[0])
u10 = min(float(predecir(f0, "q10")[0]), u50)
u90 = max(float(predecir(f0, "q90")[0]), u50)
p50, p10, p90 = u50 * area, u10 * area, u90 * area

tabV, tabO, tabM, tabD = st.tabs(
    ["Valoración", "Oportunidad", "Mercado", "Metodología"])


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
def png(zb, shape, lo, hi):
    Z = np.frombuffer(zb, dtype=np.float64).reshape(shape)
    t = np.clip((Z - lo) / max(hi - lo, 1e-9), 0, 1) ** .9
    ps = np.array([s[0] for s in STOPS])
    rgb = np.zeros(t.shape + (3,), np.uint8)
    for k in range(3):
        rgb[..., k] = np.clip(np.interp(t, ps, [s[1][k] for s in STOPS]), 0, 255)
    im = Image.fromarray(rgb, "RGB").resize((shape[1] * 7, shape[0] * 7), Image.BICUBIC)
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


# ───────────────────────── pestaña · valoración ───────────────────────
with tabV:
    Z, dlo = campo(ciudad, area, rooms, baths, year, d_metro, ext)
    lo_z, hi_z = float(np.percentile(Z, 2)), float(np.percentile(Z, 98))
    b64 = png(np.ascontiguousarray(Z).tobytes(), Z.shape, lo_z, hi_z)
    W, H = MAPA_W, MAPA_H
    niv = list(np.linspace(lo_z, hi_z, 9))[1:-1]
    paths = isolineas(Z, niv, W, H)
    prox = int(np.argmin([abs(n - p50) for n in niv]))
    px = (lon0 - (clon - dlo)) / (2 * dlo) * W
    py = ((clat + DLA) - lat0) / (2 * DLA) * H
    cx, cy = W / 2, H / 2
    bar = 5 / (2 * dlo * KM * math.cos(math.radians(clat))) * W
    iso = "".join(f'<path class="iso{" hi" if i == prox else ""}" d="{p}"/>'
                  for i, p in enumerate(paths) if p)

    a1, a2 = st.columns([1, 1.25], gap="large")
    with a1:
        st.markdown(
            f'<div class="lbl">Valoración · {distrito}</div>'
            f'<div class="figura">{num(p50)}<em>€</em></div>'
            f'<div class="banda">{eur(p10)} — {eur(p90)} · intervalo 80 %</div>'
            f'<div class="cifras">'
            f'<div><div class="v">{eur(u50)}</div><div class="k">por m²</div></div>'
            f'<div><div class="v">{num(area)} m²</div><div class="k">superficie</div></div>'
            f'<div><div class="v">{2018-year}</div><div class="k">años</div></div>'
            f'<div><div class="v">×{FAC:.2f}</div><div class="k">reindexado</div></div>'
            f'</div>'
            f'<div class="nota" style="margin-top:24px">Nivel publicado del distrito: '
            f'<b>{eur(meta["distritos"][distrito]["e2026"] or meta["nivel_ciudad_2026"])}'
            f'/m²</b>. La estructura procede de {num(meta["n_anuncios"])} anuncios reales; '
            f'el nivel, del índice corriente.</div>'
            + (f'<div class="aviso"><b>Fuera de dominio.</b> Por encima de {p99} m² las '
               f'observaciones son escasas: la estimación es orientativa.</div>'
               if area > p99 else '')
            + ('<div class="aviso">Distrito sin nivel publicado propio; se aplica el factor '
               'de ciudad.</div>' if ESTIMADO else ''),
            unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
<div class="mapa" style="background-image:url(data:image/png;base64,{b64})">
 <svg class="ov" viewBox="0 0 {W} {H}" preserveAspectRatio="none">{iso}</svg>
 <svg class="ov" viewBox="0 0 {W} {H}">
  <line x1="{cx-7}" y1="{cy}" x2="{cx+7}" y2="{cy}" stroke="{INK}" stroke-width="1.3"/>
  <line x1="{cx}" y1="{cy-7}" x2="{cx}" y2="{cy+7}" stroke="{INK}" stroke-width="1.3"/>
  <text class="anot" x="{cx}" y="{cy+20}" text-anchor="middle">CENTRO</text>
  <g transform="translate(18,{H-18})">
   <line x1="0" y1="0" x2="{bar:.0f}" y2="0" stroke="{INK}" stroke-width="1.4"/>
   <line x1="0" y1="-4" x2="0" y2="4" stroke="{INK}" stroke-width="1.4"/>
   <line x1="{bar:.0f}" y1="-4" x2="{bar:.0f}" y2="4" stroke="{INK}" stroke-width="1.4"/>
   <text class="anot" x="{bar/2:.0f}" y="-8" text-anchor="middle">5 km</text></g>
  <g transform="translate({W-22},20)">
   <polygon points="0,-8 4,6 0,3 -4,6" fill="{INK}"/>
   <text class="anot" x="0" y="19" text-anchor="middle">N</text></g>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="{INK}" stroke-width="1.3"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="{INK}"/>
 </svg>
</div>
<div class="lgd"><span class="bar"></span><span>{compact(lo_z)} €</span>
 <span style="flex:1"></span><span>{compact(hi_z)} €</span>
 <span style="opacity:.75">· el mismo activo en cada punto</span></div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")

    def curva(xs, ys, xn, yn, unidad, ident):
        Wc, Hc, L, R, T, B = 520, 180, 12, 508, 24, 42
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        pad = (ymax - ymin) * .18 or 1
        ymin, ymax = ymin - pad, ymax + pad
        fx = lambda v: L + (v - xmin) / max(xmax - xmin, 1e-9) * (R - L)
        fy = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (Hc - B - T)
        pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
        ex = lambda v: num(v, 1).rstrip("0").rstrip(",") if unidad == " km" else num(v)
        g = "".join(f'<line x1="{L}" y1="{T+(Hc-B-T)*i/3:.1f}" x2="{R}" '
                    f'y2="{T+(Hc-B-T)*i/3:.1f}" stroke="{LINE}"/>' for i in range(1, 3))
        return (f'<svg viewBox="0 0 {Wc} {Hc}" width="100%" style="display:block;'
                f'overflow:visible">{g}'
                f'<polygon points="{L},{Hc-B} {pts} {R},{Hc-B}" fill="{ACC}" opacity=".07"/>'
                f'<polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="1.8" '
                f'stroke-linejoin="round"/>'
                f'<line x1="{fx(xn):.1f}" y1="{T-6}" x2="{fx(xn):.1f}" y2="{Hc-B}" '
                f'stroke="{ACC_DIM}" stroke-dasharray="2 4"/>'
                f'<circle cx="{fx(xn):.1f}" cy="{fy(yn):.1f}" r="4.2" fill="{ACC}" '
                f'stroke="#FFFFFF" stroke-width="1.8"/>'
                f'<line x1="{L}" y1="{Hc-B}" x2="{R}" y2="{Hc-B}" stroke="{LINE}"/>'
                f'<g font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{MUTE}">'
                f'<text x="{L}" y="{Hc-B+17}">{ex(xmin)}{unidad}</text>'
                f'<text x="{R}" y="{Hc-B+17}" text-anchor="end">{ex(xmax)}{unidad}</text>'
                f'<text x="{L}" y="{T-10}">{compact(max(ys))} €</text></g></svg>')

    xa = np.linspace(max(25, area - 55), min(600, area + 55), 22)
    ya = predecir([fila(float(a), lat0, lon0) for a in xa]) * xa
    with g1:
        st.markdown('<div class="eb">Elasticidad · superficie</div>', unsafe_allow_html=True)
        st.markdown(curva(list(xa), list(ya), area, p50, " m²", "a"), unsafe_allow_html=True)
        st.markdown(f'<div class="nota" style="margin-top:8px">Cada m² adicional en torno a '
                    f'los {area} m² añade '
                    f'<b>{eur((float(ya[-1])-float(ya[0]))/(xa[-1]-xa[0]))}</b>.</div>',
                    unsafe_allow_html=True)
    xd = np.linspace(0.05, 3.0, 22)
    yd = predecir([fila(area, lat0, lon0, dm=float(v)) for v in xd]) * area
    with g2:
        st.markdown('<div class="eb">Gradiente · acceso al metro</div>', unsafe_allow_html=True)
        st.markdown(curva(list(xd), list(yd), min(d_metro, 3.0), p50, " km", "d"),
                    unsafe_allow_html=True)
        st.markdown(f'<div class="nota" style="margin-top:8px">Pasar de 50 m a 3 km de una '
                    f'boca de metro resta un '
                    f'<b>{(float(yd[0])-float(yd[-1]))/float(yd[0])*100:.1f} %</b>.</div>',
                    unsafe_allow_html=True)

# ───────────────────────── pestaña · oportunidad ──────────────────────
with tabO:
    o1, o2 = st.columns([1, 1.7], gap="large")
    with o1:
        st.markdown('<div class="lbl">Precio pedido</div>', unsafe_allow_html=True)
        pedido = st.number_input("Precio del vendedor (€)", 0, 20_000_000,
                                 int(round(p50 / 1000) * 1000), 5000,
                                 label_visibility="collapsed")
        st.markdown(f'<div class="nota" style="margin-top:12px">Se compara con el intervalo '
                    f'del 80 % de la valoración: <b>{eur(p10)}</b> a <b>{eur(p90)}</b>.</div>',
                    unsafe_allow_html=True)
    with o2:
        if pedido <= 0:
            ver, col, det = "Introduce un precio", MUTE, ""
        elif pedido < p10:
            ver, col = "Oportunidad", VERDE
            det = (f"Se pide {eur(p10-pedido)} por debajo del extremo inferior. Solo un 10 % "
                   f"de las viviendas comparables se ofrecen a este precio o menos.")
        elif pedido > p90:
            ver, col = "Sobrevalorado", ROJO
            det = (f"Se pide {eur(pedido-p90)} por encima del extremo superior. Solo un 10 % "
                   f"de las comparables alcanzan este precio.")
        else:
            ver, col = "Precio de mercado", AMBAR
            det = (f"Dentro del intervalo, en el percentil "
                   f"{(pedido-p10)/max(p90-p10,1)*100:.0f} de la banda. Diferencia frente a "
                   f"la estimación central: {eur(pedido-p50)}.")
        st.markdown(f'<div class="tarj"><div class="t" style="color:{col}">{ver}</div>'
                    f'<div class="d">{det}</div></div>', unsafe_allow_html=True)

    if pedido > 0:
        amax = max(p90, pedido) * 1.1
        sx_ = lambda v: 6 + v / amax * 988
        st.markdown(
            f'<svg viewBox="0 0 1000 86" width="100%" preserveAspectRatio="none" '
            f'style="display:block;overflow:visible;margin-top:26px">'
            f'<rect x="{sx_(p10):.1f}" y="32" width="{sx_(p90)-sx_(p10):.1f}" height="24" '
            f'fill="{ACC}" opacity=".11"/>'
            f'<line x1="{sx_(p10):.1f}" y1="32" x2="{sx_(p10):.1f}" y2="56" '
            f'stroke="{ACC_DIM}" stroke-width="1.3"/>'
            f'<line x1="{sx_(p90):.1f}" y1="32" x2="{sx_(p90):.1f}" y2="56" '
            f'stroke="{ACC_DIM}" stroke-width="1.3"/>'
            f'<line x1="6" y1="44" x2="994" y2="44" stroke="{LINE}"/>'
            f'<line x1="{sx_(p50):.1f}" y1="26" x2="{sx_(p50):.1f}" y2="62" '
            f'stroke="{ACC}" stroke-width="2"/>'
            f'<text x="{sx_(p50):.1f}" y="18" fill="{ACC}" font-size="10.5" '
            f'text-anchor="middle" font-family="IBM Plex Mono,monospace">estimación</text>'
            f'<circle cx="{sx_(pedido):.1f}" cy="44" r="7" fill="{col}" stroke="#FFFFFF" '
            f'stroke-width="2"/>'
            f'<text x="{sx_(pedido):.1f}" y="79" fill="{col}" font-size="10.5" '
            f'text-anchor="middle" font-family="IBM Plex Mono,monospace">'
            f'pedido {compact(pedido)} €</text></svg>', unsafe_allow_html=True)

# ───────────────────────── pestaña · mercado ──────────────────────────
with tabM:
    R = resumen()
    st.markdown('<div class="eb">Rendimiento · validación por bloques espaciales</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi">'
        f'<div><div class="v">{meta["r2_bloques"]:.3f}</div><div class="k">R² · {ciudad}</div></div>'
        f'<div><div class="v">{eur(meta["mae"])}</div><div class="k">Error medio</div></div>'
        f'<div><div class="v">{meta["cobertura_intervalo"]*100:.1f} %</div>'
        f'<div class="k">Cobertura (teórica 80 %)</div></div>'
        f'<div><div class="v">{num(meta["n_anuncios"])}</div><div class="k">Anuncios</div></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns([1.15, 1], gap="large")
    with m1:
        st.markdown('<div class="eb">Los tres mercados</div>', unsafe_allow_html=True)
        fl = "".join(
            f'<tr><td>{c}</td><td class="m">{R[c]["r2_bloques"]:.3f}</td>'
            f'<td class="m">{eur(R[c]["mae"])}</td>'
            f'<td class="m">{R[c]["cobertura_intervalo"]*100:.0f} %</td>'
            f'<td class="m" style="color:{ACC}">×{R[c]["factor_ciudad"]:.2f}</td></tr>'
            for c in CIUDADES)
        st.markdown(
            f'<table class="tb"><tr><th>Mercado</th><th>R²</th><th>Error medio</th>'
            f'<th>Cobertura</th><th>Reindexado</th></tr>{fl}</table>'
            f'<div class="nota" style="margin-top:14px">València, con un nivel de precio '
            f'menos de la mitad que Madrid, funciona como prueba de generalización: la misma '
            f'arquitectura cede <b>{R["Madrid"]["r2_bloques"]-R["Valencia"]["r2_bloques"]:.3f}'
            f'</b> de R² al trasladarse a un mercado estructuralmente distinto.</div>',
            unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="eb">Revalorización por distrito · 2018 a 2026</div>',
                    unsafe_allow_html=True)
        dd = sorted([(d, v) for d, v in meta["distritos"].items()
                     if v["n"] >= 120 and not v["estimado"]],
                    key=lambda x: -x[1]["factor"])[:10]
        if dd:
            mx = max(v["factor"] for _, v in dd)
            rows = "".join(
                f'<text x="0" y="{i*28+13}" fill="{INK}" font-size="12" '
                f'font-family="Instrument Sans,sans-serif">{d}</text>'
                f'<rect x="190" y="{i*28+3}" width="{max(v["factor"]/mx*205,2):.1f}" '
                f'height="12" fill="{ACC}" opacity="{.3+.7*v["factor"]/mx:.2f}" rx="1"/>'
                f'<text x="{190+max(v["factor"]/mx*205,2)+9:.1f}" y="{i*28+13}" fill="{MUTE}" '
                f'font-size="10.5" font-family="IBM Plex Mono,monospace">'
                f'×{v["factor"]:.2f}</text>'
                f'<line x1="0" y1="{i*28+21}" x2="460" y2="{i*28+21}" stroke="{LINE}"/>'
                for i, (d, v) in enumerate(dd))
            st.markdown(f'<svg viewBox="0 0 460 {len(dd)*28}" width="100%" '
                        f'style="display:block">{rows}</svg>', unsafe_allow_html=True)
        st.markdown('<div class="nota" style="margin-top:12px">Un factor único de ciudad '
                    'ignoraría esta dispersión.</div>', unsafe_allow_html=True)

# ───────────────────────── pestaña · metodología ──────────────────────
with tabD:
    d1, d2 = st.columns(2, gap="large")
    bloques = [
        ("Datos", f"{num(meta['n_anuncios'])} anuncios de {ciudad} "
         f"({num(meta['n_viviendas'])} viviendas únicas) de los cuatro trimestres de 2018, "
         f"del conjunto abierto <i>idealista18</i> (Rey-Blanco, Arbués, López y Páez, "
         f"<i>Environment and Planning B</i>, 2024; licencia ODbL), enriquecido con "
         f"información catastral."),
        ("Modelo", "Ensamblado de árboles con refuerzo de gradiente sobre el logaritmo del "
         "precio por metro cuadrado, no sobre el precio total: así el valor sigue creciendo "
         "con la superficie más allá del rango con datos abundantes, en lugar de saturarse. "
         "Se imponen restricciones de monotonía en las variables cuyo signo no admite duda."),
        ("Incertidumbre", f"Tres modelos de regresión cuantílica (percentiles 10, 50 y 90) "
         f"dan un intervalo propio a cada inmueble, estrecho en viviendas corrientes y ancho "
         f"en las atípicas. La cobertura real medida es del "
         f"{meta['cobertura_intervalo']*100:.1f} % frente al 80 % teórico."),
        ("Validación", "Cruzada por bloques espaciales de unos 100 metros, que agrupan "
         "también los anuncios repetidos del mismo inmueble entre trimestres y evitan así "
         "la fuga de información entre entrenamiento y contraste."),
        ("Actualización", "La estructura relativa se aprende de 2018; el nivel de precios "
         "procede del índice publicado por distrito en agosto de 2026. La correlación de "
         "rangos entre el mapa de valor de 2018 y el actual es de 0,975 en Madrid: la "
         "geografía del valor apenas ha cambiado, solo su nivel."),
        ("Límites", f"Si desde 2018 ha cambiado <i>qué</i> valora el comprador —y algo ha "
         f"cambiado, como el peso del espacio exterior— el modelo no lo recoge. Por encima "
         f"del percentil 99 de superficie ({p99} m² en {ciudad}) la estimación se marca como "
         f"orientativa. No sustituye a una tasación homologada."),
    ]
    for i, (t, c) in enumerate(bloques):
        with (d1 if i % 2 == 0 else d2):
            st.markdown(f'<div style="margin-bottom:28px"><div class="eb">{t}</div>'
                        f'<div class="nota">{c}</div></div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="pie"><div>Valora · Motor de valoración residencial</div>'
    f'<div>Estructura idealista18 2018 · Nivel agosto 2026</div>'
    f'<div>No sustituye a una tasación homologada</div></div>', unsafe_allow_html=True)
