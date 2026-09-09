import json, math, base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from PIL import Image

BASE = Path(__file__).resolve().parent
ART = BASE / "artefactos" / "artefactos"

st.set_page_config(page_title="VALORA · Motor de valoración residencial",
                   layout="wide", initial_sidebar_state="collapsed")

VOID, DEEP, RULE = "#0C1216", "#151D22", "#28353C"
BONE, MUTE = "#F0F2EE", "#8FA0A6"
AMBER, AMBER_DIM = "#F0A93F", "#A8762F"
VERDE, ROJO = "#4CB782", "#E06A5A"
CIUDADES = ["Madrid", "Barcelona", "Valencia"]
STOPS = [(0.00, (26, 52, 66)), (0.25, (23, 86, 88)), (0.50, (36, 132, 106)),
         (0.72, (163, 142, 73)), (0.88, (240, 169, 63)), (1.00, (250, 232, 200))]
MAPA_W, MAPA_H, DLA, NY = 660, 450, 0.075, 46
NX = int(round(NY * MAPA_W / MAPA_H))


# ─────────────────────────────── carga ────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar(ciudad):
    meta = json.loads((ART / f"{ciudad.lower()}_meta.json").read_text(encoding="utf-8"))
    mods = {q: xgb.Booster(model_file=str(ART / f"{ciudad.lower()}_{q}.ubj"))
            for q in ("q10", "q50", "q90")}
    cen = pd.DataFrame(meta["centroides"])
    return meta, mods, cen


@st.cache_data(show_spinner=False)
def resumen():
    return json.loads((ART / "resumen.json").read_text(encoding="utf-8"))


# ─────────────────────────────── formato ──────────────────────────────
def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def eur(x, dec=0):
    return f"{num(x, dec)} €"


def compact(v):
    return f"{num(v/1_000_000, 1)}M" if v >= 1_000_000 else f"{num(v/1000)}k"


# ─────────────────────────────── estilos ──────────────────────────────
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
 border-bottom:1px solid {RULE};padding-bottom:14px;margin-bottom:24px}}
.wm{{font-family:'Bodoni Moda',serif;font-size:1.7rem;letter-spacing:.42em;
 text-transform:uppercase}}
.mast .sub{{font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.24em;
 text-transform:uppercase;color:{MUTE}}}
.eb{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.26em;
 text-transform:uppercase;color:{MUTE};display:flex;align-items:center;gap:14px;margin:0 0 14px}}
.eb::after{{content:"";flex:1;height:1px;background:{RULE}}}
.grp{{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.22em;
 text-transform:uppercase;color:{AMBER_DIM};padding-bottom:8px;
 border-bottom:1px solid {RULE};margin-bottom:14px}}
.stage{{display:flex;gap:32px;align-items:stretch}}
.scene{{position:relative;flex:1 1 58%;min-width:0;border:1px solid {RULE};overflow:hidden;
 background-color:{DEEP};background-size:100% 100%;background-repeat:no-repeat;
 aspect-ratio:{MAPA_W}/{MAPA_H};animation:unveil 1.1s cubic-bezier(.16,.84,.34,1) both}}
@keyframes unveil{{from{{clip-path:inset(0 100% 0 0)}}to{{clip-path:inset(0 0 0 0)}}}}
.scene .ov{{position:absolute;inset:0;width:100%;height:100%}}
.iso{{fill:none;stroke:#0C1216;stroke-width:.7;opacity:.28}}
.iso.hi{{stroke:{BONE};opacity:.85;stroke-width:1.3}}
.grat{{stroke:{BONE};opacity:.09;stroke-width:.5}}
.anot{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.1em;fill:{BONE}}}
.anot.dk{{fill:#0C1216}}
.plate{{flex:0 0 38%;display:flex;flex-direction:column;justify-content:center}}
.tag{{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.26em;
 text-transform:uppercase;color:{AMBER};margin-bottom:12px}}
.val{{font-family:'Bodoni Moda',serif;font-size:4.1rem;line-height:.94;
 display:flex;align-items:baseline;gap:8px}}
.val em{{font-style:normal;font-size:1.6rem;color:{AMBER}}}
.rango{{font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:{MUTE};margin-top:12px;
 letter-spacing:.06em}}
.sq{{margin-top:18px;display:flex;gap:20px;flex-wrap:wrap}}
.sq div{{border-left:1px solid {AMBER_DIM};padding-left:10px}}
.sq .v{{font-family:'IBM Plex Mono',monospace;font-size:.92rem}}
.sq .k{{font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE};margin-top:3px}}
@keyframes halo{{0%,100%{{r:7;opacity:.85}}50%{{r:17;opacity:0}}}}
.pin-halo{{animation:halo 3.4s ease-out infinite}}
.lgd{{display:flex;align-items:center;gap:11px;margin-top:10px;
 font-family:'IBM Plex Mono',monospace;font-size:.56rem;letter-spacing:.16em;
 text-transform:uppercase;color:{MUTE}}}
.lgd .bar{{flex:0 0 170px;height:6px;background:linear-gradient(90deg,
 rgb(26,52,66),rgb(23,86,88) 25%,rgb(36,132,106) 50%,rgb(163,142,73) 72%,
 rgb(240,169,63) 88%,rgb(250,232,200))}}
.prose{{font-family:'Bodoni Moda',serif;font-size:.96rem;line-height:1.64;color:{MUTE}}}
.prose b{{color:{BONE};font-weight:500}}
.aviso{{border-left:2px solid {AMBER};background:rgba(240,169,63,.07);padding:11px 14px;
 font-family:'Jost',sans-serif;font-size:.82rem;color:{BONE};margin-top:14px}}
.veredicto{{border:1px solid {RULE};background:{DEEP};padding:20px 22px}}
.veredicto .t{{font-family:'Bodoni Moda',serif;font-size:1.7rem;line-height:1.1}}
.veredicto .d{{font-family:'Jost',sans-serif;font-size:.86rem;color:{MUTE};margin-top:8px}}
.diag{{display:flex;border-top:1px solid {AMBER_DIM};border-bottom:1px solid {RULE}}}
.diag>div{{flex:1;padding:15px 0 13px;border-right:1px solid {RULE}}}
.diag>div:last-child{{border-right:none}}
.diag .v{{font-family:'IBM Plex Mono',monospace;font-size:1.3rem}}
.diag .k{{font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.18em;
 text-transform:uppercase;color:{MUTE};margin-top:5px}}
.foot{{border-top:1px solid {RULE};margin-top:42px;padding-top:16px;display:flex;
 justify-content:space-between;gap:18px;flex-wrap:wrap;
 font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.2em;
 text-transform:uppercase;color:{MUTE}}}
@media(prefers-reduced-motion:reduce){{.scene{{animation:none}}.pin-halo{{animation:none}}}}
[data-testid="stWidgetLabel"] p{{font-family:'Jost',sans-serif!important;
 font-size:.78rem!important;color:{MUTE}!important;font-weight:300!important}}
[data-baseweb="select"]>div{{background:{DEEP}!important;border-color:{RULE}!important;
 border-radius:2px!important}}
[data-testid="stNumberInput"] input{{background:{DEEP}!important;color:{BONE}!important;
 border-color:{RULE}!important}}
[data-testid="stCheckbox"] p{{color:{MUTE}!important;font-size:.76rem!important}}
[data-testid="stExpander"]{{border:1px solid {RULE};background:{DEEP};border-radius:2px}}
[data-testid="stExpander"] summary p{{font-family:'IBM Plex Mono',monospace!important;
 font-size:.6rem!important;letter-spacing:.2em;text-transform:uppercase;color:{MUTE}!important}}
@media(max-width:1000px){{
 .block-container{{padding:1.3rem 1.1rem 3rem!important}}
 .stage{{flex-direction:column;gap:20px}} .plate{{flex:none}}
 .val{{font-size:2.8rem}} .diag{{flex-wrap:wrap}}
 .diag>div{{flex:1 0 50%;border-bottom:1px solid {RULE}}}
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mast"><div class="wm">Valora</div>'
            '<div class="sub">Motor de valoración residencial · Madrid · Barcelona · València</div>'
            '</div>', unsafe_allow_html=True)

# ─────────────────────────────── controles ────────────────────────────
st.markdown('<div class="eb">Parámetros del activo</div>', unsafe_allow_html=True)
k1, k2, k3 = st.columns([1, 1, 1.05], gap="large")

with k1:
    st.markdown('<div class="grp">Plaza</div>', unsafe_allow_html=True)
    ciudad = st.selectbox("Mercado", CIUDADES)
    meta, mods, cen = cargar(ciudad)
    dis_ok = sorted([d for d, v in meta["distritos"].items() if v["n"] >= 120])
    nivel = {d: meta["distritos"][d]["e2026"] or meta["nivel_ciudad_2026"] for d in dis_ok}
    dis_ok = sorted(dis_ok, key=lambda d: -nivel[d])
    distrito = st.selectbox("Distrito", dis_ok)
    d_metro = st.slider("Distancia al metro (km)", 0.0, 5.0, 0.3, 0.05)

with k2:
    st.markdown('<div class="grp">Activo</div>', unsafe_allow_html=True)
    p99 = int(meta["dominio"]["area_p99"])
    area = st.slider("Superficie construida (m²)", 25, 600, 90)
    year = st.slider("Año de construcción", 1900, 2018, 1970)
    r1, r2 = st.columns(2)
    rooms = r1.number_input("Habitaciones", 0, 15, 3)
    baths = r2.number_input("Baños", 0, 10, 2)

with k3:
    st.markdown('<div class="grp">Dotaciones</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    lift = e1.checkbox("Ascensor", value=True)
    terrace = e2.checkbox("Terraza")
    parking = e1.checkbox("Plaza de garaje")
    air = e2.checkbox("Climatización")
    pool = e1.checkbox("Piscina")
    doorman = e2.checkbox("Portería")

FEAT = meta["features"]
FAC = meta["distritos"][distrito]["factor"]
ESTIMADO = meta["distritos"][distrito]["estimado"]

# centroide del distrito elegido
sub = cen[cen.distrito == distrito]
lat0 = float(sub.lat.mean()) if len(sub) else meta["centro"]["lat"]
lon0 = float(sub.lon.mean()) if len(sub) else meta["centro"]["lon"]
clat, clon = meta["centro"]["lat"], meta["centro"]["lon"]
KM = 111.0


def dist_km(la, lo):
    return np.sqrt(((la - clat) * KM) ** 2 +
                   ((lo - clon) * KM * math.cos(math.radians(clat))) ** 2)


def fila(a, la, lo, dm=None):
    r = dict(meta["medianas"])
    r.update({"CONSTRUCTEDAREA": a, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
              "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
              "LATITUDE": la, "LONGITUDE": lo,
              "DISTANCE_TO_CITY_CENTER": float(dist_km(la, lo)),
              "DISTANCE_TO_METRO": d_metro if dm is None else dm,
              "HASLIFT": int(lift), "HASTERRACE": int(terrace),
              "HASPARKINGSPACE": int(parking), "HASAIRCONDITIONING": int(air),
              "HASSWIMMINGPOOL": int(pool), "HASDOORMAN": int(doorman),
              "PERIOD": 201812})
    return {c: r.get(c, 0.0) for c in FEAT}


def predecir(filas, q="q50", factor=None):
    d = xgb.DMatrix(pd.DataFrame(filas)[FEAT].astype(float))
    return np.exp(mods[q].predict(d)) * (FAC if factor is None else factor)


# ─────────────────────────────── valoración ───────────────────────────
f0 = [fila(area, lat0, lon0)]
u50 = float(predecir(f0, "q50")[0])
u10 = float(predecir(f0, "q10")[0])
u90 = float(predecir(f0, "q90")[0])
u10, u90 = min(u10, u50), max(u90, u50)
p50, p10, p90 = u50 * area, u10 * area, u90 * area
fuera = area > p99


# ───────────────────── superficie de valor de la ciudad ───────────────
@st.cache_data(show_spinner=False)
def campo(ciudad, area, rooms, baths, year, d_metro, ext, distrito):
    mt, mo, ce = cargar(ciudad)
    dlo = DLA * (MAPA_W / MAPA_H) / math.cos(math.radians(mt["centro"]["lat"]))
    la0, lo0 = mt["centro"]["lat"], mt["centro"]["lon"]
    las = np.linspace(la0 + DLA, la0 - DLA, NY)
    los = np.linspace(lo0 - dlo, lo0 + dlo, NX)
    LO, LA = np.meshgrid(los, las)
    base = dict(mt["medianas"])
    base.update({"CONSTRUCTEDAREA": area, "ROOMNUMBER": rooms, "BATHNUMBER": baths,
                 "CADCONSTRUCTIONYEAR": year, "CONSTRUCTIONYEAR": year,
                 "DISTANCE_TO_METRO": d_metro, "HASLIFT": ext[0], "HASTERRACE": ext[1],
                 "HASPARKINGSPACE": ext[2], "HASAIRCONDITIONING": ext[3],
                 "HASSWIMMINGPOOL": ext[4], "HASDOORMAN": ext[5], "PERIOD": 201812})
    df = pd.DataFrame([{c: base.get(c, 0.0) for c in mt["features"]}] * LA.size)
    df["LATITUDE"] = LA.ravel(); df["LONGITUDE"] = LO.ravel()
    df["DISTANCE_TO_CITY_CENTER"] = np.sqrt(
        ((LA - la0) * KM) ** 2 +
        ((LO - lo0) * KM * math.cos(math.radians(la0))) ** 2).ravel()
    u18 = np.exp(mo["q50"].predict(xgb.DMatrix(df[mt["features"]].astype(float))))
    # factor por celda: centroide de barrio más cercano
    cl, co = ce.lat.values, ce.lon.values
    fac_b = np.array([mt["distritos"].get(d, {}).get("factor", mt["factor_ciudad"])
                      for d in ce.distrito.values])
    dd = ((LA.ravel()[:, None] - cl[None, :]) ** 2 +
          (LO.ravel()[:, None] - co[None, :]) ** 2)
    return (u18 * fac_b[dd.argmin(1)] * area).reshape(NY, NX).astype(np.float64), dlo


ext = (int(lift), int(terrace), int(parking), int(air), int(pool), int(doorman))
Z, dlo = campo(ciudad, area, rooms, baths, year, d_metro, ext, distrito)
lo_z, hi_z = float(np.percentile(Z, 2)), float(np.percentile(Z, 98))


@st.cache_data(show_spinner=False)
def png(zb, shape, lo, hi):
    Zz = np.frombuffer(zb, dtype=np.float64).reshape(shape)
    t = np.clip((Zz - lo) / max(hi - lo, 1e-9), 0, 1) ** .85
    ps = np.array([s[0] for s in STOPS])
    rgb = np.zeros(t.shape + (3,), np.uint8)
    for k in range(3):
        rgb[..., k] = np.clip(np.interp(t, ps, [s[1][k] for s in STOPS]), 0, 255)
    im = Image.fromarray(rgb, "RGB").resize((shape[1] * 7, shape[0] * 7), Image.BICUBIC)
    b = BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


b64 = png(np.ascontiguousarray(Z).tobytes(), Z.shape, lo_z, hi_z)


def isolineas(Zz, niveles, W, H):
    ny, nx = Zz.shape
    sx, sy = W / (nx - 1), H / (ny - 1)
    out = []
    for lv in niveles:
        G = Zz > lv; seg = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b_, c_, e = Zz[j, i], Zz[j, i+1], Zz[j+1, i+1], Zz[j+1, i]
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


W, H = MAPA_W, MAPA_H
niv = list(np.linspace(lo_z, hi_z, 10))[1:-1]
paths = isolineas(Z, niv, W, H)
prox = int(np.argmin([abs(n - p50) for n in niv]))
px = (lon0 - (clon - dlo)) / (2 * dlo) * W
py = ((clat + DLA) - lat0) / (2 * DLA) * H
cx, cy = W / 2, H / 2
KM_W = 2 * dlo * KM * math.cos(math.radians(clat))
bar = 5 / KM_W * W

grat = "".join(f'<line class="grat" x1="{W*i/6:.0f}" y1="0" x2="{W*i/6:.0f}" y2="{H}"/>'
               for i in range(1, 6)) + \
       "".join(f'<line class="grat" x1="0" y1="{H*i/4:.0f}" x2="{W}" y2="{H*i/4:.0f}"/>'
               for i in range(1, 4))
iso = "".join(f'<path class="iso{" hi" if i == prox else ""}" d="{p}"/>'
              for i, p in enumerate(paths) if p)

st.markdown(f"""
<div class="stage">
  <div class="plate">
    <div class="tag">Valoración · {distrito} · {ciudad}</div>
    <div class="val">{num(p50)}<em>€</em></div>
    <div class="rango">{eur(p10)} &nbsp;—&nbsp; {eur(p90)}
      <span style="opacity:.6">· intervalo 80 %</span></div>
    <div class="sq">
      <div><div class="v">{eur(u50)}</div><div class="k">por m²</div></div>
      <div><div class="v">{num(area)} m²</div><div class="k">superficie</div></div>
      <div><div class="v">{2018-year}</div><div class="k">años</div></div>
      <div><div class="v">×{FAC:.2f}</div><div class="k">reindex 2026</div></div>
    </div>
    <div class="prose" style="margin-top:18px">Nivel publicado del distrito en agosto de
      2026: <b>{eur(meta['distritos'][distrito]['e2026'] or meta['nivel_ciudad_2026'])}/m²</b>.
      La estructura relativa procede de {num(meta['n_anuncios'])} anuncios reales;
      el nivel, del índice corriente.</div>
    {'<div class="aviso"><b>Fuera de dominio.</b> Por encima de ' + str(p99) +
     ' m² (percentil 99) hay muy pocas observaciones: la estimación es orientativa.</div>'
     if fuera else ''}
    {'<div class="aviso">Este distrito no tiene nivel publicado individual; se aplica el '
     'factor de ciudad.</div>' if ESTIMADO else ''}
  </div>
  <div class="scene" style="background-image:url(data:image/png;base64,{b64})">
    <svg class="ov" viewBox="0 0 {W} {H}" preserveAspectRatio="none">{grat}{iso}</svg>
    <svg class="ov" viewBox="0 0 {W} {H}">
      <g opacity=".9">
        <line x1="{cx-7}" y1="{cy}" x2="{cx+7}" y2="{cy}" stroke="#0C1216" stroke-width="1.4"/>
        <line x1="{cx}" y1="{cy-7}" x2="{cx}" y2="{cy+7}" stroke="#0C1216" stroke-width="1.4"/>
        <text class="anot dk" x="{cx}" y="{cy+20}" text-anchor="middle">CENTRO</text></g>
      <g transform="translate(18,{H-20})">
        <line x1="0" y1="0" x2="{bar:.0f}" y2="0" stroke="{BONE}" stroke-width="1.6"/>
        <line x1="0" y1="-4" x2="0" y2="4" stroke="{BONE}" stroke-width="1.6"/>
        <line x1="{bar:.0f}" y1="-4" x2="{bar:.0f}" y2="4" stroke="{BONE}" stroke-width="1.6"/>
        <text class="anot" x="{bar/2:.0f}" y="-9" text-anchor="middle">5 km</text></g>
      <g transform="translate({W-24},22)">
        <polygon points="0,-9 4.5,7 0,3.5 -4.5,7" fill="{BONE}" opacity=".85"/>
        <text class="anot" x="0" y="20" text-anchor="middle" opacity=".85">N</text></g>
      <circle class="pin-halo" cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none"
              stroke="{BONE}" stroke-width="1.6"/>
      <circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{AMBER}" stroke="#0C1216"
              stroke-width="1.6"/>
    </svg>
  </div>
</div>
<div class="lgd"><span class="bar"></span><span>{compact(lo_z)} €</span>
 <span style="flex:1"></span><span>{compact(hi_z)} €</span>
 <span style="opacity:.7">· el mismo activo en cada punto de {ciudad}</span></div>
""", unsafe_allow_html=True)

# ────────────────────── detector de oportunidades ─────────────────────
st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eb">Detector de oportunidades</div>', unsafe_allow_html=True)
o1, o2 = st.columns([1, 1.6], gap="large")
with o1:
    pedido = st.number_input("Precio pedido por el vendedor (€)", 0,
                             20_000_000, int(round(p50 / 1000) * 1000), 5000)
with o2:
    if pedido <= 0:
        veredicto, color, det = "Introduce un precio", MUTE, ""
    elif pedido < p10:
        veredicto, color = "Oportunidad", VERDE
        det = (f"Se pide {eur(p10-pedido)} por debajo del extremo inferior del intervalo. "
               f"Solo un 10 % de viviendas comparables se ofrecen a este precio o menos.")
    elif pedido > p90:
        veredicto, color = "Sobrevalorado", ROJO
        det = (f"Se pide {eur(pedido-p90)} por encima del extremo superior. "
               f"Solo un 10 % de comparables alcanzan este precio.")
    else:
        veredicto, color = "Precio de mercado", AMBER
        pos = (pedido - p10) / max(p90 - p10, 1) * 100
        det = (f"Dentro del intervalo de confianza, en el percentil {pos:.0f} de la banda. "
               f"Diferencia frente a la estimación central: {eur(pedido-p50)}.")
    st.markdown(f'<div class="veredicto"><div class="t" style="color:{color}">{veredicto}'
                f'</div><div class="d">{det}</div></div>', unsafe_allow_html=True)

if pedido > 0:
    SW, SH = 1000, 84
    amax = max(p90, pedido) * 1.1
    sx_ = lambda v: 6 + v / amax * 988
    st.markdown(
        f'<svg viewBox="0 0 {SW} {SH}" width="100%" preserveAspectRatio="none" '
        f'style="display:block;overflow:visible;margin-top:16px">'
        f'<rect x="{sx_(p10):.1f}" y="30" width="{sx_(p90)-sx_(p10):.1f}" height="26" '
        f'fill="{AMBER}" opacity=".13"/>'
        f'<line x1="{sx_(p10):.1f}" y1="30" x2="{sx_(p10):.1f}" y2="56" '
        f'stroke="{AMBER_DIM}" stroke-width="1.4"/>'
        f'<line x1="{sx_(p90):.1f}" y1="30" x2="{sx_(p90):.1f}" y2="56" '
        f'stroke="{AMBER_DIM}" stroke-width="1.4"/>'
        f'<line x1="6" y1="43" x2="994" y2="43" stroke="{RULE}"/>'
        f'<line x1="{sx_(p50):.1f}" y1="24" x2="{sx_(p50):.1f}" y2="62" '
        f'stroke="{AMBER}" stroke-width="2.2"/>'
        f'<text class="m" x="{sx_(p50):.1f}" y="16" fill="{AMBER}" font-size="11" '
        f'text-anchor="middle" font-family="IBM Plex Mono,monospace">estimación</text>'
        f'<circle cx="{sx_(pedido):.1f}" cy="43" r="7" fill="{color}" '
        f'stroke="{VOID}" stroke-width="2"/>'
        f'<text x="{sx_(pedido):.1f}" y="78" fill="{color}" font-size="11" '
        f'text-anchor="middle" font-family="IBM Plex Mono,monospace">'
        f'pedido {compact(pedido)} €</text></svg>', unsafe_allow_html=True)

# ─────────────────────────── curvas de respuesta ──────────────────────
def curva(xs, ys, xn, yn, unidad, ident):
    Wc, Hc, L, R, T, B = 520, 190, 12, 508, 26, 44
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    pad = (ymax - ymin) * .18 or 1
    ymin, ymax = ymin - pad, ymax + pad
    fx = lambda v: L + (v - xmin) / max(xmax - xmin, 1e-9) * (R - L)
    fy = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (Hc - B - T)
    pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
    ex = lambda v: num(v, 1).rstrip("0").rstrip(",") if unidad == " km" else num(v)
    g = "".join(f'<line x1="{L}" y1="{T+(Hc-B-T)*i/4:.1f}" x2="{R}" '
                f'y2="{T+(Hc-B-T)*i/4:.1f}" stroke="{RULE}"/>' for i in range(1, 4))
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


st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)
g1, g2 = st.columns(2, gap="large")
xa = np.linspace(max(25, area - 55), min(600, area + 55), 22)
ya = predecir([fila(float(a), lat0, lon0) for a in xa]) * xa
with g1:
    st.markdown('<div class="eb">Elasticidad · superficie</div>', unsafe_allow_html=True)
    st.markdown(curva(list(xa), list(ya), area, p50, " m²", "a"), unsafe_allow_html=True)
    pend = (float(ya[-1]) - float(ya[0])) / (xa[-1] - xa[0])
    st.markdown(f'<div class="prose" style="font-size:.88rem;margin-top:6px">Cada m² '
                f'adicional en torno a los {area} m² añade <b>{eur(pend)}</b>.</div>',
                unsafe_allow_html=True)

xd = np.linspace(0.05, 3.0, 22)
yd = predecir([fila(area, lat0, lon0, dm=float(v)) for v in xd]) * area
with g2:
    st.markdown('<div class="eb">Gradiente · acceso al metro</div>', unsafe_allow_html=True)
    st.markdown(curva(list(xd), list(yd), min(d_metro, 3.0), p50, " km", "d"),
                unsafe_allow_html=True)
    cai = (float(yd[0]) - float(yd[-1])) / float(yd[0]) * 100
    st.markdown(f'<div class="prose" style="font-size:.88rem;margin-top:6px">Pasar de 50 m '
                f'a 3 km de una boca de metro resta un <b>{cai:.1f}%</b>.</div>',
                unsafe_allow_html=True)

# ─────────────────────────── diagnóstico ──────────────────────────────
R = resumen()
st.markdown("<div style='height:42px'></div>", unsafe_allow_html=True)
st.markdown('<div class="eb">Rendimiento del motor · validación por bloques espaciales</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="diag">'
    f'<div><div class="v">{meta["r2_bloques"]:.3f}</div><div class="k">R² · {ciudad}</div></div>'
    f'<div><div class="v">{eur(meta["mae"])}</div><div class="k">Error medio absoluto</div></div>'
    f'<div><div class="v">{meta["cobertura_intervalo"]*100:.1f} %</div>'
    f'<div class="k">Cobertura real (teórica 80 %)</div></div>'
    f'<div><div class="v">{num(meta["n_anuncios"])}</div><div class="k">Anuncios</div></div>'
    f'</div>', unsafe_allow_html=True)

st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    st.markdown('<div class="eb">Los tres mercados</div>', unsafe_allow_html=True)
    filas = "".join(
        f'<tr><td style="padding:9px 0;border-bottom:1px solid {RULE}">{c}</td>'
        f'<td class="m" style="text-align:right;border-bottom:1px solid {RULE}">'
        f'{R[c]["r2_bloques"]:.3f}</td>'
        f'<td class="m" style="text-align:right;border-bottom:1px solid {RULE}">'
        f'{eur(R[c]["mae"])}</td>'
        f'<td class="m" style="text-align:right;border-bottom:1px solid {RULE}">'
        f'{R[c]["cobertura_intervalo"]*100:.0f} %</td>'
        f'<td class="m" style="text-align:right;border-bottom:1px solid {RULE};'
        f'color:{AMBER}">×{R[c]["factor_ciudad"]:.2f}</td></tr>' for c in CIUDADES)
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:.86rem">'
        f'<tr style="font-family:IBM Plex Mono,monospace;font-size:.55rem;'
        f'letter-spacing:.16em;text-transform:uppercase;color:{MUTE}">'
        f'<td style="padding-bottom:8px">Mercado</td>'
        f'<td style="text-align:right">R²</td><td style="text-align:right">MAE</td>'
        f'<td style="text-align:right">Cobertura</td>'
        f'<td style="text-align:right">Reindex</td></tr>{filas}</table>'
        f'<div class="prose" style="font-size:.86rem;margin-top:12px">València, con un nivel '
        f'de precio menos de la mitad que Madrid, sirve de prueba de generalización: la misma '
        f'arquitectura cede <b>{(R["Madrid"]["r2_bloques"]-R["Valencia"]["r2_bloques"]):.3f}'
        f'</b> de R² al trasladarse a un mercado estructuralmente distinto.</div>',
        unsafe_allow_html=True)

with c2:
    st.markdown('<div class="eb">Reindexación por distrito</div>', unsafe_allow_html=True)
    dd = sorted([(d, v) for d, v in meta["distritos"].items()
                 if v["n"] >= 120 and not v["estimado"]],
                key=lambda x: -x[1]["factor"])[:9]
    if dd:
        mx = max(v["factor"] for _, v in dd)
        rows = "".join(
            f'<text x="0" y="{i*30+14}" fill="{BONE}" font-size="12" '
            f'font-family="Jost,sans-serif" font-weight="300">{d}</text>'
            f'<rect x="185" y="{i*30+3}" width="{max(v["factor"]/mx*230,2):.1f}" height="14" '
            f'fill="{AMBER}" opacity="{.28+.72*v["factor"]/mx:.2f}"/>'
            f'<text x="{185+max(v["factor"]/mx*230,2)+9:.1f}" y="{i*30+15}" fill="{MUTE}" '
            f'font-size="10.5" font-family="IBM Plex Mono,monospace">×{v["factor"]:.2f}</text>'
            f'<line x1="0" y1="{i*30+23}" x2="460" y2="{i*30+23}" stroke="{RULE}"/>'
            for i, (d, v) in enumerate(dd))
        st.markdown(f'<svg viewBox="0 0 460 {len(dd)*30}" width="100%" '
                    f'style="display:block">{rows}</svg>', unsafe_allow_html=True)
    st.markdown(f'<div class="prose" style="font-size:.84rem;margin-top:10px">Revalorización '
                f'de 2018 a 2026 por distrito. Un factor único de ciudad ignoraría esta '
                f'dispersión.</div>', unsafe_allow_html=True)

with st.expander("Metodología, fuentes y límites"):
    st.markdown(f"""
<div class="prose" style="font-size:.9rem">
<b>Datos.</b> {num(meta['n_anuncios'])} anuncios de {ciudad}
({num(meta['n_viviendas'])} viviendas únicas) de los cuatro trimestres de 2018, del conjunto
abierto <i>idealista18</i> (Rey-Blanco, Arbués, López y Páez, <i>Environment and Planning B</i>,
2024; licencia ODbL), enriquecido con información catastral.<br><br>
<b>Modelo.</b> Ensamblado de árboles con refuerzo de gradiente sobre el logaritmo del precio
por metro cuadrado, no sobre el precio total: así el valor sigue creciendo con la superficie
más allá del rango con datos abundantes, en lugar de saturarse. Se imponen restricciones de
monotonía en las variables cuyo signo no admite duda (baños, ascensor, garaje, piscina,
portería, climatización y terraza suman; distancia al centro y al metro restan).<br><br>
<b>Incertidumbre.</b> Tres modelos de regresión cuantílica (percentiles 10, 50 y 90) dan un
intervalo propio para cada inmueble, estrecho en viviendas corrientes y ancho en las atípicas.
La cobertura real medida es del {meta['cobertura_intervalo']*100:.1f} % frente al 80 % teórico.<br><br>
<b>Validación.</b> Cruzada por bloques espaciales de unos 100 metros, que agrupan también los
anuncios repetidos del mismo inmueble entre trimestres y evitan así la fuga de información.<br><br>
<b>Actualización.</b> La estructura relativa se aprende de 2018; el nivel de precios procede
del índice publicado por distrito en agosto de 2026. La correlación de rangos entre el mapa de
valor de 2018 y el actual es de 0,975 en Madrid: la geografía del valor apenas ha cambiado,
solo su nivel. El índice se actualiza mensualmente.<br><br>
<b>Límites.</b> Si desde 2018 ha cambiado <i>qué</i> valora el comprador —y algo ha cambiado,
como el peso del espacio exterior— el modelo no lo recoge. Por encima del percentil 99 de
superficie ({p99} m² en {ciudad}) las observaciones son escasas y la estimación se marca como
orientativa. No sustituye a una tasación homologada.
</div>""", unsafe_allow_html=True)

st.markdown(
    f'<div class="foot"><div>Valora · Motor de valoración residencial</div>'
    f'<div>Estructura: idealista18 (2018) · Nivel: índice de oferta, agosto 2026</div>'
    f'<div>Estimación orientativa · No sustituye a una tasación homologada</div></div>',
    unsafe_allow_html=True)
