from pathlib import Path
import numpy as np
import pandas as pd
import rdata
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
DATA = Path(__file__).resolve().parent / "data"
DISTRITOS = {'Madrid': {'01': 'Centro', '02': 'Arganzuela', '03': 'Retiro', '04': 'Salamanca', '05': 'Chamartín', '06': 'Tetuán', '07': 'Chamberí', '08': 'Fuencarral', '09': 'Moncloa', '10': 'Latina', '11': 'Carabanchel', '12': 'Usera', '13': 'Puente de Vallecas', '14': 'Moratalaz', '15': 'Ciudad Lineal', '16': 'Hortaleza', '17': 'Villaverde', '18': 'Villa de Vallecas', '19': 'Vicálvaro', '20': 'San Blas', '21': 'Barajas'}, 'Barcelona': {'01': 'Ciutat Vella', '02': 'Eixample', '03': 'Sants-Montjuïc', '04': 'Les Corts', '05': 'Sarrià-Sant Gervasi', '06': 'Gràcia', '07': 'Horta-Guinardó', '08': 'Nou Barris', '09': 'Sant Andreu', '10': 'Sant Martí'}, 'Valencia': {'01': 'Ciutat Vella', '02': "L'Eixample", '03': 'Extramurs', '04': 'Campanar', '05': 'La Saïdia', '06': 'El Pla del Real', '07': "L'Olivereta", '08': 'Patraix', '09': 'Jesús', '10': 'Quatre Carreres', '11': 'Poblats Marítims', '12': 'Camins al Grau', '13': 'Algirós', '14': 'Benimaclet', '15': 'Rascanya', '16': 'Benicalap', '17': 'Pobles del Nord', '18': "Pobles de l'Oest", '19': 'Pobles del Sud'}}

CONV = rdata.conversion.SimpleConverter(default_encoding='utf-8')

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
    poly = pd.DataFrame(leer_rda(f'{ciudad}_Polygons.rda', f'{ciudad}_Polygons'))
    geoms, codigos, nombres = ([], [], [])
    for i in range(len(poly)):
        g = poligono(poly['geometry'].values[i])
        if g is None or not g.is_valid:
            continue
        geoms.append(g)
        codigos.append(str(poly['LOCATIONID'].values[i]).split('-')[7])
        nombres.append(str(poly['LOCATIONNAME'].values[i]))
    arbol = STRtree(geoms)
    puntos = [Point(lo, la) for lo, la in zip(df.LONGITUDE.values, df.LATITUDE.values)]
    idx = arbol.query(puntos, predicate='within')
    asign = np.full(len(df), -1, dtype=int)
    asign[idx[0]] = idx[1]
    mapa = DISTRITOS[ciudad]
    df['DISTRITO'] = [mapa.get(codigos[i], 'Otros') if i >= 0 else 'Otros' for i in asign]
    centroides = [{'lat': round(g.centroid.y, 5), 'lon': round(g.centroid.x, 5), 'barrio': n, 'distrito': mapa.get(c, 'Otros')} for g, c, n in zip(geoms, codigos, nombres)]
    return (df, centroides)
