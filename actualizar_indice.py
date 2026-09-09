"""
ALDABA · Actualización del nivel de precios
===========================================
Reescribe indice.json con los precios corrientes por distrito.

Uso:
    python actualizar_indice.py                 # muestra qué hay ahora
    python actualizar_indice.py --ine           # trae el IPV del INE y reindexa
    python actualizar_indice.py --editar        # plantilla para meterlos a mano

No toca los modelos. Después de ejecutarlo basta con subir indice.json.
"""
import argparse, json, sys
from datetime import date
from pathlib import Path

RUTA = Path(__file__).resolve().parent / "indice.json"

# Serie del Índice de Precios de Vivienda del INE (base 2015 = 100), vivienda
# libre, variación por comunidad autónoma. Tabla 25171 en el INE.
INE_TABLA = "https://servicioswebine.ine.es/wstempus/js/ES/DATOS_TABLA/25171?nult=4"
CCAA = {"Madrid": "Madrid, Comunidad de",
        "Barcelona": "Cataluña",
        "Valencia": "Comunitat Valenciana"}


def cargar():
    return json.loads(RUTA.read_text(encoding="utf-8"))


def guardar(ind):
    RUTA.write_text(json.dumps(ind, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {RUTA.name} actualizado — fecha: {ind['fecha']}")


def mostrar(ind):
    print(f"Nivel vigente: {ind['fecha']}  ({ind['fuente']})\n")
    for ciudad, d in ind["ciudades"].items():
        print(f"  {ciudad}  ·  media de ciudad {d['_ciudad']:,} €/m²".replace(",", "."))
        for k, v in sorted(((k, v) for k, v in d.items() if k != "_ciudad"),
                           key=lambda x: -x[1]):
            print(f"      {k:24s} {v:>7,} €/m²".replace(",", "."))
        print()


def desde_ine(ind):
    """Aplica al índice vigente la variación del IPV desde su última lectura."""
    try:
        import requests
    except ImportError:
        sys.exit("Necesitas 'requests':  pip install requests")
    print(f"Consultando el INE…\n  {INE_TABLA}")
    r = requests.get(INE_TABLA, timeout=30)
    r.raise_for_status()
    datos = r.json()

    series = {}
    for s in datos:
        nombre = s.get("Nombre", "")
        vals = [d for d in s.get("Data", []) if d.get("Valor") is not None]
        if len(vals) >= 2:
            series[nombre] = vals

    aplicados = 0
    for ciudad, region in CCAA.items():
        cand = [n for n in series if region in n and "Índice" in n]
        if not cand:
            print(f"  ! {ciudad}: no encuentro la serie de {region}, lo dejo igual")
            continue
        v = series[cand[0]]
        var = v[0]["Valor"] / v[1]["Valor"]
        for k in ind["ciudades"][ciudad]:
            ind["ciudades"][ciudad][k] = round(ind["ciudades"][ciudad][k] * var)
        print(f"  {ciudad}: variación x{var:.4f} aplicada a "
              f"{len(ind['ciudades'][ciudad])} zonas")
        aplicados += 1

    if aplicados:
        hoy = date.today()
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                 "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        ind["fecha"] = f"{meses[hoy.month-1]} de {hoy.year}"
        ind["fecha_corta"] = f"{meses[hoy.month-1][:3]}-{hoy.year}"
        ind["fuente"] = "INE · Índice de Precios de Vivienda, sobre base idealista"
    return ind


def plantilla(ind):
    print("Sustituye los números por los del mes corriente y pega el resultado "
          "en indice.json:\n")
    print(json.dumps(ind, ensure_ascii=False, indent=1))
    print("\nDónde mirarlos:")
    print("  · idealista — informes de precios por distrito, mensual")
    print("    https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/")
    print("  · Ministerio de Vivienda — valor tasado por municipio, trimestral")
    print("  · INE — Índice de Precios de Vivienda por provincia, trimestral")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ine", action="store_true", help="traer la variación del INE")
    p.add_argument("--editar", action="store_true", help="volcar plantilla editable")
    a = p.parse_args()
    ind = cargar()
    if a.ine:
        guardar(desde_ine(ind))
    elif a.editar:
        plantilla(ind)
    else:
        mostrar(ind)
