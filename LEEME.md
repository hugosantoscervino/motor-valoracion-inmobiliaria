# VALORA · Motor de valoración residencial

Madrid · Barcelona · València

## Qué hay aquí

| Fichero | Qué es |
|---|---|
| `app.py` | La aplicación web |
| `entrenar.py` | Tubería completa de entrenamiento (va al anexo de la memoria) |
| `artefactos/` | 9 modelos + 3 metadatos + resumen |
| `requirements.txt` | Dependencias |
| `.streamlit/config.toml` | Tema |

## Cómo desplegarlo

Sube todo al repositorio, respetando la carpeta `artefactos/`. Streamlit
se reinicia solo.

## Cómo reentrenar

Necesita el paquete `idealista18` descomprimido, con `data/` al lado:

    pip install rdata xgboost scikit-learn shapely pandas numpy
    python entrenar.py            # las tres ciudades
    python entrenar.py Madrid     # una sola

Genera `artefactos/` de nuevo.

## Cómo actualizar los precios a un trimestre posterior

En `entrenar.py`, el diccionario `NIVEL_2026` guarda el €/m² publicado por
distrito. Se sustituyen los valores por los del mes corriente y se vuelve a
ejecutar: los modelos no cambian, solo el nivel. Fuente: informes de precios
de idealista, o el valor tasado trimestral del Ministerio de Vivienda.

## Datos

`idealista18` — Rey-Blanco, Arbués, López y Páez (2024),
*Environment and Planning B: Urban Analytics and City Science*.
DOI 10.1177/23998083241242844. Licencia ODbL.
