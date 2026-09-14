# VALORA · Motor académico de valoración inmobiliaria

Trabajo de Fin de Máster · ciencia de datos y analítica de negocio. Predice **precios de oferta de 2018**, no precios de compraventa ni tasaciones. La proyección a precios posteriores mediante `indice.json` es un escenario **sin validación externa**.

## Estructura

| Archivo | Función |
|---|---|
| `entrenar.py` | Limpieza, particiones por vivienda, modelos, baselines, calibración y evaluaciones espacial y temporal |
| `modelado.py` | Variables derivadas, métricas y cuantiles de conformalización compartidos |
| `geografia.py` | Conversión de `.rda` y asignación de distritos con los polígonos originales |
| `app.py` | Demo Streamlit que carga **los mismos** modelos y métricas de evaluación |
| `artefactos/` | Modelos nuevos y resultados de 2018, generados por `entrenar.py` |
| `notebooks/TFM_analisis_y_validacion.ipynb` | Anexo reproducible en Colab con análisis, comparaciones y explicabilidad |
| `indice.json` | Niveles externos heredados para escenarios ilustrativos; verificar antes de presentarlos como datos actuales |
| `actualizar_indice.py` | Actualización manual con CSV documentado; no consulta una serie del INE incompatible por definición |
| `tests/` | Pruebas mínimas del contrato entre entrenar y servir |

## Reproducir

1. Instala Python 3.12 y ejecuta `python -m pip install -r requirements.txt`.
2. Descomprime `data(2).zip` facilitado con el TFM: `data/Madrid_Sale.rda`, `data/Madrid_Polygons.rda` y los equivalentes de Barcelona y Valencia, junto a `entrenar.py`. No incluyas el ZIP entero en GitHub si no hace falta: está disponible en la fuente original.
3. Ejecuta `python entrenar.py` (tres ciudades) o `python entrenar.py Madrid` (solo una). Los modelos, metadatos, comparables de **entrenamiento**, errores por distrito de **test** y `resumen.json` se escriben directamente en `artefactos/`.
4. Ejecuta `python -m streamlit run app.py` y `python -m unittest discover -s tests`.
5. En Colab, abre el notebook y sube el ZIP del repositorio junto al ZIP de datos. La propia celda los descomprime y repite el entrenamiento; exporta el notebook ejecutado a HTML como anexo.

## Protocolo de evaluación

- Se eliminan duplicados por `ASSETID` y trimestre y observaciones fuera de los rangos declarados. Los tres conjuntos principales (aprox. 70 %/15 %/15 % de activos) son disjuntos por `ASSETID`. Los cuantiles 10/50/90 se ajustan **solo con train**; una corrección conformal se ajusta **solo con calibración**, y test mide MAE, RMSE, error porcentual, R² y cobertura una vez.
- La referencia de ciudad y la referencia de distrito se estiman **solo con train**, y se comparan en el mismo test. No se escoge otro modelo mirando test.
- La prueba espacial entrena otro modelo excluyendo distritos completos. La temporal entrena otro modelo con Q1–Q3 y evalúa activos nuevos de Q4. Estas son pruebas adicionales; no se confunden con el test principal ni con el modelo desplegado.
- La ablación entrena otro modelo sin coordenadas ni distancias; la simulación de la interfaz evalúa el modelo desplegado imputando con las medianas de train las variables que el usuario no introduce. **La cobertura conformal principal no se traslada automáticamente a esta simulación.**
- El modelo desplegado es el modelo del test principal (train), sin reentrenarlo con calibración/test después de calcular métricas. Reentrenar con todos los registros obligaría a recalibrar y obtener nueva evidencia.
- Los 2018 €/m² para el reescalado se calculan sobre el conjunto de entrenamiento; los valores externos del archivo `indice.json` proceden de la versión original del proyecto y **no se han auditado individualmente**. La diferencia entre media externa de anuncios y media de la muestra puede sesgar la proyección; no hay evaluación posterior a 2018.

## Limitaciones y fuentes

La posición es aproximada en los datos originales, el precio es de anuncio, faltan atributos como reforma y conservación, y la imputación para la interfaz supone viviendas con valores típicos. SHAP/contribuciones describen el modelo, no efectos causales. Ver `data/README.md` para el origen, referencias y licencia de la base de datos. Para cumplir la guía de la UCM, la memoria de hasta 20 caras cuenta los resultados a un lector de negocio; notebook y código van en anexo. Las cifras deben copiarse del nuevo `artefactos/resumen.json`, no de ejecuciones anteriores.
