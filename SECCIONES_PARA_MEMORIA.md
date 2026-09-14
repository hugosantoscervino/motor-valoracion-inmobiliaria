# Secciones redactadas para integrar en la memoria del TFM

**Estado:** texto nuevo basado en el código ejecutado. No sustituye la memoria previa porque no se adjuntó en este turno. Hay que insertar o adaptar estos párrafos en el documento de hasta 20 caras y mantener fuera del cuerpo el detalle técnico del notebook. Todos los euros de error y coberturas se refieren **exclusivamente a precios de oferta de 2018**.

## Resumen ejecutivo

Este trabajo construye un motor de estimación del precio de oferta de viviendas en Madrid, Barcelona y Valencia, a partir de anuncios de 2018. Responde a una pregunta de negocio: si se conocen superficie, localización y algunos atributos del inmueble, ¿cuál es un rango razonable de oferta y cuánto mejora la predicción frente a multiplicar los metros cuadrados por el precio habitual de una zona? El producto entregado combina datos georreferenciados, un modelo interpretable, bandas de incertidumbre calibradas y una aplicación que permite introducir una vivienda nueva. El precio previsto es el que figura en un anuncio, no el valor de cierre de una operación ni una tasación oficial.

La evaluación principal se ha construido separando viviendas, no simplemente filas. Una misma vivienda puede aparecer en más de un trimestre, por lo que todas sus observaciones se asignan siempre a un único conjunto. Los resultados muestran mejoras respecto a reglas de precio por metro cuadrado, aunque también una caída de precisión al limitar los datos de entrada a los campos disponibles en la interfaz. Se incluye una prueba adicional con distritos excluidos y otra que pronostica viviendas nuevas del último trimestre. Un índice de precios posterior permite dibujar **escenarios ilustrativos**; no existe una muestra reciente con la que haya quedado validada su precisión en 2026.

## Problema, datos y derechos de uso

Se utiliza [idealista18](https://github.com/paezha/idealista18), producto de datos georreferenciados de anuncios inmobiliarios publicado con [licencia ODbL](https://github.com/paezha/idealista18/blob/master/LICENSE.md) y descrito por Rey-Blanco, Arbués, López y Páez ([artículo científico](https://doi.org/10.1177/23998083241242844)). Contiene anuncios correspondientes a cuatro trimestres de 2018, coordenadas aproximadas, características de la vivienda e información enriquecida del Catastro y entorno. La variable PRICE indica la petición del vendedor; UNITPRICE es el cociente entre precio y superficie. Se documenta la atribución y la licencia junto a los datos. En el modelo, UNITPRICE solo se emplea como variable a predecir después de transformarla, jamás como variable de entrada: usarlo para predecir PRICE implicaría fuga directa del objetivo.

El flujo reproducible elimina anuncios repetidos para una misma pareja de identificador de vivienda y trimestre, filtra precios y superficies fuera de rangos declarados, y asigna distrito a las coordenadas mediante los polígonos incluidos en la fuente. El fichero de metadatos de cada ciudad informa del número inicial, las exclusiones y la muestra final. No se interpreta que un precio de anuncio sea una venta efectiva.

## Método de modelización y evaluación

Se ajustan tres modelos XGBoost de regresión cuantílica sobre el logaritmo del precio por metro cuadrado: percentiles 10, 50 y 90. La predicción se convierte a euros multiplicando por la superficie. Esta elección permite capturar relaciones no lineales entre atributos, tamaño y posición sin confundir el precio de la vivienda con el precio unitario. Se comparan dos referencias transparentes calculadas **únicamente a partir del entrenamiento**: precio mediano por metro cuadrado de ciudad y de distrito.

Se reservan, por viviendas identificadas mediante ASSETID, aproximadamente 70 % de los activos para entrenamiento, 15 % para calibración y 15 % para test. Ningún activo aparece simultáneamente en dos grupos. Los límites de los cuantiles se corrigen mediante un cuantil conformal calculado únicamente con los errores del grupo de calibración, con un objetivo de cobertura del 80 %. El test se utiliza para comunicar MAE, RMSE, R², error porcentual mediano, ancho de banda y cobertura observada. Una vez comunicado ese test, el modelo servido no se vuelve a entrenar con esos datos: así su evaluación sigue correspondiendo al artefacto desplegado.

El diseño añade tres comprobaciones relevantes. En la prueba espacial se entrena **otro** modelo sin varios distritos completos; en la temporal se entrena **otro** con Q1–Q3 y se evalúan los activos nuevos del Q4. Una ablación vuelve a entrenar el modelo sin coordenadas ni distancias para estimar cuánto aporta la localización. Por último, se simula la utilización de la app sustituyendo características que el usuario no aporta por medianas calculadas solo en train. Las dos pruebas de robustez y la simulación responden preguntas distintas y no deben presentarse como si fueran el mismo test.

## Resultados principales

| Ciudad | Anuncios test | Baseline distrito MAE | Modelo MAE | Mejora MAE | Cobertura del 80 % | MAE con campos de la app | MAE espacial | MAE temporal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Madrid | 11,327 | 84.658 € | 58.719 € | 30.6 % | 82.3 % | 81.879 € | 73.013 € | 65.340 € |
| Barcelona | 6,985 | 75.101 € | 54.605 € | 27.3 % | 81.0 % | 63.721 € | 54.243 € | 57.335 € |
| Valencia | 4,095 | 52.983 € | 34.135 € | 35.6 % | 80.6 % | 52.270 € | 40.827 € | 39.652 € |


*Tabla generada automáticamente desde artefactos/resumen.json. MAE = error absoluto medio. La cobertura se refiere al intervalo calibrado y al test principal de 2018. La prueba espacial y la temporal utilizan otros modelos; la simulación de la app usa el modelo principal con entradas incompletas.*

Las referencias por distrito permiten una decisión sencilla de negocio, pero pierden información sobre características interiores y posición concreta. El modelo completo mejora su MAE en las tres ciudades. La cobertura observada es próxima al 80 % planteado; aun así, no significa que cada distrito, tipo de vivienda o periodo posterior disponga de la misma garantía. Es especialmente importante no citar el MAE del modelo con todos los atributos para describir la precisión de la aplicación: en ella varios datos que el conjunto de entrenamiento sí conoce se sustituyen por medianas. Esa diferencia medida puede guiar una mejora del producto, pidiendo características adicionales o entrenando y validando una versión adaptada estrictamente a los campos que un usuario conoce.

## Interpretabilidad, utilidad y límites

El notebook calcula aportaciones SHAP de árboles en una muestra y representa las variables con mayor magnitud media. Tales aportaciones indican cómo llega el modelo a una estimación, pero no cuantifican el efecto causal de reformar una vivienda, instalar ascensor o cambiar de distrito. La aplicación presenta el precio estimado y un rango calibrado de 2018, junto a las métricas y ofertas de referencia seleccionadas **exclusivamente del entrenamiento**. Al introducir niveles de mercado más recientes, muestra aparte una proyección mediante el factor entre medias de €/m² de zona. Los niveles heredados en indice.json requieren todavía fuentes con enlace, fecha y definición cotejada por distrito. Por ello el intervalo y error de 2018 no se presentan como validados para la cifra reescalada.

Entre las limitaciones destacan el precio solicitado frente al efectivo, las ubicaciones aproximadas, la falta de información sobre estado y reformas, los cambios del mercado posteriores a 2018 y el posible desajuste entre atributos presentes en el dataset y los conocidos por quien usa la app. El coeficiente de reescalado supone una relación de precios estable dentro de la zona; esta hipótesis no se ha comprobado con ventas o anuncios posteriores. Tampoco una alta bondad de ajuste de 2018 demuestra buena precisión para una vivienda singular. Antes de cualquier uso comercial o asesoramiento real sería imprescindible validar con anuncios recientes independientes, revisar representatividad por zona y documentar el origen de cada índice.

## Conclusiones y pasos siguientes

La contribución del trabajo es una valoración de anuncios reproducible, comparada contra reglas sencillas, con tres pruebas complementarias y un prototipo que carga exactamente el modelo evaluado. La primera mejora prioritaria consiste en reunir una muestra independiente de precios de oferta recientes, de modo que pueda comprobarse el reescalado y, si procede, recalibrar bandas. La segunda consiste en decidir qué información puede facilitar una persona usuaria y entrenar una versión que reproduzca esas entradas, contrastándola frente a la actual en una nueva validación. Después convendría estudiar errores en viviendas de alto precio, tamaños poco frecuentes y distritos con pocas observaciones, evitando convertir resultados globales en promesas individuales.

## Anexos, reproducibilidad y bibliografía

El código de entrenamiento, los metadatos y sus dependencias están en el repositorio. El notebook TFM_analisis_y_validacion.ipynb debe entregarse **ejecutado y exportado a HTML** como anexo. Se recomienda enlazar desde la memoria su tabla de flujo de datos, la comparación de baselines, los resultados de robustez y las aportaciones de variables. Los tutores deben poder acceder al repositorio o al ZIP de la entrega.

Referencias principales: Rey-Blanco et al. (2024), [idealista18](https://github.com/paezha/idealista18) y [DOI:10.1177/23998083241242844](https://doi.org/10.1177/23998083241242844); [licencia ODbL](https://github.com/paezha/idealista18/blob/master/LICENSE.md); [guía docente UCM](adjuntar-en-entrega). Añadir para cada cifra de precio posterior a 2018 el informe específico, zona, periodo, URL y fecha de consulta antes de utilizarla en conclusiones.
