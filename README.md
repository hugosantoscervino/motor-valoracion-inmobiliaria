# Datos para reproducir VALORA

Fuente: [idealista18](https://github.com/paezha/idealista18), Rey-Blanco, Arbués, López y Páez (2024), [DOI:10.1177/23998083241242844](https://doi.org/10.1177/23998083241242844), [licencia ODbL](https://github.com/paezha/idealista18/blob/master/LICENSE.md).

Coloca aquí `Madrid_Sale.rda`, `Madrid_Polygons.rda`, `Barcelona_Sale.rda`, `Barcelona_Polygons.rda`, `Valencia_Sale.rda` y `Valencia_Polygons.rda`, extraídos del paquete de datos original. Los `.rda` de POIs y `properties_by_district.rda` existen en la distribución, pero esta versión del código no los utiliza. No subas `data.zip` como si fuera un dataset propio. Al compartir datos derivados respeta la atribución y las condiciones de la licencia.

`PRICE` es el precio solicitado por el anunciante y `UNITPRICE` es €/m². Hay anuncios del mismo `ASSETID` en distintos trimestres de 2018: sus repeticiones no pueden repartirse entre entrenamiento y test. Las coordenadas publicadas son aproximadas; no identifican con precisión el portal.
