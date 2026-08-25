import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE/'modelo_valoracion.joblib')
meta = json.loads((BASE/'resultados_tfm.json').read_text(encoding='utf-8'))
st.set_page_config(page_title='Valorador inmobiliario', layout='centered')
st.title('Valorador inmobiliario · Madrid y Barcelona')
st.caption('MVP académico basado en anuncios de 2018. No sustituye una tasación oficial.')
city = st.selectbox('Ciudad', ['Madrid','Barcelona'])
area = st.number_input('Superficie construida (m²)', 20.0, 500.0, 90.0)
rooms = st.number_input('Habitaciones', 0, 15, 3)
baths = st.number_input('Baños', 0, 10, 2)
year = st.number_input('Año de construcción (Catastro)', 1600, 2018, 1970)
d_center = st.number_input('Distancia al centro (km)', 0.0, 30.0, 3.0)
d_metro = st.number_input('Distancia al metro (km)', 0.0, 10.0, 0.4)
lat_default, lon_default = (40.4168,-3.7038) if city=='Madrid' else (41.3874,2.1686)
lat = st.number_input('Latitud', value=float(lat_default), format='%.5f')
lon = st.number_input('Longitud', value=float(lon_default), format='%.5f')
lift = st.checkbox('Ascensor', value=True); terrace = st.checkbox('Terraza'); parking = st.checkbox('Parking'); air = st.checkbox('Aire acondicionado')
if st.button('Estimar precio'):
    d={c:0 for c in meta['features']}
    d.update({'CITY':city,'CONSTRUCTEDAREA':area,'ROOMNUMBER':rooms,'BATHNUMBER':baths,'CADCONSTRUCTIONYEAR':year,'DISTANCE_TO_CITY_CENTER':d_center,'DISTANCE_TO_METRO':d_metro,'LATITUDE':lat,'LONGITUDE':lon,'HASLIFT':int(lift),'HASTERRACE':int(terrace),'HASPARKINGSPACE':int(parking),'HASAIRCONDITIONING':int(air),'AMENITYID':3,'CADMAXBUILDINGFLOOR':6,'CADDWELLINGCOUNT':20,'CADASTRALQUALITYID':5,'BUILTTYPEID_3':1,'FLOORCLEAN':2,'FLATLOCATIONID':1})
    d['BUILDING_AGE']=max(0,2018-year); d['AREA_PER_ROOM']=area/max(rooms,1); d['FLOOR_RATIO']=d['FLOORCLEAN']/max(d['CADMAXBUILDINGFLOOR'],1); d['NEAR_METRO_500M']=int(d_metro<=0.5); d['AMENITIES_COUNT']=int(lift)+int(terrace)+int(parking)+int(air)
    X=pd.DataFrame([{c:d.get(c) for c in meta['features']}]); pred=float(np.expm1(model.predict(X))[0]); q=float(meta['p90_abs_error_eur'])
    st.metric('Precio estimado', f'{pred:,.0f} €'); st.write(f'Banda empírica aproximada del 90%: **{max(0,pred-q):,.0f} € – {pred+q:,.0f} €**')
