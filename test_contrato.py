"""Pruebas del contrato entre preparación, resultados guardados y predicción."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from modelado import preparar, correccion_conformal

ROOT=Path(__file__).resolve().parents[1]

class Contract(unittest.TestCase):
    def test_target_and_period_do_not_enter_features(self):
        d=pd.DataFrame({'ASSETID':['a'],'PRICE':[100000], 'UNITPRICE':[1000],
                        'PERIOD':[201812],'CONSTRUCTEDAREA':[100],
                        'ROOMNUMBER':[2],'CADCONSTRUCTIONYEAR':[2000]})
        x=preparar(d)
        self.assertFalse({'ASSETID','PRICE','UNITPRICE','PERIOD'} & set(x))
        self.assertEqual(x['AREA_POR_HABITACION'].iloc[0],50)

    def test_calibration_uses_finite_sample_quantile(self):
        correction=correccion_conformal(np.array([0,0,0,10,10]),np.zeros(5),np.ones(5),alpha=.2)
        self.assertEqual(correction,9)

    def test_served_models_match_metadata(self):
        for city in ('madrid','barcelona','valencia'):
            meta=json.loads((ROOT/'artefactos'/f'{city}_meta.json').read_text())
            x=pd.DataFrame([meta['medianas_train']])[meta['features']]
            self.assertFalse({'PRICE','UNITPRICE','PERIOD','ASSETID'} & set(x))
            self.assertGreater(meta['evaluacion']['n_test'],100)
            for q in ('q10','q50','q90'):
                model=xgb.XGBRegressor()
                model.load_model(str(ROOT/'artefactos'/f'{city}_{q}.ubj'))
                v=float(model.predict(x)[0])
                self.assertTrue(np.isfinite(v))

if __name__=='__main__':unittest.main()
