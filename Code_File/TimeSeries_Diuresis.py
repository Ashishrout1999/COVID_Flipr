import pandas as pd
import numpy as np
from pandas import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

DU_data = pd.read_excel('../Dataset/Train_dataset(1).xlsx', 'Diuresis_TS', date_parser=parser)

I = DU_data.iloc[8:15,10:10725]

A = []

# A contains diureis value of patients on 27 March
for i in range(1,10715):
    I_model = I.iloc[:,[i]]
    model = ExponentialSmoothing(np.asarray(I_model), trend ='mul', seasonal = None)
    model_fit = model.fit()
    yhat = model_fit.predict(7,7)
    A.append(float(yhat))
  
    
    
    
    

