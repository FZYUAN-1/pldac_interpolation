import model_physique as m_phy
import methods as mtds
import dataSource as ds
import numpy as np


df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)

tr = ds.trouve_data_case(df,(2,1),latitude_min,longitude_min,ecart_x,ecart_y)

m = m_phy.modele_physique()

print(tr)

# 
datax = []
tr_arr = tr.to_numpy()
for i in range(len(tr)-2):
    datax.append([tr_arr[i],tr_arr[i+1],tr_arr[i+2]])
    i+=3

datax = np.array(datax)

