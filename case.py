"""
argmin_alpha (X * W = Y_p)

X = [latix, longx]
Y = [latiy, longy]
W = [[w11, w12],[w21, w22]] pour chaque case
Y_p :
p_latiy = w11 * latix + w21 * longx
p_longy = w12 * latix + w22 * longx

dans le cas monidre caréé 
erreur = sum_x_y ((latiy - p_latiy) ** 2 + (longy - p_longy) ** 2)
dffer_de_w11 = sum_x_y (2 * latix * (latiy - p_latiy)) = 0
dffer_de_w21 = sum_x_y (2 * longx * (latiy - p_latiy)) = 0
dffer_de_w12 = sum_x_y (2 * latix * (longy - p_longy)) = 0
dffer_de_w22 = sum_x_y (2 * longx * (longy - p_longy)) = 0
"""

def fit_case_ML(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    "Retourner le paramètre W appris dans la case indiquée."
    df_case = ds.trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y)
    trips = np.unique(df['Trip'])
    for t in trips:
        df2 = df[df['Trip'] == t][['Latitude','Longitude']]
        N = df2['Latitude'].shape[0]
        ens_points = np.hstack((df2['Latitude'].values.reshape(N,1), df2['Longitude'].values.reshape(N,1)))
        ens_x = ens_points[:-1]
        ens_y = ens_points[1:]
    
    latix2 = sum(ens_x[:,0] ** 2)
    latilongx = sum(ens_x[:,0] * ens_x[:,1])
    longx2 = sum(ens_x[:,1] ** 2)
    latixy = sum(ens_x[:,0] * ens_y[:,0])
    latilongxy = sum(ens_x[:,0] * ens_y[:,1])
    longxy = sum(ens_x[:,1] * ens_y[:,1])
    longlatixy = sum(ens_x[:,1] * ens_y[:,0])



    w1 = np.linalg.solve(np.array([[latix2, latilongx],[latilongx, longx2]]), np.array([2*latixy, 2*longlatixy]))
    w2 = np.linalg.solve(np.array([[latix2, latilongx],[latilongx, longx2]]), np.array([2*latilongxy, 2*longxy]))
    
    return np.vstack((w1, w2))

latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)
fit_case_ML(df, (2,1), latitude_min, latitude_max, ecart_x, ecart_y)
