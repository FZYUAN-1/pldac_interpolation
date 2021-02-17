import models.model_physique as m_phy
import test.methods as mtds

def importData():
    #1.
    df = pd.read_csv("dataset/DataGpsDas.csv", nrows=1000000)
    df = df[(df["Latitude"] >= 42.282970-0.003) & (df["Latitude"] <= 42.282970+0.003) 
            & (df["Longitude"] >= -83.735390-0.003) & (df["Longitude"] <= -83.735390+0.003)]
    trips, counts = np.unique(df["Trip"], return_counts=True)
    trips = trips[counts>100]
    df = df[df['Trip'].isin(trips)]
    return df




model = m_phy.modele_physique()

datax = []
tr_arr = tr.to_numpy()
for i in range(len(tr)-2):
    datax.append([tr_arr[i],tr_arr[i+1],tr_arr[i+2]])
    i+=3

datax = np.array(datax)