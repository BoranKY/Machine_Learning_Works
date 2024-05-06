import datetime
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import datetime as dt
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


############################################ EXPLORATORY DATA ANALYSIS ##########################################

df = pd.read_csv("datasets/flo_data_20k.csv")

df.head()

today = dt.datetime(2021,6,2)
df.info()


date_col = [col for col in df.columns if "date" in col ]

for col in date_col:
    df[col] = pd.to_datetime(df[col])

df.head()

df["Total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["Total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


df["Recency"] = (today - df["last_order_date"])
df["Tenure"] = (df["last_order_date"] - df["first_order_date"])


df["Recency"] = df["Recency"].dt.days
df["Tenure"] = df["Tenure"].dt.days


cols = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online",
        "Total_order","Total_value","Recency","Tenure"]

df.head()

############################################ Customer Segmentation with K-Means ##########################################


dff = df[cols]
dff.head()

sc = MinMaxScaler((0,1))
dff = sc.fit_transform(dff)


k_means = KMeans(n_clusters=4).fit(dff)
k_means.get_params()


k_means = KMeans()
elbow = KElbowVisualizer(k_means,k=(2,20))
elbow.fit(dff)
elbow.show()

elbow.elbow_value_

k_means_final = KMeans(n_clusters=elbow.elbow_value_).fit(dff)

clusters = k_means_final.labels_

df["clusters"] = clusters
df.head()
df["clusters"] = df["clusters"] + 1

df["clusters"].describe().T

df.groupby("clusters").agg({"clusters":["count"]})


############################################ Customer Segmentation with Hierarchical Clustering ##########################################



hc_average = linkage(dff,"average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()


clus = AgglomerativeClustering(n_clusters=5,linkage="average")

clus_hg = clus.fit_predict(dff)

df["hi_cluster_no"] = clus_hg
df["hi_cluster_no"] = df["hi_cluster_no"] + 1
df.head()

df.groupby("hi_cluster_no").agg({"hi_cluster_no":["count"]})














