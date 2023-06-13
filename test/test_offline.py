import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.offline import plot
import sys
sys.path.append("/home/nick/Clustering/src")
from offline import batching


def scatter(df, x, y, color=None, title="Scatter Plot", font_size=None):
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig, filename=f"{title}.html")


data = pd.read_csv("/home/nick/Clustering/test/LungCap.csv")
data = data.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
data = data.drop(columns=["Gender female", "Smoke no"])  # remove extra columns

model = batching(
    df=data, 
    test_frac=0.5,
    clusters=2,
)

# standardize the columns to take on values between 0 and 1
columns = data.columns
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=columns)

test = data.copy().tail(int(len(data)*0.5))

# train a PCA model
n_comp = 2 # number of principal components
component = PCA(n_components=n_comp, random_state=42)
component.fit(test)

# compute components for all the data
components = pd.DataFrame(
    component.transform(test), 
    columns=[f"PC{str(i + 1)}" for i in range(n_comp)],
)

# plot the clusters on the principal components
components["Cluster"] = model.predictions["Cluster"]

scatter(
    components,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="Lung Capacity Clusters",
    font_size=16,
)
