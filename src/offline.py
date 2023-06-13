import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def batching(df, test_frac, clusters):
    model = Batching()
    model.train(df, test_frac, clusters)
    model.predict()

    return model


class Batching:
    def train(self, df, test_frac, clusters):
        self.data = df.copy()
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        # train a model to identify clusters
        self.model = KMeans(n_clusters=clusters, random_state=42)
        self.model.fit(train)

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))

        # identify the clusters
        cluster = self.model.predict(test)

        self.predictions = pd.DataFrame({
            "Cluster": cluster,
        })
