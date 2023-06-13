import pandas as pd
from river import stream
from river import cluster


def streaming(df, test_frac, clusters, halflife=0.5):
    model = Streaming()
    model.train(df, test_frac, clusters, halflife)
    model.predict()

    return model


class Streaming:
    def train(self, df, test_frac, clusters, halflife):
        self.data = df.copy()
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = cluster.KMeans(n_clusters=clusters, halflife=halflife, seed=42)

        for x, _ in stream.iter_pandas(train):
            self.model = self.model.learn_one(x)

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))

        self.predictions = pd.DataFrame()

        for x, _ in stream.iter_pandas(test):
            cluster = self.model.predict_one(x)
            self.model = self.model.learn_one(x)

            pred = pd.DataFrame({
                "Cluster": [cluster],
            })
            self.predictions = pd.concat([
                self.predictions, 
                pred,
            ], axis="index").reset_index(drop=True)

