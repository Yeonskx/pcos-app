import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

# Must be defined here so pickle can find it
class TOMIMSelector(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, percentile=0):
        self.random_state  = random_state
        self.percentile    = percentile
        self.selected_idx_ = None
        self.threshold_    = None

    def fit(self, X, y):
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        self.threshold_    = np.percentile(mi, self.percentile)
        self.selected_idx_ = np.where(mi >= self.threshold_)[0]
        if len(self.selected_idx_) == 0:
            self.selected_idx_ = np.argsort(mi)[-5:]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [input_features[i] for i in self.selected_idx_]
        return self.selected_idx_


p2_model = pickle.load(open("p2_model.pkl", "rb"))

print("Pipeline steps:")
for name, step in p2_model.steps:
    print(f"  '{name}' → {type(step).__name__}")