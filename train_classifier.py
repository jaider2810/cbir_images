import os
import numpy as np
import joblib
from descripteurs import concat_feat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X = []
y = []

for cls in os.listdir("dataset"):
    for f in os.listdir(os.path.join("dataset", cls)):
        p = os.path.join("dataset", cls, f)
        X.append(concat_feat(p))
        y.append(cls)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(X_train, y_train)

joblib.dump(knn, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")
