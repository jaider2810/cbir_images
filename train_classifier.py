import os
import joblib
from descripteurs import concat_feat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = []
y = []

for cls in os.listdir("dataset"):
    for f in os.listdir(os.path.join("dataset", cls)):
        p = os.path.join("dataset", cls, f)
        X.append(concat_feat(p))
        y.append(cls)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

joblib.dump(model, "best_model.joblib")
