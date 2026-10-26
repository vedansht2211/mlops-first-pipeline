from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path



# load the data

iris = load_iris()

X = iris.data
y=iris.target

# split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions

y_pred = model.predict(X_test)

# check accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# save the model

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "iris_random_forest_model.joblib"
joblib.dump(model, model_path)
print(f"model saved to {model_path}")

