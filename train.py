from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
# Client API lets us manage registered models and their aliases
from mlflow import MlflowClient


# Configure mlflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Group related training runs under one experiments
mlflow.set_experiment("iris-classification")



# load the data

iris = load_iris()

X = iris.data
y=iris.target

# split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=42)


n_estimators = 2
random_state = 12

model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.65
}

# Start a tracked training run
with mlflow.start_run() as run:

    # record the training configuration
    mlflow.log_params(model_parameters)

    # build classifier using selected parameters
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # train
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # check precision
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    print("Precision:", precision)

    # check recall
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    print("Recall:", recall)


    # Model Quality Gate
    # define the minm acceptable test accuracy for now
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy: {accuracy:.2f}%"
            f"is below the required {MIN_TEST_ACCURACY}"
        )
    print("Model Quality Gate passed ......")
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    # save the model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "iris_random_forest_model.joblib"
    joblib.dump(model, model_path)
    print(f"model saved to {model_path}")

    # model registration
    mlflow.sklearn.log_model(
        sk_model = model,
        name = "iris-model",
        registered_model_name = "iris-classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    # Assign the champion alias to the newly registered model
    client = MlflowClient()
    # Identify current training run
    # we use this run ID to find the model registered by this particular run
    current_run_id = mlflow.active_run().info.run_id

    # Retrieve all versions registered under this model name
    registered_versions = client.search_model_versions(
        "name = 'iris-classifier'"
    )

    #find the version created by the current training run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    client.set_registered_model_alias(
        name = "iris-classifier",
        alias = "champion",
        version = current_model_version.version
    )
    print("Registered Model : iris-classifier")

