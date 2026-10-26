from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/iris_random_forest_model.joblib")
model = joblib.load(MODEL_PATH)


app = FastAPI(title="Iris Random Forest API")

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def ready_to_serve():
    return {"status": "ready to serve..."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

#end point for making predictiosn with iris random forest
@app.post("/predict")
def predict(data :IrisFeatures):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    return{
        "prediction" : prediction,
        "class_name" : ["sentosa", "versicolor", "virginica"][prediction]

    }

