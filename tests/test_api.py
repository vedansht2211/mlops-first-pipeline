# Import testclient so we can send simulate http requests
from fastapi.testclient import TestClient

import pytest

# import the application object defined in main.py
from main import create_app

# create a reusable test client for all test functions
#client = TestClient(create_app)

# The test will no longer require a runnign MLFlow server
# they test api independently using a fake model

class FakeIrisModel:
    def predict(self, features):
        return [0]


def fake_model_loader():
    return FakeIrisModel()

def fake_version_loader():
    return "test-version"

@pytest.fixture
# fixture helps 
def client():
    app = create_app(
        model_loader = fake_model_loader,
        version_loader= fake_version_loader
    )

    with TestClient(app) as test_client:
        yield test_client



def test_health_endpoint(client):
    """
    verify that the application is running and can respond to a health-check request
    """

    response = client.get("/health")

    #a successful HTTP request should return status code 200
    assert response.status_code == 200

    #now check response body
    assert response.json() == {
        "status":"healthy",
        "model_name": "iris_classification",
        "model_version":"test-version"
        }

def test_valid_iris_prediction(client):
    """
    verify that valid flower measurements produce a prediction
    """
    payload = {
        "sepal_length" : 5.1,
        "sepal_width" : 3.5,
        "petal_length" :  1.6,
        "petal_width" : 0.5
    }

    # send a simulated HTTP POST request with json input
    response = client.post("/predict",json=payload)
    assert response.status_code == 200

    #convert the json response into a py dict
    result =  response.json()

    # check the response contract : both fields must exist
    assert "prediction" in result
    assert "class_name" in result

    # the predicted class must be one of the three iris classes
    assert result["class_name"] in ["sentosa", "versicolor", "virginica"]

    # the predicted integer must be 0,1,2
    assert result["prediction"] in [0,1,2]



def test_invalid_iris_input(client):
    """
    verify that invalid inputs are rejected by FastAPI
    """
    payload = {
            "sepal_length" : 5.1,
            "sepal_width" : 3.5,
            "petal_length" :  -1.6,
            "petal_width" : 0.5
        }
    
    # send a simulated HTTP POST request with json input
    response = client.post("/predict",json=payload)

    #FastAPI normally return HTTP 422 for validation error
    assert response.status_code == 422
    
