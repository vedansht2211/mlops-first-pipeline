# Import testclient so we can send simulate http requests
from fastapi.testclient import TestClient

# import the application object defined in main.py
from main import app

# create a reusable test client for all test functions
client = TestClient(app)



def test_health_endpoint():
    """
    verify that the application is running and can respond to a health-check request
    """

    response = client.get("/health")

    #a successful HTTP request should return status code 200
    assert response.status_code == 200

    #now check response body
    assert response.json() == {"status":"healthy"}

def test_valid_iris_prediction():
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



def test_invalid_iris_input():
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
    
