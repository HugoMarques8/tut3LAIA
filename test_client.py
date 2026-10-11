import requests

def test_get_request():
    url = "http://localhost:8000/predict"
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example features for prediction
    response = requests.post(url, json=payload)
    print(response.json())  # Print the response from the API

def test_batch_request():
    url = "http://localhost:8000/predict-batch"
    payload = {"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}  # Example batch features for prediction
    response = requests.post(url, json=payload)
    print(response.json())  # Print the response from the API

if __name__ == "__main__":
    test_get_request()
    test_batch_request()