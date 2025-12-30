from src.predict import predict

def test_prediction_output_format():
    sample_input = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

    result = predict(sample_input)

    assert "prediction" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
