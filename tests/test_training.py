import os

def test_model_files_exist():
    assert os.path.exists("models/random_forest_model.pkl")
    assert os.path.exists("models/scaler.pkl")
