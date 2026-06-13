import mlflow
import dagshub
import json
import pandas as pd
import joblib
from pathlib import Path
import logging
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error


import dagshub
dagshub.init(repo_owner='rajputriya887', repo_name='uber-demand-prediction', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/rajputriya887/uber-demand-prediction.mlflow")

mlflow.set_experiment("DVC Pipeline")

set_config(transform_output="pandas")

logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_model(model_path):
    model = joblib.load(model_path)
    return model


def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)


if __name__ == "__main__":

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"
    
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"]
    
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")
    
    model_path = root_path / "models/model.joblib"
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    
    y_pred = model.predict(X_test_encoded)
    
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Loss: {loss}")
    
    with mlflow.start_run(run_name="model"):    
  
        mlflow.log_params(model.get_params())
        
        mlflow.log_metric("MAPE", loss)
        
        training_data = mlflow.data.from_pandas(pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        validation_data = mlflow.data.from_pandas(pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        
        mlflow.log_input(training_data, "training")
        mlflow.log_input(validation_data, "validation")
        
        model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
        
        logged_model = mlflow.sklearn.log_model(model, "demand_prediction", 
                                         signature=model_signature,
                                         pip_requirements="requirements.txt")
        
        
        run_id = logged_model.run_id
        artifact_path = logged_model.artifact_path
        model_uri = logged_model.model_uri 
        logger.info("Mlflow logging complete")
    
    json_file_save_path = root_path / "run_information.json"
    save_run_information(run_id=run_id,
                         artifact_path=artifact_path,
                         model_uri=model_uri,
                         path=json_file_save_path)
    logger.info("Run information saved successfully")
        
        
    run_id = logged_model.run_id
    artifact_path = logged_model.artifact_path
    model_uri = logged_model.model_uri 
    logger.info("Mlflow logging complete")
    
    json_file_save_path = root_path / "run_information.json"
    save_run_information(run_id=run_id,
                         artifact_path=artifact_path,
                         model_uri=model_uri,
                         path=json_file_save_path)
    logger.info("Run information saved successfully")
    
    