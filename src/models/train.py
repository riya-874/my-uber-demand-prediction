import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


set_config(transform_output="pandas")

logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def save_model(model, save_path):
    joblib.dump(model, save_path)
    
    
if __name__ == "__main__":

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    data_path = root_path / "data/processed/train.csv"
    

    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    

    df.set_index("tpep_pickup_datetime", inplace=True)
    

    X_train = df.drop(columns=["total_pickups"])
    y_train = df["total_pickups"]
    

    encoder = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first",sparse_output=False), ["region","day_of_week"])
        ], remainder="passthrough", n_jobs=-1)
        
 
    encoder.fit(X_train)

    encoder_save_path = root_path / "models/encoder.joblib"
    joblib.dump(encoder, encoder_save_path)
    logger.info("Encoder saved successfully")

    X_train_encoded = encoder.fit_transform(X_train)
    logger.info("Data encoded successfully")

    lr = LinearRegression()


    lr.fit(X_train_encoded, y_train)
    logger.info("Model trained successfully")
    
    model_save_path = root_path / "models/model.joblib"
    save_model(lr, model_save_path)
    logger.info("Model saved successfully")