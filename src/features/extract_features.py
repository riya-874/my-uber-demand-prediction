import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def read_cluster_input(data_path, chunksize=100000, usecols=["pickup_latitude","pickup_longitude"]):
    df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)
    return df_reader


def save_model(model, save_path):
    joblib.dump(model, save_path)
    

def read_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = safe_load(file)
    return params
    

if __name__ == "__main__":

    current_path = Path(__file__)

    root_path = current_path.parent.parent.parent

    data_path = root_path / "data/interim/df_without_outliers.csv"
    

    df_all = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    df_all = df_all.sort_values("tpep_pickup_datetime")
    logger.info("Data loaded and sorted chronologically")
    
    df_train = df_all[df_all["tpep_pickup_datetime"] < "2016-03-01"].copy()
    df_test = df_all[df_all["tpep_pickup_datetime"] >= "2016-03-01"].copy()
    logger.info(f"Split data into Train ({len(df_train)} rows) and Test ({len(df_test)} rows)")
    
    scaler = StandardScaler()
    train_coords = df_train[["pickup_longitude", "pickup_latitude"]]
    scaler.fit(train_coords)
    
    scaler_save_path = root_path / "models/scaler.joblib"
    save_model(scaler, scaler_save_path)
    logger.info("Scaler fitted on training data and saved successfully")
    
    mini_batch_params = read_params()["extract_features"]["mini_batch_kmeans"]
    print("Parameters for clustering are ", mini_batch_params)
    
    mini_batch = MiniBatchKMeans(**mini_batch_params)
    scaled_train_coords = scaler.transform(train_coords)
    mini_batch.fit(scaled_train_coords)
        
    kmeans_save_path = root_path / "models/mb_kmeans.joblib"
    joblib.dump(mini_batch, kmeans_save_path)
    logger.info("K-Means fitted on training data and saved successfully")
    
    df_train['region'] = mini_batch.predict(scaler.transform(df_train[["pickup_longitude", "pickup_latitude"]]))
    df_test['region'] = mini_batch.predict(scaler.transform(df_test[["pickup_longitude", "pickup_latitude"]]))
    
    df_final = pd.concat([df_train, df_test], axis=0)
    logger.info("Cluster predictions are added cleanly to datasets")
    
    df_final = df_final.drop(columns=["pickup_latitude","pickup_longitude"])
    logger.info("Latitude and Longitude columns are dropped")
    
    df_final.set_index('tpep_pickup_datetime', inplace=True)

    region_grp = df_final.groupby("region")

    resampled_data = (
                        region_grp['region']
                        .resample("15min")
                        .count()
                    )
    logger.info("Data converted to 15 min intervals successfully")
    resampled_data.name = "total_pickups"
    
    resampled_data = resampled_data.reset_index(level=0)

    epsilon_val = 10
    resampled_data.replace({'total_pickups': {0 : epsilon_val}}, inplace=True)
    
    ewma_params = read_params()["extract_features"]["ewma"]
    print("Parameters for EWMA are ", ewma_params)    
    
    resampled_data["avg_pickups"] = (
                resampled_data
                .groupby("region")['total_pickups']
                .ewm(**ewma_params)
                .mean()
                .shift(1)
                .round()
                .values
            )
    logger.info("Average pickups calculated successfully using EWMA")
    
    save_path = root_path / "data/processed/resampled_data.csv"
    resampled_data.to_csv(save_path, index=True)
    logger.info("Data saved successfully")