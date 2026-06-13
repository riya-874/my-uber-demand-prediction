import dask.dataframe as dd
import logging
from pathlib import Path

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70

min_fare_amount_val = 0.50
max_fare_amount_val = 81.0
min_trip_distance_val = 0.25
max_trip_distance_val = 24.43


def read_dask_df(data_path: Path, parse_dates: list=["tpep_pickup_datetime"],
                 columns: list=['trip_distance', 
                                'tpep_pickup_datetime', 
                                'pickup_longitude',
                                'pickup_latitude',
                                'dropoff_longitude', 
                                'dropoff_latitude', 
                                'fare_amount']):
    dd_df = dd.read_csv(data_path, parse_dates=parse_dates, usecols=columns)
    return dd_df


def dask_pipeline(df):

    df = df.loc[(df["pickup_latitude"].between(min_latitude, max_latitude, inclusive="both")) & 
    (df["pickup_longitude"].between(min_longitude, max_longitude, inclusive="both")) & 
    (df["dropoff_latitude"].between(min_latitude, max_latitude, inclusive="both")) & 
    (df["dropoff_longitude"].between(min_longitude, max_longitude, inclusive="both")), :]
    
    df = df.loc[(df["fare_amount"].between(min_fare_amount_val,max_fare_amount_val,inclusive="both")) & 
    (df["trip_distance"].between(min_trip_distance_val,max_trip_distance_val,inclusive="both"))]
    
    logger.info("Outliers are removed successfully")

    cols_to_drop = ['trip_distance', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']
    df = df.drop(cols_to_drop, axis=1)
    logger.info("Columns are dropped successfully")
    

    df = df.compute()
    logger.info("Dask DataFrame is computed successfully")
    return df

if __name__ == "__main__":

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    raw_data_dir = root_path / "data/raw"
    
    df_names = ["yellow_tripdata_2016-01.csv",
                "yellow_tripdata_2016-02.csv",
                "yellow_tripdata_2016-03.csv"]

    dfs = []

    for df_name in df_names:
        df_path = raw_data_dir / df_name
        df = read_dask_df(df_path)
        dfs.append(df)
    logger.info("Dask DataFrames are read successfully")
    
    df_final = dd.concat(dfs, axis=0)
    logger.info("All datasets merged successfully")
    
    df_final = dask_pipeline(df_final)
    logger.info("Dask pipeline is executed successfully")
    
    df_without_outliers_path = root_path / "data/interim/df_without_outliers.csv"
    df_final.to_csv(df_without_outliers_path, index=False)
    logger.info("DataFrame is saved successfully")