import datetime as dt
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")

root_path = Path(__file__).parent
model_path = root_path / "models" / "model.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(
        f" Could not find model.joblib at {model_path}. Did you run 'dvc repro' first?"
    )
    st.stop()

plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"

kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"

scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
kmeans = joblib.load(kmeans_path)


df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index(
    "tpep_pickup_datetime"
)


st.title("Uber Demand in New York City 🚕🌆")

st.sidebar.title("Options")
map_type = st.sidebar.radio(
    label="Select the type of Map",
    options=["Complete NYC Map", "Only for Neighborhood Regions"],
    index=1,
)

st.subheader("Date")
date = st.date_input(
    "Select the date",
    value=None,
    min_value=dt.date(year=2016, month=3, day=1),
    max_value=dt.date(year=2016, month=3, day=31),
)
st.write("**Date:**", date)

st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    # next time interval
    delta = dt.timedelta(minutes=15)
    next_interval = (
        dt.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute,
        )
        + delta
    )
    st.write("Demand for Time: ", next_interval.time())

    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)

    # sample a latitude longitude value
    st.subheader("Location")
    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    current_region = sample_loc["region"].item()
    st.write("**Your Current Location**")
    st.write(f"Lat: {lat}")
    st.write(f"Long: {long}")

    with st.spinner("Fetching your Current Region"):
        sleep(3)

    st.write("Region ID: ", current_region)

    scaled_cord = scaler.transform(
        sample_loc[["pickup_longitude", "pickup_latitude"]]
    )

    st.subheader("MAP")

    colors = [
        "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
        "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
        "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
        "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
        "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
        "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA",
    ]

    region_colors = {
        r: colors[i % len(colors)]
        for i, r in enumerate(df_plot["region"].unique().tolist())
    }
    df_plot["color"] = df_plot["region"].map(region_colors)

    # make prediction pipeline
    pipe = Pipeline([("encoder", encoder), ("reg", model)])

    if map_type == "Complete NYC Map":
        progress_bar = st.progress(
            value=0, text="Operation in progress. Please wait."
        )
        for percent_complete in range(100):
            sleep(0.01)
            progress_bar.progress(
                percent_complete + 1, text="Operation in progress. Please wait."
            )

       
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB dark_matter")
        
        # Batch add the color-coded points to the map canvas
        for _, row in df_plot.iterrows():
            folium.CircleMarker(
                location=[row["pickup_longitude"], row["pickup_latitude"]],
                radius=5,
                color=row["color"],
                fill=True,
                fill_color=row["color"],
                fill_opacity=0.7,
                weight=0
            ).add_to(m)
            
        # Render map inside Streamlit
        st_folium(m, width=700, height=500, returned_objects=[])

        progress_bar.empty()

        input_data = df.loc[index, :].sort_values("region")
        target = input_data["total_pickups"]
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(0, 30):
            color = colors[ind]
            demand = predictions[ind]
            region_id = f"{ind} (Current region)" if current_region == ind else ind
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )

    elif map_type == "Only for Neighborhood Regions":
        distances = kmeans.transform(scaled_cord).values.ravel().tolist()
        distances = list(enumerate(distances))
        sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]
        indexes = sorted([ind[0] for ind in sorted_distances])

        df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]

        progress_bar = st.progress(
            value=0, text="Operation in progress. Please wait."
        )
        for percent_complete in range(100):
            sleep(0.01)
            progress_bar.progress(
                percent_complete + 1, text="Operation in progress. Please wait."
            )

        m = folium.Map(location=[long, lat], zoom_start=12, tiles="CartoDB dark_matter")
        
        for _, row in df_plot_filtered.iterrows():
            folium.CircleMarker(
                location=[row["pickup_longitude"], row["pickup_latitude"]],
                radius=8,  
                color=row["color"],
                fill=True,
                fill_color=row["color"],
                fill_opacity=0.8,
                weight=0
            ).add_to(m)
            
        st_folium(m, width=700, height=500, returned_objects=[])

        progress_bar.empty()

        input_data = df.loc[index, :]
        input_data = input_data.loc[
            input_data["region"].isin(indexes), :
        ].sort_values("region")
        target = input_data["total_pickups"]
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(0, 9):
            color = colors[indexes[ind]]
            demand = predictions[ind]
            region_id = (
                f"{indexes[ind]} (Current region)"
                if current_region == indexes[ind]
                else indexes[ind]
            )
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f"Region ID: {region_id} <br>"
                f"Demand: {int(demand)} <br> <br>",
                unsafe_allow_html=True,
            )