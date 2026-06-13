# --- UPDATE THIS AT THE TOP OF YOUR PLOT DATA GENERATION SCRIPT ---
import pandas as pd
import joblib
from pathlib import Path

script_dir = Path(__file__).parent

root_path = script_dir.parent.parent  

data_path = root_path / "data" / "interim" / "final_data.csv"
kmeans_path = root_path / "models" / "mb_kmeans.joblib"
scaler_path = root_path / "models" / "scaler.joblib"
output_path = root_path / "data" / "external" / "plot_data.csv"

print("Loading datasets and models...")
df = pd.read_csv(data_path)
kmeans = joblib.load(kmeans_path)
scaler = joblib.load(scaler_path)


scaled_centers = kmeans.cluster_centers_
real_centers = scaler.inverse_transform(scaled_centers)

coords_lookup = pd.DataFrame(real_centers, columns=["pickup_latitude", "pickup_longitude"])
coords_lookup["region"] = coords_lookup.index

sampled_regions = []
print(" Sampling 500 data points per region...")

for region_id in df["region"].unique():
    region_data = df[df["region"] == region_id]
    
    sample_size = min(500, len(region_data))
    region_sample = region_data.sample(n=sample_size, random_state=42)
    
    sampled_regions.append(region_sample)

plot_df = pd.concat(sampled_regions, ignore_index=True)

print("Merging geographical coordinates...")
plot_df = plot_df.merge(coords_lookup, on="region", how="left")

output_path.parent.mkdir(parents=True, exist_ok=True)

plot_df.to_csv(output_path, index=False)
print(f" Created sample file at: {output_path}")
print(f"Total rows generated: {len(plot_df)} ({len(df['region'].unique())} regions x 500 samples)")