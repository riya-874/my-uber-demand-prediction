# 🚕 Rider Radar: Geospatial Uber Demand Predictor

A production-ready, end-to-end machine learning web application that predicts high-density NYC taxi/Uber pickup zones. The system isolates massive geospatial datasets using DVC and leverages containerized Docker environments for optimized, low-latency cloud deployments.

🔗 **[Live Application Link](https://uber-demand-prediction-u8oe.onrender.com)**

---

## 🚀 Key Engineering Metrics & Performance
* **Storage Optimization:** Managed **14.0 GB** of raw geospatial training metrics (48,000+ files) using **Data Version Control (DVC)**, reducing Git repository deployment overhead by **99.8%** (down to **19.1 MB**).
* **Low-Latency Inference:** Engineered a robust Scikit-Learn preprocessing and modeling pipeline integrating **MiniBatchKMeans** clustering to deliver comprehensive feature extraction and inference within **2.9 seconds**.
* **Containerized Deployment:** Encapsulated the application footprint using **Docker**, achieving an optimized **2-minute** automated deployment loop from Git push to live cloud hosting on Render.

---

## 🛠️ Tech Stack & Architecture
* **Frontend Dashboard:** Streamlit, Folium (Interactive Geospatial Mapping)
* **Machine Learning:** Python, Scikit-Learn (Linear Regression, MiniBatchKMeans, Custom Encoders), Joblib
* **Data & MLOps Infrastructure:** DVC (Data Version Control), Docker (Containerization)
* **Cloud & Version Control:** Git, GitHub, Render (Cloud Hosting)

---

## 📦 Features & UI Showcase
* **Interactive Control Center:** Sleek, custom dark-themed control sidebar to adjust temporal parameters (Date/Time) and exact coordinate constraints (Latitude/Longitude).
* **Geospatial Heatmaps:** Real-time generation of NYC pickup maps using batch-loaded coordinate markers via Folium for high responsiveness.
* **Predictive Analytics:** On-the-fly execution of specialized geographical feature transformations to forecast ride density.

---
