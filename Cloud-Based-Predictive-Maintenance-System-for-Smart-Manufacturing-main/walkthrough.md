# Final Walkthrough: Predictive Maintenance Simulator

## Overview
We've successfully constructed and deployed a fully functional, local predictive maintenance simulation architecture using the `ai4i2020.csv` dataset. The simulation avoids real AWS bills while mimicking a production ML pipeline.

## Achieved Milestones

### 1. Infrastructure (LocalStack & Docker)
- Configured a `docker-compose.yml` to spin up a LocalStack emulator (`localstack/localstack:3.8.1`).
- Used `init_env.py` with `boto3` to:
  - Connect to `http://localhost:4566`
  - Initialize the `factory-data` local S3 bucket.
  - Upload `ai4i2020.csv` into the lake.

### 2. Machine Learning Brain
- The `train_model.py` script automatically:
  - Pulled `ai4i2020.csv` directly from LocalStack S3.
  - Constructed a stratified dataset using pandas and Scikit-Learn.
  - Trained a robust Random Forest classifier for predicting the 'Machine failure' category.
  - Accurately assessed the system achieving **~98% Accuracy** on unseen sensor combinations.
  - Saved the trained weights successfully as `model.pkl`.

### 3. Live Streaming Sensor Data
- The `sensor_stream.py` is actively iterating through the historical records. 
- It packages row readings (Air/Process Temp, Speed, Torque, Tool wear) into JSON payloads.
- It is simulating a 2-second IoT burst stream by dumping `.json` files into `live_data/` and pushing them simultaneously to the `live_stream/` prefix within the LocalStack S3 bucket.

### 4. Streamlit Dashboard
- **Active Right Now:** A Streamlit server is continuously querying the incoming stream telemetry.
- It utilizes `model.pkl` to render real-time Health Status predictions.
- **Red Alert Notifications:** The interface is coded to trigger alerts whenever the model spots an active threshold for imminent machine failure based on torque shifts and temperatures.

## Next Step
- The Streamlit Dashboard should open automatically in your browser (typically `http://localhost:8501`).
- You can watch the metrics update every few seconds along with the predicted model outputs seamlessly!
