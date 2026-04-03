# AWS Predictive Maintenance Local Simulation

This project simulates an AWS-based predictive maintenance system using LocalStack to mock S3, Scikit-learn for machine learning, and Streamlit for the dashboard.

## Prerequisites
- Docker Desktop installed and running.
- Python 3.8+ installed.

## Setup Instructions

### 1. Install Dependencies
Run this command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Start LocalStack (Mock AWS)
Ensure Docker is running, then start the LocalStack container:
```bash
docker-compose up -d
```

### 3. Initialize Environment & Upload Data
Create the S3 bucket and upload the historical dataset:
```bash
python init_env.py
```

### 4. Train the ML Model
Generate the `model.pkl` file by training the Random Forest classifier:
```bash
python train_model.py
```

## Running the Simulation

To see the project in action, you need to run the sensor stream and the dashboard simultaneously.

### 5. Start IoT Sensor Stream
In a new terminal, start the simulated IoT data stream:
```bash
python sensor_stream.py
```
*This will generate a new JSON reading every 2 seconds.*

### 6. Launch Dashboard
In another terminal, start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
*The dashboard will open in your browser (usually at http://localhost:8501).*

## Project Structure
- `ai4i2020.csv`: The UCI Predictive Maintenance Dataset.
- `docker-compose.yml`: LocalStack configuration.
- `init_env.py`: Script to setup the local S3 bucket.
- `train_model.py`: Downloads data from S3 and trains the RF model.
- `sensor_stream.py`: Simulates live IoT data.
- `dashboard.py`: Streamlit frontend for monitoring and alerts.
- `model.pkl`: The trained ML model (generated after training).
- `live_data/`: Temporary folder where "live" JSON payloads are stored for the dashboard.
