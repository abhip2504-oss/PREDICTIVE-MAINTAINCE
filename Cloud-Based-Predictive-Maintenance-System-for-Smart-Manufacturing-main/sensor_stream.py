import pandas as pd
import time
import json
import boto3
import os
import uuid

# Configuration
AWS_ENDPOINT_URL = "http://localhost:4566"
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
BUCKET_NAME = "factory-data"
FILE_NAME = "ai4i2020.csv"
STREAM_PREFIX = "live_stream/"
LOCAL_DIR = "live_data"

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def stream_data():
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        
    s3 = get_s3_client()
    print("Loading data for simulation...")
    try:
        df = pd.read_csv(FILE_NAME)
    except FileNotFoundError:
        print(f"Error: {FILE_NAME} not found locally.")
        return

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    print("Starting IoT sensor stream simulation...")
    for index, row in df.iterrows():
        payload = {
            "timestamp": int(time.time()),
            "sensor_id": f"sensor-{row.get('Product ID', 'unknown')}",
        }
        for f in features:
            payload[f] = row[f]
        
        json_data = json.dumps(payload)
        file_id = str(uuid.uuid4())
        local_filepath = os.path.join(LOCAL_DIR, f"{file_id}.json")
        s3_key = f"{STREAM_PREFIX}{file_id}.json"
        
        # Save locally for Streamlit
        with open(local_filepath, "w") as f:
            f.write(json_data)
            
        # Upload to LocalStack S3
        try:
            s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=json_data)
            print(f"Streamed -> S3 + Local: {payload}")
        except Exception as e:
            print(f"Failed to stream to S3: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    stream_data()
