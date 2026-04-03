import pandas as pd
import boto3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import io

# LocalStack S3 configuration
AWS_ENDPOINT_URL = "http://localhost:4566"
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
BUCKET_NAME = "factory-data"
FILE_NAME = "ai4i2020.csv"
MODEL_NAME = "model.pkl"

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def train_and_save_model():
    print(f"Reading local {FILE_NAME}...")
    try:
        df = pd.read_csv(FILE_NAME)
    except Exception as e:
        print(f"Failed to load data locally: {e}")
        return
    

    print("Data loaded perfectly. Preprocessing...")
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Machine failure'
    
    X = df[features]
    y = df[target]
    
    # Stratified split to handle potential imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print(f"Saving trained model as {MODEL_NAME}...")
    joblib.dump(model, MODEL_NAME)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
