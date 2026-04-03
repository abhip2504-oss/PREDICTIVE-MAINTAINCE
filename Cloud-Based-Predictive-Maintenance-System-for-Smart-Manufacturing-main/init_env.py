import os
import boto3
from botocore.exceptions import ClientError

# Configuration for LocalStack S3 mock
AWS_ENDPOINT_URL = "http://localhost:4566"
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
BUCKET_NAME = "factory-data"
FILE_NAME = "ai4i2020.csv"

def get_s3_client():
    """Create a boto3 client pointing to LocalStack."""
    return boto3.client(
        's3',
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def init_environment():
    """Initializes the S3 bucket and uploads the AI4I dataset."""
    s3 = get_s3_client()
    
    # Create the data lake bucket
    try:
        print(f"Attempting to create S3 bucket: '{BUCKET_NAME}' at {AWS_ENDPOINT_URL}...")
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"[*] Bucket '{BUCKET_NAME}' created successfully.")
    except ClientError as e:
        if e.response['Error']['Code'] in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
            print(f"[*] Bucket '{BUCKET_NAME}' already exists.")
        else:
            print(f"[!] Error creating bucket: {e}")
            raise e

    # Verify local file presence
    if not os.path.exists(FILE_NAME):
        print(f"[!] Critical Error: Could not find {FILE_NAME} in the current directory.")
        print("Please ensure the dataset file is in the same directory as this script.")
        return

    # Upload historical dataset
    print(f"Uploading '{FILE_NAME}' as historical batch data lake...")
    try:
        s3.upload_file(FILE_NAME, BUCKET_NAME, FILE_NAME)
        print("[*] Upload successful! Environment is fully setup.")
    except Exception as e:
        print(f"[!] Upload failed: {e}")
        raise e

if __name__ == "__main__":
    init_environment()
