import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config

# S3 bucket name for MEVA dataset
BUCKET_NAME = "mevadata-public"  # Example bucket name; replace if different
LOCAL_DOWNLOAD_DIR = "/mnt/SSD2/MEVA_dataset"  # Local directory to save downloaded files

def download_meva_dataset():
    # Create local directory if it doesn't exist
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

    # Create an S3 client without authentication (public bucket)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # List all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                local_path = os.path.join(LOCAL_DOWNLOAD_DIR, key)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Download file
                print(f"Downloading: {key}")
                s3.download_file(BUCKET_NAME, key, local_path)

    print("✅ MEVA dataset download complete.")

if __name__ == "__main__":
    try:
        download_meva_dataset()
    except Exception as e:
        print(f"❌ Error: {e}")
