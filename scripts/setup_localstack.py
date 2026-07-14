import os
import shutil
from typing import Optional

import boto3
from botocore.config import Config


def ensure_localstack_services(endpoint_url: Optional[str] = None) -> None:
    endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    s3 = session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))
    sqs = session.client("sqs", endpoint_url=endpoint_url)
    dynamodb = session.client("dynamodb", endpoint_url=endpoint_url)

    for bucket_name in ["raw-data", "processed-data"]:
        try:
            s3.create_bucket(Bucket=bucket_name)
        except Exception:
            pass

    try:
        sqs.create_queue(QueueName="raw-data-queue")
    except Exception:
        pass

    try:
        dynamodb.create_table(
            TableName="etl-metadata",
            AttributeDefinitions=[{"AttributeName": "job_name", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "job_name", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
    except Exception:
        pass


if __name__ == "__main__":
    ensure_localstack_services()
