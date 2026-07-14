"""Glue automation helpers.

Creates AWS Glue job definitions and triggers. Designed to work with real AWS
or LocalStack (use --endpoint-url to point to LocalStack).

Usage examples:
  python scripts/glue_automation.py --create-job --job-name glue-etl-job --script-location s3://my-bucket/scripts/glue_job.py
  python scripts/glue_automation.py --create-trigger --job-name glue-etl-job --trigger-name daily-trigger --cron 'cron(0 2 * * ? *)'

Note: This script only creates jobs/triggers via boto3; ensure AWS credentials
or LocalStack endpoint are provided in the environment.
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Optional

import boto3


def client(service: str, endpoint_url: Optional[str] = None):
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client(service, **kwargs)


def create_glue_job(job_name: str, script_location: str, role: str = "AWSGlueServiceRole", endpoint_url: Optional[str] = None):
    glue = client("glue", endpoint_url)
    try:
        # Basic job definition; users can adapt this to Glue version, arguments, etc.
        resp = glue.create_job(
            Name=job_name,
            Role=role,
            Command={"Name": "pythonshell", "ScriptLocation": script_location},
            GlueVersion="3.0",
            MaxRetries=0,
            NumberOfWorkers=2,
            WorkerType="Standard",
        )
        print(json.dumps(resp, indent=2, default=str))
    except glue.exceptions.AlreadyExistsException:
        print(f"Glue job '{job_name}' already exists")


def create_trigger(trigger_name: str, job_name: str, cron_expression: str, endpoint_url: Optional[str] = None):
    glue = client("glue", endpoint_url)
    try:
        resp = glue.create_trigger(
            Name=trigger_name,
            Type="SCHEDULED",
            Schedule=cron_expression,
            Actions=[{"JobName": job_name}],
            StartOnCreation=True,
        )
        print(json.dumps(resp, indent=2, default=str))
    except glue.exceptions.AlreadyExistsException:
        print(f"Trigger '{trigger_name}' already exists")


def start_job_run(job_name: str, endpoint_url: Optional[str] = None):
    glue = client("glue", endpoint_url)
    resp = glue.start_job_run(JobName=job_name)
    print(json.dumps(resp, indent=2, default=str))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-url", default=None)
    p.add_argument("--create-job", action="store_true")
    p.add_argument("--create-trigger", action="store_true")
    p.add_argument("--start-job", action="store_true")
    p.add_argument("--job-name")
    p.add_argument("--script-location")
    p.add_argument("--role", default="AWSGlueServiceRole")
    p.add_argument("--trigger-name")
    p.add_argument("--cron")
    return p.parse_args()


def main():
    args = parse_args()
    if args.create_job:
        if not args.job_name or not args.script_location:
            print("--job-name and --script-location required for --create-job", file=sys.stderr)
            sys.exit(1)
        create_glue_job(args.job_name, args.script_location, args.role, args.endpoint_url)

    if args.create_trigger:
        if not args.trigger_name or not args.job_name or not args.cron:
            print("--trigger-name, --job-name and --cron required for --create-trigger", file=sys.stderr)
            sys.exit(1)
        create_trigger(args.trigger_name, args.job_name, args.cron, args.endpoint_url)

    if args.start_job:
        if not args.job_name:
            print("--job-name required for --start-job", file=sys.stderr)
            sys.exit(1)
        start_job_run(args.job_name, args.endpoint_url)


if __name__ == "__main__":
    main()
