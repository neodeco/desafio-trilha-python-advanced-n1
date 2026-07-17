"""Simple monitoring for Glue job runs.

Polls Glue job runs and writes a short status log. Works with AWS or LocalStack.

Usage:
  python scripts/monitor_glue_jobs.py --job-name glue-etl-job --interval 30
"""
from __future__ import annotations
import argparse
import time
from typing import Optional

import boto3


def client(service: str, endpoint_url: Optional[str] = None):
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client(service, **kwargs)


def monitor(job_name: str, interval: int = 30, endpoint_url: Optional[str] = None):
    glue = client("glue", endpoint_url)
    print(f"Monitoring Glue job '{job_name}' every {interval}s. Ctrl-C to stop.")
    try:
        while True:
            runs = glue.get_job_runs(JobName=job_name, MaxResults=5)
            for r in runs.get("JobRuns", []):
                print(f"RunId={r['Id']} State={r['JobRunState']} Started={r.get('StartedOn')} Completed={r.get('CompletedOn')}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped monitoring")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", required=True)
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--endpoint-url", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    monitor(args.job_name, args.interval, args.endpoint_url)


if __name__ == "__main__":
    main()
