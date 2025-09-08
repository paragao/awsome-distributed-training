# Managed Tiered Checkpointing
A high-performance, tiered storage library for distributed checkpointing that enables efficient checkpoint management across multiple storage tiers including in-memory, and Amazon S3.

## Overview

The amzn-sagemaker-checkpointing library provides seamless integrations with different checkpointing solutions of distributed training frameworks:

    Tiered Storage Architecture: Automatic management across in-memory, and S3 storage tiers
    Frameworks supported: Pytorch DCP
    High Performance: Optimized for large-scale distributed training workloads
    Fault Tolerance: Automatic fallback mechanisms and consistency guarantees
    Flexible Configuration: Customizable storage policies
    Logging: Structured logging with rank, step, and operation details

## Key Features
### Tiered Storage Strategy

    In-Memory Tier: Ultra-fast checkpoint storage for immediate access
    S3 Tier: Durable cloud storage for long-term checkpoint retention

### Intelligent Fallback

    Automatic fallback from in-memory to S3 when memory reads fail
    Consistency guarantees across storage tiers
    Graceful degradation under failure conditions

## Pre-Requisites
You need to install the pre-requisites from the `requirements.txt` file. To do so run the command `pip install -r requirements.txt`. 

If you are working with Amazon S3 buckets, for the persistent storage tier, in the same account as your Sagemaker Hyperpod EKS cluster then you need to add the following IAM Policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "s3:DeleteObject",
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::<bucket_name>",
                "arn:aws:s3:::<bucket_name>/*"
            ],
            "Effect": "Allow"
        }
    ]
}
```

Or, if you are working with Amaozn S3 buckets on a different account, then you need to attached the policy below: 
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CheckPointCrossAccountAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": <AWS principal>
            },
            "Action": [
                "s3:DeleteObject",
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::<bucket_name>",
                "arn:aws:s3:::<bucket_name>/*"
            ]
        }
    ]
}
```

On both of them, please make sure you change the `bucket_name` and `AWS Principal` on the policies above.

## Running the example
To run the example, use the training script in your job definition file (YAML).

