import os
from dotenv import load_dotenv
from s3fs.core import S3FileSystem
# import boto3
# from getCredentials import AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY

load_dotenv('credentials.env')


AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3=S3FileSystem(key=AWS_ACCESS_KEY_ID,secret=AWS_SECRET_ACCESS_KEY)
# s3 = boto3.resource(
#     service_name='s3',
#     region_name='us-east-2',
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY
# )
# for bucket in s3.buckets.all():
#     print(bucket.name)