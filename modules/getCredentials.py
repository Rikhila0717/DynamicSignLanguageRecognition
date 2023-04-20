import os
from dotenv import load_dotenv
from s3fs.core import S3FileSystem


load_dotenv('credentials.env')


AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3=S3FileSystem(key=AWS_ACCESS_KEY_ID,secret=AWS_SECRET_ACCESS_KEY)
