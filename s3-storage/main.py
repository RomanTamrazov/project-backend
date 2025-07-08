from aiobotocore.session import get_session
from contextlib import asynccontextmanager
import asyncio
import mimetypes
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

import uvicorn


class S3Client:
    def __init__(
        self,
        access_key: str,
        secret_key: str,
        endpoint_url: str,
        bucket_name: str
    ):
        self.config = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url
        }
        self.bucket_name = bucket_name
        self.session = get_session()
    
    @asynccontextmanager
    async def get_client(self):
        async with self.session.create_client("s3", **self.config) as client:
            yield client
        
    async def upload_file(
        self,
        file_path: str,
        object_name: str
    ):
        async with self.get_client() as client:
            with open(file_path, 'rb') as file:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=file
                )
    
    async def upload_file_bin(
        self,
        content: bytes,
        object_name: str,
        content_type: str,
    ):
        async with self.get_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=content,
                ContentType=content_type
            )
    
    async def get_file_stream(self, object_name: str, chunk_size: int = 1024*1024):
        async with self.get_client() as client:
            try:
                response = await client.get_object(
                    Bucket=self.bucket_name,
                    Key=object_name
                )
                async with response['Body'] as stream:
                    file_content = await stream.read()
                    return file_content
            except client.exceptions.NoSuchKey:
                return None
    
    async def ensure_bucket_exists(self):
        async with self.get_client() as client:
            try:
                response = await client.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully.")
                print(response)
            except client.exceptions.BucketAlreadyOwnedByYou:
                print(f"Bucket '{self.bucket_name}' already exists and is owned by you.")
            except Exception as e:
                print(f"Error creating bucket '{self.bucket_name}': {e}")



ACCESS_KEY = "testuser"
SECRET_KEY = "minio123"
ENDPOINT_URL = "http://minio:9000"


def get_content_type(filename: str, fallback="application/octet-stream") -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or fallback

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    s3_client_1 = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        bucket_name="images" 
    )
    await s3_client_1.ensure_bucket_exists()
    s3_client_2 = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        bucket_name="text" 
    )
    await s3_client_2.ensure_bucket_exists()


@app.post("/upload/{bucket_name}")
async def upload_file_route(bucket_name: str, file: UploadFile = File(...)):
    s3_client = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        bucket_name=bucket_name
    )
    content_type = file.content_type or get_content_type(file.filename)
    await s3_client.upload_file_bin(
        content=await file.read(),
        object_name=file.filename,
        content_type=content_type
    )
    return {"file_key": file.filename}


@app.get("/get/{bucket_name}/{object_name}")
async def get_file_route(bucket_name: str, object_name: str):
    s3_client = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        bucket_name=bucket_name
    )
    file_stream = await s3_client.get_file_stream(object_name=object_name)
    if file_stream is None:
        raise HTTPException(status_code=404, detail="Not found")
    buffer = BytesIO(file_stream)
    content_type = "application/octet-stream"
    return StreamingResponse(content=buffer,
                             media_type=content_type,
                             headers={"Content-Disposition": f"attachment; filename={object_name}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
