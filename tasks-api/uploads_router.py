from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import httpx


URL = "http://example.com"

router = APIRouter(prefix="/upload")

class TaskRequestModel(BaseModel):
    

# @router.post("/image")
# async def upload_image(file: UploadFile = File(...)):
#     file_bytes = await file.read()
#     files = {
#         "file": (file.filename, file_bytes, file.content_type)
#     }
#     async with httpx.AsyncClient() as client:
#         response = await client.post(url=f"{URL}/upload/images", files=files)
    
#     return {
#         "status_code": response.status_code,
#         "response": response.json() if response.headers.get("content-type") == "application/json" \
#              else response.text
#     }

# @router.post("/text")
# async def upload_image(file: UploadFile = File(...)):
#     file_bytes = await file.read()
#     files = {
#         "file": (file.filename, file_bytes, file.content_type)
#     }
#     async with httpx.AsyncClient() as client:
#         response = await client.post(url=f"{URL}/upload/text", files=files)
    
#     return {
#         "status_code": response.status_code,
#         "response": response.json() if response.headers.get("content-type") == "application/json" \
#              else response.text
#     }