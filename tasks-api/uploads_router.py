from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Any
from database.enums import TaskCategoryEnum
import httpx


URL = "http://example.com"

router = APIRouter(prefix="/upload")

class TaskRequestModel(BaseModel):
    assigned_user_id: Optional[int]
    category: TaskCategoryEnum
    data_json: dict[str, Any]


@router.post("/image")
async def upload_task_with_image(file: UploadFile = File(...)):
    

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