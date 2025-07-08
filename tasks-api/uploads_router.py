from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, Annotated, Union
from database.enums import TaskCategoryEnum
from database.controller import create_task
import httpx
import json


URL = "http://upload-api:5050"

router = APIRouter(prefix="/upload")

class TaskRequestModel(BaseModel):
    assigned_user_id: Optional[int] = None
    category: TaskCategoryEnum
    data_json: dict[str, Any]
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

async def get_task_data(task: str = Form(...)):
    return TaskRequestModel(**json.loads(task))

@router.post("/image")
async def upload_task_with_image(task: TaskRequestModel = Depends(get_task_data), file: UploadFile = File(...),
                                second_file: Annotated[Union[UploadFile, None], File()] = None):
    upload_url = f"{URL}/upload/images"
    file_bytes = await file.read()
    file_key_2 = None
    files = {
        "file": (file.filename, file_bytes, file.content_type)
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url=upload_url, files=files)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code)
        file_key_1 = response.json()["file_key"]
    if second_file:
        second_file_bytes = await second_file.read()
        files_2 = {
            "file": (second_file.filename, second_file_bytes,
                    second_file.content_type)
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url=upload_url, files=files_2)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code)
            file_key_2 = response.json()["file_key"]
    await create_task(task_category=task.category,
                     data_json=task.data_json,
                     user_id=task.assigned_user_id,
                     file_key_1=file_key_1, file_key_2=file_key_2 if file_key_2 else None)


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