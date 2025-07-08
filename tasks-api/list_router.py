from fastapi import APIRouter
from typing import Optional
from database.controller import get_tasks


router = APIRouter()


@router.get("/list")
async def list_tasks(id: int | None = None):
    tasks = await get_tasks(task_id=id)
    return tasks