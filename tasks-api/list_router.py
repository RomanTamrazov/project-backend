from fastapi import APIRouter
from typing import Optional
from database.controller import get_tasks


router = APIRouter()


@router.get("/list")
async def list_tasks(id: int | None = None, limit: int | None = None, offset: int = 0):
    tasks = await get_tasks(task_id=id, limit=limit, offset=offset)
    return tasks