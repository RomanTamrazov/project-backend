import asyncio
from datetime import datetime
from database.dao.basedao import BaseDAO
from database.dao.session_maker import connection
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, create_model
from typing import List, Optional
from database.pydantic_schemes import TaskModel


class TaskDAO(BaseDAO[Task]):
    model = Task


default_order_lmbd = lambda a: a.created_at.desc()


@connection(commit=False)
async def get_tasks(session: AsyncSession, user_id: int | None = None,
                    task_category: TaskCategoryEnum | None = None,
                    limit: int | None = None,
                    offset: int = 0):
    data = await TaskDAO.find_all(session=session, filters=TaskModel(
        assigned_user_id=user_id,
        category=task_category
    ), order_by_lmbd=default_order_lmbd, limit=limit, offset=offset)
    validated_data = [TaskModel.model_validate(i) for i in data]
    return validated_data

@connection(commit=True)
async def create_task(session: AsyncSession, user_id: int, task_category: TaskCategoryEnum):
    await TaskDAO.add(session=session, values=TaskModel(
        assigned_user_id = user_id,
        category = task_category
    ))
