import asyncio
from datetime import datetime
from database.dao.basedao import BaseDAO
from database.dao.session_maker import connection
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, create_model
from typing import List, Optional


class UserDAO(BaseDAO[User]):
    model = User


class TaskDAO(BaseDAO[Task]):
    model = Task


default_order_lmbd = lambda a: a.created_at.desc()
