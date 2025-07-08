from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from database.enums import TaskCategoryEnum


class TaskModel(BaseModel):
    assigned_user_id: int
    category: TaskCategoryEnum
    data_json: str

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)