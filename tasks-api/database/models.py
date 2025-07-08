from datetime import datetime
from typing import List, Optional
from sqlalchemy import BigInteger, Column, String, ForeignKey, Table, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine

engine = create_async_engine('sqlite+aiosqlite:///db.sqlite3', echo=True)

async_session = async_sessionmaker(engine)

class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


async def async_main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
