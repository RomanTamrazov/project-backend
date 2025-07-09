from fastapi import FastAPI
# from faststream import FastStream, Context
# from faststream.kafka import KafkaBroker
from aiokafka import AIOKafkaConsumer
from typing import Any
import uvicorn
import asyncio

# broker = KafkaBroker(bootstrap_servers="kafka:9092")
# stream = FastStream(broker=broker)

app = FastAPI()

# @broker.subscriber("tasks")
# async def process_task(msg: str, message=Context()):
#     print(f"Received test message: {msg[:100]}...")
#     await message.ack()  # Manual acknowledgement

consumer = None

async def get_consumer():
    global consumer
    if consumer is None:
        consumer = AIOKafkaConsumer(
            "tasks",
            bootstrap_servers="kafka:9092",
            auto_offset_reset="earliest",
            group_id="task-processor",
        )
        await consumer.start()
    return consumer

# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(process_messages())  # Run in background

@app.on_event("shutdown")
async def shutdown_event():
    if consumer:
        await consumer.stop()

# @app.on_event("startup")
# async def startup():
#     await broker.start()

# @app.on_event("shutdown")
# async def shutdown():
#     await broker.close()

async def has_messages(timeout: float = 0.01) -> bool:
    try:
        consumer = await get_consumer()
        msg = await asyncio.wait_for(consumer.getone(), timeout=timeout)
        # Put the message back in the queue (seek to previous offset)
        await consumer.seek(msg.topic, msg.partition, msg.offset)
        return True
    except (asyncio.TimeoutError, StopAsyncIteration):
        return False


async def process_messages():
    consumer = await get_consumer()
    try:
        if await has_messages():
            msg = await consumer.getone()
            print(f"Consumed: {msg.value.decode()}")
    except Exception as e:
        print(f"Consumer error: {e}")


@app.get("/")
async def test():
    await process_messages()
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)