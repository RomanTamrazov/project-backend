from fastapi import FastAPI
from faststream import FastStream, Context
from faststream.kafka import KafkaBroker
from typing import Any
import uvicorn

broker = KafkaBroker(bootstrap_servers="kafka:9092")
stream = FastStream(broker=broker)

app = FastAPI()

@broker.subscriber("tasks")
async def process_task(msg: str, message=Context()):
    print(f"Received test message: {msg[:100]}...")
    await message.ack()  # Manual acknowledgement

@app.on_event("startup")
async def startup():
    await broker.start()

@app.on_event("shutdown")
async def shutdown():
    await broker.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)