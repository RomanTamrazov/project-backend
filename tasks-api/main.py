from fastapi import FastAPI
from uploads_router import router as uploads
from list_router import router as list_router
import uvicorn


app = FastAPI()
app.include_router(uploads, prefix="/task")
app.include_router(list_router, prefix="/task")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
