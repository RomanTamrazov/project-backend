from fastapi import FastAPI
from uploads_router import router as uploads
import uvicorn


app = FastAPI()
app.include_router(uploads, prefix="/task")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
