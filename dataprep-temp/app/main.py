import multiprocessing as mp

import uvicorn
from api import v1
from core import config
from fastapi import FastAPI

app = FastAPI(
    title=config.PROJECT_TITLE, description=config.PROJECT_DESC, docs_url="/api/docs"
)

app.include_router(v1.router)


@app.get("/")
async def root():
    return config.PROJECT_TITLE + " API is Live..."


if __name__ == "__main__":
    mp.freeze_support()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
