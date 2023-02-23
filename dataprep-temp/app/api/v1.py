import multiprocessing as mp

from fastapi import APIRouter
from qpiai_data_prep.config import RequestModel
from qpiai_data_prep.process import process_main

r = router = APIRouter()


@r.post("/api", status_code=202)
async def qpiai_api_router(req_args: RequestModel):
    p = mp.Process(target=process_main, args=[req_args.dict()])
    p.start()
    return "Accepted"
