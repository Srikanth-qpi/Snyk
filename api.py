from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional



app = FastAPI()

@app.get("/")
async def get():
    return {"message":"This is met"}

@app.post("/courses")
async def post():
    return {"message":"This is a post method"}

@app.put("/")
async def put():
    return {"message": "This is a put method"}

@app.get("/items")
async def get_items():
    return {"message":"This is a items route"}

@app.get("/items/{item_id}")
async def get_items(item_id:int):
    return {"item_id":item_id}

class Item(BaseModel):
    name:str
    description:Optional[str] = None
    price:float
    tax:Optional[float] = None

@app.post("/items")
async def create_item(item:Item):
    item_dict = item.dict()
    if item_dict["tax"]:
        price_with_tax = item_dict["tax"] + item_dict["price"]
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

