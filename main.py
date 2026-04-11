from fastapi import FastAPI
from src.newcomplexnewton import ComplexNewton

app = FastAPI()

@app.get("/")
async def main():
    cn = ComplexNewton()
    cn.setFunctions("z^3-1")
    delta = cn.run()
    return {"message": "Hello world!", "runtime": delta}