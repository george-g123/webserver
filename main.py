from fastapi import FastAPI
from src.newcomplexnewton import ComplexNewton

from src.MathParser.lexer import Lexer
from src.MathParser.parser_ import Parser

# app = FastAPI()

# @app.get("/")
# async def main():
#     cn = ComplexNewton()
#     cn.setFunctions("z^3-1")
#     delta = cn.run()
#     return {"message": "Hello world!", "runtime": delta}

def start():
    lx = Lexer("e^z+i")
    tokens = lx.generateTokens()
    pr = Parser(tokens)
    tree = pr.parse()
    print(tree)

if __name__ == "__main__":
    start()