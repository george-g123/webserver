# from fastapi import FastAPI
# from src.newcomplexnewton import ComplexNewton

# app = FastAPI()


# @app.get("/")
# async def main():
#     cn = ComplexNewton()
#     delta = cn.run()
#     return {"message": "Hello world!", "runtime": delta}

from src.MathParser.lexer import Lexer
from src.MathParser.parser_ import Parser

def start():
    text = ".0"
    lexer = Lexer(text)
    tokens = lexer.generateTokens()
    parser = Parser(tokens)
    tree = parser.parse()
    print(tree)

if __name__ == "__main__":
    start()