from .tokens import TokenType
from .nodes import *

class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.advance()

    def advance(self):
        try:
            self.currentToken = next(self.tokens)
        except StopIteration:
            self.currentToken = None

    def raiseError(self):
        raise Exception("Invalid Syntax")

    def parse(self):
        if self.currentToken == None:
            return None

        result = self.expr()

        if self.currentToken != None:
            self.raiseError()

        return result

    def expr(self):
        result = self.term()

        while self.currentToken != None and self.currentToken.type in (TokenType.PLUS, TokenType.MINUS):
            if self.currentToken.type == TokenType.PLUS:
                self.advance()
                result = AddNode(result, self.term())
            elif self.currentToken.type == TokenType.MINUS:
                self.advance()
                result = SubtractNode(result, self.term())

        return result

    def term(self):
        result = self.exponent()

        while self.currentToken != None and self.currentToken.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            if self.currentToken.type == TokenType.MULTIPLY:
                self.advance()
                result = MultiplyNode(result, self.exponent())
            elif self.currentToken.type == TokenType.DIVIDE:
                self.advance()
                result = DivideNode(result, self.exponent())

        return result

    def exponent(self):
        result = self.factor()

        if self.currentToken != None and self.currentToken.type == TokenType.EXP:
            self.advance()
            result = ExpNode(result, self.exponent()) 

        return result

    def factor(self):
        token = self.currentToken

        if token.type == TokenType.LPAREN:
            self.advance()
            result = self.expr()
            if self.currentToken.type != TokenType.RPAREN:
                self.raiseError()
            self.advance()
            return result

        elif token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value)

        elif token.type == TokenType.VAR:
            self.advance()
            return VarNode(token.value)

        elif token.type == TokenType.PLUS:
            self.advance()
            return UnaryPlus(self.factor())

        elif token.type == TokenType.MINUS:
            self.advance()
            return UnaryMinus(self.factor())

        elif token.type == TokenType.FUNC:
            functionName = token.value
            self.advance()

            if self.currentToken == None or self.currentToken.type != TokenType.LPAREN:
                self.raiseError()

            self.advance()
            functionArgument = self.expr()

            if self.currentToken == None or self.currentToken.type != TokenType.RPAREN:
                self.raiseError()

            self.advance()
            return FunctionNode(functionName, functionArgument)        
        
        self.raiseError()