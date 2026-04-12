from .tokens import Token, TokenType
WHITESPACE = " \n\t"
DIGITS = "0123456789"

class Lexer:
    def __init__(self, text : str):
        self.text = iter(text)
        self.advance()

    def advance(self):
        try:
            self.currentChar = next(self.text)
        except StopIteration:
            self.currentChar = None
        
    def generateTokens(self):
        while self.currentChar != None:
            if self.currentChar in WHITESPACE:
                self.advance()
            elif self.currentChar == "." or self.currentChar in DIGITS:
                yield self.generateNumber()
            elif self.currentChar == "+":
                self.advance()
                yield Token(TokenType.PLUS, "+")
            elif self.currentChar == "-":
                self.advance()
                yield Token(TokenType.MINUS, "-")
            elif self.currentChar == "*":
                self.advance()
                yield Token(TokenType.MULTIPLY, "*")
            elif self.currentChar == "/":
                self.advance()
                yield Token(TokenType.DIVIDE, "/")
            elif self.currentChar == "^":
                self.advance()
                yield Token(TokenType.EXP, "**")
            elif self.currentChar == "(":
                self.advance()
                yield Token(TokenType.LPAREN, "(")
            elif self.currentChar == ")":
                self.advance()
                yield Token(TokenType.RPAREN, ")")
            elif self.currentChar.isalnum():
                yield self.generateFunctions()
            else:
                raise Exception(f"Illegal Character: '{self.currentChar}'")

    def generateNumber(self):
        numberBuffer : str = self.currentChar
        decimalCounter : int = 0
        self.advance()

        while self.currentChar != None and (self.currentChar == "." or self.currentChar in DIGITS):
            if self.currentChar == ".":
                decimalCounter += 1
                if decimalCounter > 1:
                    break
            
            numberBuffer += self.currentChar
            self.advance()

        if numberBuffer.startswith("."):
            numberBuffer = "0" + numberBuffer

        if numberBuffer.endswith("."):
            numberBuffer += "0"

        return Token(TokenType.NUMBER, numberBuffer)

    def generateFunctions(self):
        functionBuffer : str = ""

        while self.currentChar != None and self.currentChar.isalnum():
            functionBuffer += self.currentChar
            self.advance()
        
        functionBuffer = functionBuffer.lower()

        if functionBuffer == "z":
            return Token(TokenType.VAR, "z")
        elif functionBuffer == "i":
            return Token(TokenType.NUMBER, "1j")
        elif functionBuffer == "e":
            return Token(TokenType.VAR, "e")
        elif functionBuffer in ["sin", "cos", "tan", "sinh", "cosh", "exp", "log", "sqrt"]:
            return Token(TokenType.FUNC, functionBuffer)
        else:
            raise Exception(f"Unknown function: {functionBuffer}")