from dataclasses import dataclass
from enum import Enum
from typing import Any

class TokenType(Enum):
    NUMBER = 0
    PLUS = 1
    MINUS = 2
    MULTIPLY = 3
    DIVIDE = 4
    LPAREN = 5
    RPAREN = 6
    EXP = 7
    VAR = 8

@dataclass
class Token:
    type: TokenType
    value : str

    def __repr__(self):
        return self.type.name + (f":{self.value}" if self.value != None else "")