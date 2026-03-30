from dataclasses import dataclass
from typing import Any

from llvmlite.ir import values

@dataclass
class NumberNode:
    value : str    
    def __repr__(self):
        return f"{self.value}"

@dataclass
class AddNode:
    nodeA : Any
    nodeB : Any
    def __repr__(self):
        return f"({self.nodeA}+{self.nodeB})"

@dataclass
class SubtractNode:
    nodeA : Any
    nodeB : Any
    def __repr__(self):
        return f"({self.nodeA}-{self.nodeB})"

@dataclass
class MultiplyNode:
    nodeA : Any
    nodeB : Any
    def __repr__(self):
        return f"({self.nodeA}*{self.nodeB})"

@dataclass
class DivideNode:
    nodeA : Any
    nodeB : Any
    def __repr__(self):
        return f"({self.nodeA}/{self.nodeB})"

@dataclass
class ExpNode:
    nodeA : Any
    nodeB : Any
    def __repr__(self):
        return f"({self.nodeA}**{self.nodeB})"
    
@dataclass
class VarNode:
    value : str
    def __repr__(self):
        return f"{self.value}"

@dataclass
class UnaryPlus:
    node : Any
    def __repr__(self):
        return f"(+{self.node})"

@dataclass 
class UnaryMinus:
    node : Any
    def __repr__(self):
        return f"(-{self.node})"