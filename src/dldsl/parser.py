
from lark import Lark


dl_parser = Lark(r"""
    EQUALS: "="
    OPERATOR: /[\+\-\*\/]/
    ALPHANUMERIC: /[a-zA-Z0-9_]+/
    
    
    statement: tensor EQUALS expr
    expr: tensor (operator tensor)*
    operator: OPERATOR 
    tensor: ALPHANUMERIC index 
    index: "[" [axis ("," axis)*] "]"
    axis: ALPHANUMERIC | INT 

    %import common.NEWLINE
    %import common.INT
    %import common.FLOAT
    %import common.WS
    %ignore WS
    """, 
    start='statement'
)


def parse(s: str):
    return dl_parser.parse(s)
