
from lark import Lark


dl_parser = Lark(r"""
    EQUALS: "="

    ALPHANUMERIC: /[a-zA-Z0-9_]+/
    
    statement: tensor EQUALS expr

    expr: atom | sum | product

    sum: (product | atom) addsub+

    addsub: "+" (product | atom)
          | "-" (product | atom) -> neg

    product: atom divmul+

    divmul: "*" atom 
          | "/" atom -> inv

    atom: tensor           
        | "-" tensor -> neg
        | "(" expr ")"
    
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
