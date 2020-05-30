from lark import Lark

dl_parser = Lark(r"""
    EQUALS: "="
    OPERATOR: /[\+\-\*\/]/
    ALPHANUMERIC: /[a-zA-Z0-9_]+/

    statement: tensor EQUALS expr
    expr: operand (OPERATOR operand)*
    operand: tensor | INT | FLOAT
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

# Assignment
print(dl_parser.parse("Y[i, k] = W[i, j] * X[j, k]"))
