variables = {}
import operator
class InstructionList:
    def __init__(self, children=None):
        if children is None:
            children = []
        self.children = children

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def eval(self):
        """
        Evaluates all the class children and returns the result
        of their eval method in a list or returns an ExitStatement
        in case one is found
        """

        ret = []
        for n in self:
            res = n.eval()
            #print(res," res")
            ret.append((res))
        return ret


class BaseExpression:
    def eval(self):
        raise NotImplementedError()


class Primitive(BaseExpression):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '<Primitive "{0}"({1})>'.format(self.value, self.value.__class__)

    def eval(self):
        return self.value

class Identifier(BaseExpression):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Identifier {0}>'.format(self.name)

    def assign(self, val):
        variables[self.name] = val

    def eval(self):
        return variables[self.name]

class Array(BaseExpression):
    def __init__(self, values: InstructionList):
        self.values = values

    def __repr__(self):
        return '<Array len={0} items={1}>'.format(len(self.values.children), self.values)

    def eval(self):
        return self.values.eval()

class ArrayAccess(BaseExpression):
    def __init__(self, array: Identifier, index: BaseExpression):
        self.array = array
        self.index = index

    def __repr__(self):
        return '<Array index={0}>'.format(self.index)

    def eval(self):
        try:
            return self.array.eval()[self.index.eval()]
        except:
            print("SEMANTIC ERROR")

class ArrayAssign(BaseExpression):
    def __init__(self, array: Identifier, index: BaseExpression, value: BaseExpression):
        self.array = array
        self.index = index
        self.value = value

    def __repr__(self):
        return '<Array arr={0} index={1} value={2}>'.format(self.array, self.index, self.value)

    def eval(self):
        try:
            self.array.eval()[self.index.eval()] = self.value.eval()
        except:
            print("SEMANTIC ERROR")
class Assignment(BaseExpression):
    def __init__(self, identifier: Identifier, val):
        self.identifier = identifier
        self.val = val

    def __repr__(self):
        return '<Assignment sym={0}; val={1}>'.format(self.identifier, self.val)

    def eval(self):
        self.identifier.assign(self.val.eval())

class BinaryOperation(BaseExpression):
    __operations = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '**': operator.pow,
        '/': operator.truediv,
        '%': operator.mod,
        '//': operator.floordiv,
        'and': operator.and_,
        'or': operator.or_,
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '<>': operator.ne,
    }

    def __repr__(self):
        return '<BinaryOperation left ={0} right={1} operation="{2}">'.format(self.left, self.right, self.op)

    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def eval(self):
        try:
            left = None
            right = None
            op = self.__operations[self.op]
            left = self.left.eval()
            right = self.right.eval()
            return op(left, right)
        except:
            print("SEMANTIC ERROR")

class UnaryOperation(BaseExpression):
    __operations = {
        'not': operator.not_
    }
    def __repr__(self):
        return '<Unary operation: operation={0} expr={1}>'.format(self.operation, self.expr)

    def __init__(self,expr: BaseExpression):
        self.operation = 'not'
        self.expr = expr

    def eval(self):
        return self.__operations[self.operation](self.expr.eval())

class If(BaseExpression):
    def __init__(self, condition: BaseExpression, truepart: InstructionList, elsepart=None):
        self.condition = condition
        self.truepart = truepart
        self.elsepart = elsepart

    def __repr__(self):
        return '<If condition={0} then={1} else={2}>'.format(self.condition, self.truepart, self.elsepart)

    def eval(self):
         if self.condition.eval():
            return self.truepart.eval()
         elif self.elsepart is not None:
            return self.elsepart.eval()

class While(BaseExpression):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return '<While cond={0} body={1}>'.format(self.condition, self.body)

    def eval(self):
        while self.condition.eval():
            self.body.eval();
class PrintStatement(BaseExpression):
    def __init__(self, items: InstructionList):
        self.items = items

    def __repr__(self):
        return self.eval()

    def eval(self):
        print(*self.items.eval(), end='\n', sep='')

class InExpression(BaseExpression):
    def __init__(self, a: BaseExpression, b: BaseExpression, not_in: bool=False):
        self.a = a
        self.b = b
        self.not_in = not_in

    def __repr__(self):
        return '<In {0} in {1}>'.format(self.a, self.b)

    def eval(self):
        if self.not_in:
            return self.a.eval() not in self.b.eval()
        return self.a.eval() in self.b.eval()

reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'in': 'IN',
    'while': 'WHILE',
    'print': 'PRINT',
    'and': 'AND',
    'or': 'OR',
    'not': 'NOT',
}

tokens = [
    'KEYWORD',
    'STMT_END',
    'EQUALS',
    'IDENTIFIER',
    'NUM_INT',
    'NUM_FLOAT',
    'LPAREN',
    'RPAREN',
    'LBRACK',
    'RBRACK',
    'COMMA',
    'STRING',
    'NEWLINE',
    'LSQBRACK',
    'RSQBRACK',
    'COLON',
    'QUESTION_MARK',

    'PLUS',
    'EXP',
    'MINUS',
    'MUL',
    'DIV',
    'MOD',

    'PLUS_EQ',
    'MINUS_EQ',
    'MUL_EQ',
    'DIV_EQ',
    'MOD_EQ',
    'EXP_EQ',
    'TRUE',
    'FALSE',
    'EQ',
    'NEQ',
    'GT',
    'GTE',
    'LT',
    'LTE',

    'ARROW_LTR',
    'ARROW_RTL'
] + list(reserved.values())

t_COMMA = ','
t_PLUS = r'\+'
t_EXP = r'\*\*'
t_MINUS = '-'
t_MUL = r'\*'
t_DIV = r'/'
t_MOD = '%'
t_STMT_END = ';'
t_EQUALS = '='
t_ignore_WS = r'\s+'
t_COLON = ':'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACK = '{'
t_RBRACK = '}'
t_LSQBRACK = r'\['
t_RSQBRACK = r'\]'
t_EQ = '=='
t_NEQ = '<>'
t_GT = '>'
t_GTE = '>='
t_LT = '<'
t_LTE = '<='





def t_NEWLINE(t):
    r'\n'
    t.lexer.lineno += 1
    t.lexer.linepos = 0
    pass


def t_TRUE(t):
    'True'
    t.value = True
    return t


def t_FALSE(t):
    'False'
    t.value = False
    return t


def t_IDENTIFIER(t):
    '[A-Za-z][A-Za-z0-9_]*'

    t.type = reserved.get(t.value, t.type)

    return t


def t_NUM_FLOAT(t):
    r'\d*\.\d+'
    t.value = float(t.value)
    return t


def t_NUM_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_STRING(t):
    r'("(\\"|[^"])*")|(\'(\\\'|[^\'])*\')'
    result = str(t.value)
    result = result[1:-1]
    t.value = result
    return t


def t_error(t):
    t.lexer.skip(1)
    print(t.value,"SYNTAdwadawdaX ERROR")
    pass
    #return t


# Build the lexer
import ply.lex as lex
lex.lex()
precedence = (
    ('left', 'NOT'),
    ('left','LBRACK'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'MUL', 'DIV'),
    ('left', 'EXP', 'MOD'),
)

start = "block"

def p_block(p):
    '''
    block : LBRACK statement_list RBRACK
          
    '''
    #p[0] = p[2]
    #print(p[1])
    p[0] = InstructionList(p[2])
    #print(p[0])


def p_statement_list(p):
    '''
    statement_list : statement
                   | statement_list statement
                   | LBRACK statement_list RBRACK
                   | statement_list
                   |
    '''
    if len(p) == 2:
        p[0] = InstructionList([p[1]])
    elif len(p) == 1:
        p[0] = InstructionList()
    else:
        p[1].children.append(p[2])
        p[0] = p[1]

def p_statement(p):
    '''
    statement : identifier
              | expression
              | if_statement
    '''
    p[0] = p[1]


def p_identifier(p):
    '''
    identifier : IDENTIFIER
    '''
    p[0] = Identifier(p[1])

def p_primitive(p):
    '''
    primitive : NUM_INT
              | NUM_FLOAT
              | STRING
              | boolean
    '''
    if isinstance(p[1], BaseExpression):
        p[0] = p[1]
    else:
        p[0] = Primitive(p[1])

def p_binary_op(p):
    '''
    expression : expression PLUS expression %prec PLUS
            | expression MINUS expression %prec MINUS
            | expression MUL expression %prec MUL
            | expression DIV expression %prec DIV
            | expression EXP expression %prec EXP
            | expression MOD expression %prec MOD
    '''
    p[0] = BinaryOperation(p[1], p[3], p[2])

def p_unary_operation(p):
    '''
    expression : NOT expression
    '''
    p[0] = UnaryOperation(p[2])

def p_boolean_operators(p):
    '''
    boolean : expression EQ expression
            | expression NEQ expression
            | expression GT expression
            | expression GTE expression
            | expression LT expression
            | expression LTE expression
            | expression AND expression
            | expression OR expression
    '''
    p[0] = BinaryOperation(p[1], p[3], p[2])

def p_paren(p):
    '''
    expression : LPAREN expression RPAREN
    '''
    p[0] = p[2] if isinstance(p[2], BaseExpression) else Primitive(p[2])

def p_boolean(p):
    '''
    boolean : TRUE
            | FALSE
    '''
    p[0] = Primitive(p[1])

def p_assignable(p):
    '''
    assignable : primitive
               | expression
    '''
    p[0] = p[1]

def p_comma_separated_expr(p):
    '''
    arguments : arguments COMMA expression
              | expression
              |
    '''
    if len(p) == 2:
        p[0] = InstructionList([p[1]])
    elif len(p) == 1:
        p[0] = InstructionList()
    else:
        p[1].children.append(p[3])
        p[0] = p[1]

def p_arrays(p):
    '''
    expression : LSQBRACK arguments RSQBRACK
    '''
    p[0] = Array(p[2])

def p_array_access(p):
    '''
    expression : identifier LSQBRACK expression RSQBRACK 
               | expression LSQBRACK expression RSQBRACK 
    '''
    p[0] = ArrayAccess(p[1], p[3])

def p_array_access_assign(p):
    '''
    statement : identifier LSQBRACK expression RSQBRACK EQUALS expression STMT_END
              | expression LSQBRACK expression RSQBRACK EQUALS expression STMT_END  
    '''
    p[0] = ArrayAssign(p[1], p[3], p[6])

def p_assign(p):
    '''
    expression : identifier EQUALS assignable STMT_END
    '''
    p[0] = Assignment(p[1], p[3])

def p_ifstatement(p):
    '''
    if_statement : IF expression LBRACK statement_list RBRACK
    '''
    p[0] = If(p[2], p[4])

def p_ifstatement_else(p):
    '''
    if_statement : IF expression LBRACK statement_list RBRACK ELSE LBRACK statement_list RBRACK
    '''
    p[0] = If(p[2], p[4], p[8])

def p_in_expression(p):
    '''
    expression : expression IN expression
               | expression NOT IN expression
    '''
    if len(p) == 4:
        p[0] = InExpression(p[1], p[3])
    else:
        p[0] = InExpression(p[1], p[4], True)

def p_print_statement(p):
    '''
    statement : PRINT arguments STMT_END
    '''
    p[0] = PrintStatement(p[2])

def p_expression(p):
    '''
    expression : primitive
               | STRING
               | identifier
    '''
    p[0] = p[1]

def p_while_loop(p):
    '''
    statement : WHILE expression LBRACK statement_list RBRACK 
    '''
    print(p[2])
    p[0] = While(p[2], p[4])

def p_error(p):
    print("SYNTAX ERROR")

import ply.yacc as yacc
yacc.yacc()
lines = []
import sys
with open(sys.argv[1],'r') as f:
    lines = f.read()
    #lex.input(lines)
    #while True:
        #tok = lex.token()
        #if not tok:
            #break
        #print(tok)
    result = yacc.parse(lines)
    try:
        for node in result.children:
            #print(node, " = NODE")
            p = node.eval()

    except:
        print("SYNTAX ERROR")



