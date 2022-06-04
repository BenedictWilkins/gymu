
import ast
from types import SimpleNamespace

from numpy import isin

def ast_resolve(expr):
   return _ast_expr_resolver(expr) 

def _ast_expr_resolver(expr):
    if isinstance(expr, (bool, str, int, float)):
        return expr # primitive 
    elif isinstance(expr, ast.Constant):
        return expr.value # primitive
    elif isinstance(expr, ast.Expr):
        return _ast_expr_resolver(expr.value)
    elif isinstance(expr, (ast.BinOp, ast.BoolOp, ast.UnaryOp)):
        return _ast_Operator_resolver(expr)
    elif isinstance(expr, ast.Attribute): # its a fully qualified class/method name
        return _ast_Attribute_resolver(expr)
    elif isinstance(expr, ast.Name):
        return _ast_Name_resolver(expr)
    elif isinstance(expr, ast.Call):
        return _ast_Call_resolver(expr)
    elif isinstance(expr, (ast.List, list)):
        print(expr)
    # TODO dict?
    else:
        raise ValueError(f"Unknown expression: {ast.dump(expr)}")

def _ast_List_resolver(expr):
    if isinstance(expr, ast.List):
        _list = expr.elts
    elif isinstance(expr, list):
        _list = expr
    result = []
    for n in _list:
        if isinstance(n, ast.Starred):
            result.extend(_ast_List_resolver(n.value))
        else:
            result.append(_ast_expr_resolver(n))
    return result

def _ast_Name_resolver(expr):
    assert isinstance(expr, ast.Name) # TODO NameConstant?
    return SimpleNamespace(name = expr.id, args=[], kwargs={})

def _ast_Operator_resolver(expr):
    assert isinstance(expr, (ast.BinOp, ast.BoolOp, ast.UnaryOp))
    walk = iter(ast.walk(expr))
    next(walk) # skip first (this is the operator)
    for n in walk:
        if not isinstance(n, (ast.Constant, ast.operator, ast.cmpop, ast.boolop)):
            raise ValueError(f"Expression could not be evaluated as it contains non-constant values (ast dump: {ast.dump(n)})")
    # compile and evaluate
    expr = ast.Expression(body=expr)
    result = compile(expr, filename="", mode="eval")
    result = eval(result)
    return result

def _ast_Attribute_resolver(expr):
    assert isinstance(expr, ast.Attribute)
    name = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            name.append(node.id)
        elif isinstance(node, ast.Attribute):
            name.append(node.attr)
    name = ".".join(reversed(name))
    return SimpleNamespace(name = name, args=[], kwargs={})

def _ast_Call_resolver(expr):
    assert isinstance(expr, ast.Call)
    name = _ast_expr_resolver(expr.func).name
    args = _ast_List_resolver(expr.args)
    # TODO double starred?
    kwargs = {n.arg:_ast_expr_resolver(n.value) for n in expr.keywords}
    return SimpleNamespace(name = name, args = args, kwargs = kwargs)
