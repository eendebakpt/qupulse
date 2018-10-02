from typing import Union, Dict, Tuple, Any, Sequence, Optional, Mapping
from numbers import Number
from types import CodeType

import builtins
import math

import sympy
from sympy.parsing.sympy_parser import NAME, OP, iskeyword, Basic, Symbol, lambda_notation, repeated_decimals, \
    auto_number, factorial_notation
import numpy

from unittest import mock


__all__ = ["sympify", "substitute_with_eval", "to_numpy", "get_variables", "get_free_symbols", "recursive_substitution",
           "evaluate_lambdified", "get_most_simple_representation"]


Sympifyable = Union[str, Number, sympy.Expr, numpy.str_]

##############################################################################
### Utilities to automatically detect usage of indexed/subscripted symbols ###
##############################################################################

## Custom auto_symbol transformation that deals with namespace dot notation (e.g. "foo.bar")


sympy_internal_namespace_seperator = '____'


def custom_auto_symbol_transform(tokens: Sequence[Tuple[int, str]], local_dict: Mapping[str, Any], global_dict: Mapping[str, Any]) -> None:
    """Inserts calls to ``Symbol``/``Function`` for undefined variables and deals with symbol namespaces.

    Original code taken from sympy and tweaked to allow for namespaced parameters following dot notation, e.g., ``foo.bar``.
    A string ``foo.bar`` will be treated as a single symbol. To allow internal handling, dots are replaced by '____'
    (4 underscores).
    """
    result = []
    prev_tok = (None, None)
    symbol_string = None

    tokens.append((None, None))  # so zip traverses all tokens
    for tok, next_tok in zip(tokens, tokens[1:]):
        tok_num, tok_val = tok
        next_tok_num, next_tok_val = next_tok

        if symbol_string:
            if tok_val != '.' and tok_num != NAME:
                raise SyntaxError("Not a valid namespaced sympy.symbol name")
            if tok_val == '.':
                symbol_string += sympy_internal_namespace_seperator
            elif tok_num == NAME:
                symbol_string += tok_val
            if tok_val == '.' or (tok_num == NAME and next_tok_val == '.'):
                continue
            tok_num = NAME
            tok_val = symbol_string
            symbol_string = None

        if tok_num == NAME:
            name = tok_val

            if (name in ['True', 'False', 'None']
                    or iskeyword(name)
                    # Don't convert keyword arguments
                    or (prev_tok[0] == OP and prev_tok[1] in ('(', ',')
                        and next_tok_num == OP and next_tok_val == '=')):
                result.append((NAME, name))
                continue
            elif next_tok_val == '.':
                symbol_string = str(name)
                continue
            elif name in local_dict:
                if isinstance(local_dict[name], Symbol) and next_tok_val == '(':
                    result.extend([(NAME, 'Function'),
                                   (OP, '('),
                                   (NAME, repr(str(local_dict[name]))),
                                   (OP, ')')])
                else:
                    result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue

            result.extend([
                (NAME, 'Symbol' if next_tok_val != '(' else 'Function'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ')'),
            ])
        else:
            result.append((tok_num, tok_val))

        prev_tok = (tok_num, tok_val)

    return result

sympy_transformations = (lambda_notation, custom_auto_symbol_transform,
                         repeated_decimals, auto_number, factorial_notation)


## Utilities to automatically detect usage of indexed/subscripted symbols

class IndexedBasedFinder:
    """Acts as a symbol lookup and determines which symbols in an expression a subscripted."""

    def __init__(self):
        self.symbols = set()
        self.indexed_base = set()
        self.indices = set()

        class SubscriptionChecker(sympy.Symbol):
            """A symbol stand-in which detects whether the symbol is subscripted."""

            def __getitem__(s, k):
                self.indexed_base.add(str(s))
                self.indices.add(k)
                if isinstance(k, SubscriptionChecker):
                    k = sympy.Symbol(str(k))
                return sympy.IndexedBase(str(s))[k]

        self.SubscriptionChecker = SubscriptionChecker

    def __getitem__(self, k) -> sympy.Expr:
        """Return an instance of the internal SubscriptionChecker class for each symbol to determine which symbols are
        indexed/subscripted.

        __getitem__ is (apparently) called by symbol for each token and gets either symbol names or type names such as
        'Integer', 'Float', etc. We have to take care of returning correct types for symbols (-> SubscriptionChecker)
        and the base types (-> Integer, Float, etc).
        """
        if hasattr(sympy, k): # if k is a sympy base type identifier, return the base type
            return getattr(sympy, k)

        # otherwise track the symbol name and return a SubscriptionChecker instance
        self.symbols.add(k)
        return self.SubscriptionChecker(k)

    def __contains__(self, k) -> bool:
        return True


def get_subscripted_symbols(expression: str) -> set:
    # track all symbols that are subscipted in here
    indexed_base_finder = IndexedBasedFinder()
    with mock.patch.object(sympy.parsing.sympy_parser, 'standard_transformations', sympy_transformations):
        sympy.sympify(expression, locals=indexed_base_finder)

    return indexed_base_finder.indexed_base

#############################################################
### "Built-in" length function for expressions in qupulse ###
#############################################################

class Len(sympy.Function):
    nargs = 1

    @classmethod
    def eval(cls, arg) -> Optional[sympy.Integer]:
        if hasattr(arg, '__len__'):
            return sympy.Integer(len(arg))

    is_Integer = True
Len.__name__ = 'len'


sympify_namespace = {'len': Len,
                     'Len': Len}

#########################################
### Functions for numpy compatability ###
#########################################

## Functions for numpy compatability

def numpy_compatible_mul(*args) -> Union[sympy.Mul, sympy.Array]:
    if any(isinstance(a, sympy.NDimArray) for a in args):
        result = 1
        for a in args:
            result = result * (numpy.array(a.tolist()) if isinstance(a, sympy.NDimArray) else a)
        return sympy.Array(result)
    else:
        return sympy.Mul(*args)


def numpy_compatible_ceiling(input_value: Any) -> Any:
    if isinstance(input_value, numpy.ndarray):
        return numpy.ceil(input_value).astype(numpy.int64)
    else:
        return sympy.ceiling(input_value)


def to_numpy(sympy_array: sympy.NDimArray) -> numpy.ndarray:
    if isinstance(sympy_array, sympy.DenseNDimArray):
        if len(sympy_array.shape) == 2:
            return numpy.asarray(sympy_array.tomatrix())
        elif len(sympy_array.shape) == 1:
            return numpy.asarray(sympy_array)
    return numpy.array(sympy_array.tolist())

#######################################################################################################
### Custom sympify method (which introduces all utility methods defined above into the sympy world) ###
#######################################################################################################


## Custom sympify method (which introduces all utility methods defined above into the sympy world)

def sympify(expr: Union[str, Number, sympy.Expr, numpy.str_], **kwargs) -> sympy.Expr:
    if isinstance(expr, numpy.str_):
        # putting numpy.str_ in sympy.sympify behaves unexpected in version 1.1.1
        # It seems to ignore the locals argument
        expr = str(expr)
    with mock.patch.object(sympy.parsing.sympy_parser, 'standard_transformations', sympy_transformations):
        try:
            return sympy.sympify(expr, **kwargs, locals=sympify_namespace)
        except TypeError as err:
            if True:#err.args[0] == "'Symbol' object is not subscriptable":

                indexed_base = get_subscripted_symbols(expr)
                return sympy.sympify(expr, **kwargs, locals={**{k: sympy.IndexedBase(k)
                                                                for k in indexed_base},
                                                             **sympify_namespace})

            else:
                raise


###############################################################################
### Utility functions for expression manipulation/simplification/evaluation ###
###############################################################################

def get_most_simple_representation(expression: sympy.Expr) -> Union[str, int, float]:
    if expression.free_symbols:
        return str(expression).replace(sympy_internal_namespace_seperator, '.')
    elif expression.is_Integer:
        return int(expression)
    elif expression.is_Float:
        return float(expression)
    else:
        return str(expression)


def get_free_symbols(expression: sympy.Expr) -> Sequence[sympy.Symbol]:
    """Returns all free smybols in a sympy expression.

    Since these are sympy Symbol objects, possibly namespaced symbols will follow the underscore namespace notation,
    i.e., `foo___bar`.
    """
    return tuple(symbol
                 for symbol in expression.free_symbols
                 if not isinstance(symbol, sympy.Indexed))


def get_variables(expression: sympy.Expr) -> Sequence[str]:
    """Returns all free variables in a sympy expression.

    Returned are the names of the variables. Namespaced variables will follow the dot namespace notation, i.e.
    `foo.bar`.
    """
    return tuple(map(lambda x: str(x).replace(sympy_internal_namespace_seperator, '.'), get_free_symbols(expression)))


def substitute_with_eval(expression: sympy.Expr,
                         substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    """Substitutes only sympy.Symbols. Workaround for numpy like array behaviour. ~Factor 3 slower compared to subs"""
    substitutions = {k.replace('.', sympy_internal_namespace_seperator): v if isinstance(v, sympy.Expr) else sympify(v)
                     for k, v in substitutions.items()}

    for symbol in get_free_symbols(expression):
        symbol_name = str(symbol)
        if symbol_name not in substitutions:
            substitutions[symbol_name] = symbol

    string_representation = sympy.srepr(expression)
    return eval(string_representation, sympy.__dict__, {'Symbol': substitutions.__getitem__,
                                                        'Mul': numpy_compatible_mul})


def _recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    if not expression.free_symbols:
        return expression
    elif expression.func is sympy.Symbol:
        return substitutions.get(expression, expression)

    elif expression.func is sympy.Mul:
        func = numpy_compatible_mul
    else:
        func = expression.func
    substitutions = {s: substitutions.get(s, s) for s in get_free_symbols(expression)}
    return func(*(_recursive_substitution(arg, substitutions) for arg in expression.args))


def recursive_substitution(expression: sympy.Expr,
                           substitutions: Dict[str, Union[sympy.Expr, numpy.ndarray, str]]) -> sympy.Expr:
    substitutions = {sympy.Symbol(k.replace('.',sympy_internal_namespace_seperator)): sympify(v) for k, v in substitutions.items()}
    for s in get_free_symbols(expression):
        substitutions.setdefault(s, s)
    return _recursive_substitution(expression, substitutions)


_base_environment = {'builtins': builtins, '__builtins__':  builtins}
_math_environment = {**_base_environment, **math.__dict__}
_numpy_environment = {**_base_environment, **numpy.__dict__}
_sympy_environment = {**_base_environment, **sympy.__dict__}


def evaluate_compiled(expression: sympy.Expr,
             parameters: Dict[str, Union[numpy.ndarray, Number]],
             compiled: CodeType=None, mode=None) -> Tuple[any, CodeType]:
    parameters = {k.replace('.', sympy_internal_namespace_seperator): v for k, v in parameters.items()}
    with mock.patch.object(sympy.parsing.sympy_parser, 'standard_transformations', sympy_transformations):
        if compiled is None:
            compiled = compile(sympy.printing.lambdarepr.lambdarepr(expression),
                               '<string>', 'eval')

        if mode == 'numeric' or mode is None:
            result = eval(compiled, parameters.copy(), _numpy_environment)
        elif mode == 'exact':
            result = eval(compiled, parameters.copy(), _sympy_environment)
        else:
            raise ValueError("Unknown mode: '{}'".format(mode))

        return result, compiled


def evaluate_lambdified(expression: Union[sympy.Expr, numpy.ndarray],
                        variables: Sequence[str],
                        parameters: Dict[str, Union[numpy.ndarray, Number]],
                        lambdified) -> Tuple[Any, Any]:
    variables = {v.replace('.', sympy_internal_namespace_seperator) for v in variables}
    parameters = {k.replace('.', sympy_internal_namespace_seperator):v for k,v in parameters.items()}
    with mock.patch.object(sympy.parsing.sympy_parser, 'standard_transformations', sympy_transformations):
        lambdified = lambdified or sympy.lambdify(variables, expression,
                                                  [{'ceiling': numpy_compatible_ceiling}, 'numpy'])

        return lambdified(**parameters), lambdified