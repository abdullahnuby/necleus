"""Finite, typed Boolean DSL used by the Nucleus Pulse v0.43 benchmark.

Domain: two integer features, each in [0, 99].
Grammar v0.43-core:
  Atom := LT(x_i, t), EVEN(x_i), DIV3(x_i), DIV5(x_i), where i in {0,1}, t in {25,50,75}
  Literal := Atom | NOT(Atom)
  Expr := Literal
       | AND/OR/XOR(Literal, Literal)
       | AND/OR/XOR(Atom, AND/OR/XOR(Atom, Atom))

Binary operators are canonicalized as commutative. This grammar has maximum AST
 depth 3 and at most two binary operators per program. This bounded grammar is
intentional: it is fully enumerable for exact support-consistency auditing.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement, product
import hashlib
import json
from typing import Any, Iterable, Iterator, TypeAlias

Expr: TypeAlias = tuple
DOMAIN_MAX = 99
DOMAIN_SIZE = DOMAIN_MAX + 1
THRESHOLDS = (25, 50, 75)
PREDICATE_KINDS = ("lt", "even", "div3", "div5")
BINARY_OPS = ("and", "or", "xor")
FULL_MASK = (1 << (DOMAIN_SIZE * DOMAIN_SIZE)) - 1


def atom(kind: str, var: int, threshold: int | None = None) -> Expr:
    if kind not in PREDICATE_KINDS:
        raise ValueError(f"unsupported atom kind: {kind}")
    if var not in (0, 1):
        raise ValueError("var must be 0 or 1")
    if kind == "lt":
        if threshold not in THRESHOLDS:
            raise ValueError(f"threshold must be one of {THRESHOLDS}")
        return ("lt", var, int(threshold))
    if threshold is not None:
        raise ValueError(f"{kind.upper()} atoms do not accept a threshold")
    return (kind, var)


def neg(child: Expr) -> Expr:
    if child[0] not in PREDICATE_KINDS:
        raise ValueError("v0.43-core permits NOT only directly over an atom")
    return ("not", child)


def binop(op: str, left: Expr, right: Expr) -> Expr:
    if op not in BINARY_OPS:
        raise ValueError(f"unsupported binary operation: {op}")
    if left[0] in BINARY_OPS and right[0] in BINARY_OPS:
        # Two nested binary children would exceed the bounded two-binary-node
        # grammar when used as a program root.
        raise ValueError("v0.43-core does not allow two compound children")
    ordered = sorted((left, right), key=expr_code)
    return (op, ordered[0], ordered[1])


def expr_code(expr: Expr) -> str:
    return json.dumps(expr, separators=(",", ":"), ensure_ascii=True)


def ast_hash(expr: Expr) -> str:
    return hashlib.sha256(expr_code(expr).encode("utf-8")).hexdigest()


def ast_depth(expr: Expr) -> int:
    kind = expr[0]
    if kind in PREDICATE_KINDS:
        return 1
    if kind == "not":
        return 1 + ast_depth(expr[1])
    return 1 + max(ast_depth(expr[1]), ast_depth(expr[2]))


def ast_nodes(expr: Expr) -> int:
    kind = expr[0]
    if kind in PREDICATE_KINDS:
        return 1
    if kind == "not":
        return 1 + ast_nodes(expr[1])
    return 1 + ast_nodes(expr[1]) + ast_nodes(expr[2])


def evaluate(expr: Expr, x: tuple[int, int]) -> bool:
    kind = expr[0]
    if kind == "lt":
        return x[expr[1]] < expr[2]
    if kind == "even":
        return x[expr[1]] % 2 == 0
    if kind == "div3":
        return x[expr[1]] % 3 == 0
    if kind == "div5":
        return x[expr[1]] % 5 == 0
    if kind == "not":
        return not evaluate(expr[1], x)
    left = evaluate(expr[1], x)
    right = evaluate(expr[2], x)
    if kind == "and":
        return left and right
    if kind == "or":
        return left or right
    if kind == "xor":
        return left != right
    raise ValueError(f"unknown AST node: {kind}")


@lru_cache(maxsize=None)
def atom_mask(expr: Expr) -> int:
    """Truth signature over the full finite 10,000-point input domain."""
    if expr[0] not in PREDICATE_KINDS:
        raise ValueError("atom_mask expects an atom")
    kind, var = expr[0], expr[1]
    threshold = expr[2] if kind == "lt" else None
    mask = 0
    for x0 in range(DOMAIN_SIZE):
        for x1 in range(DOMAIN_SIZE):
            x = (x0, x1)
            if kind == "lt":
                value = x[var] < threshold
            elif kind == "even":
                value = x[var] % 2 == 0
            elif kind == "div3":
                value = x[var] % 3 == 0
            else:
                value = x[var] % 5 == 0
            if value:
                mask |= 1 << (x0 * DOMAIN_SIZE + x1)
    return mask


@lru_cache(maxsize=None)
def semantic_mask(expr: Expr) -> int:
    kind = expr[0]
    if kind in PREDICATE_KINDS:
        return atom_mask(expr)
    if kind == "not":
        return FULL_MASK ^ semantic_mask(expr[1])
    left = semantic_mask(expr[1])
    right = semantic_mask(expr[2])
    if kind == "and":
        return left & right
    if kind == "or":
        return left | right
    if kind == "xor":
        return left ^ right
    raise ValueError(f"unknown AST node: {kind}")


def point_index(x: tuple[int, int]) -> int:
    x0, x1 = x
    if not (0 <= x0 <= DOMAIN_MAX and 0 <= x1 <= DOMAIN_MAX):
        raise ValueError(f"input outside domain [0,{DOMAIN_MAX}]: {x}")
    return x0 * DOMAIN_SIZE + x1


def mask_predict(mask: int, x: tuple[int, int]) -> int:
    return (mask >> point_index(x)) & 1


def family_structure(expr: Expr) -> tuple:
    """AST family descriptor: thresholds discarded; variable renaming normalized.

    Repeated-variable patterns are preserved (x0,x0 differs from x0,x1), while
    absolute variable IDs and threshold constants do not create new families.
    """
    variables: dict[int, int] = {}

    def visit(node: Expr) -> tuple:
        kind = node[0]
        if kind in PREDICATE_KINDS:
            var = node[1]
            if var not in variables:
                variables[var] = len(variables)
            return (kind, variables[var])
        if kind == "not":
            return ("not", visit(node[1]))
        # ASTs are already canonicalized; visit in canonical order so variable
        # normalization is deterministic under commutative operand exchange.
        a = visit(node[1])
        b = visit(node[2])
        return (kind, a, b)

    return visit(expr)


def family_code(expr: Expr) -> str:
    return json.dumps(family_structure(expr), separators=(",", ":"), ensure_ascii=True)


def family_id(expr: Expr) -> str:
    return hashlib.sha256(family_code(expr).encode("utf-8")).hexdigest()[:16]


def all_atoms() -> tuple[Expr, ...]:
    atoms: list[Expr] = []
    for var in (0, 1):
        atoms.extend(atom("lt", var, threshold) for threshold in THRESHOLDS)
        atoms.append(atom("even", var))
        atoms.append(atom("div3", var))
        atoms.append(atom("div5", var))
    return tuple(sorted(atoms, key=expr_code))


@lru_cache(maxsize=1)
def enumerate_programs() -> tuple[Expr, ...]:
    """Enumerate every syntactic program in the explicitly bounded grammar."""
    atoms = all_atoms()
    literals = tuple(sorted((*atoms, *(neg(a) for a in atoms)), key=expr_code))
    programs: dict[str, Expr] = {expr_code(x): x for x in literals}

    # One binary node over literals; depth <= 3 including a possible NOT leaf.
    for op in BINARY_OPS:
        for left, right in combinations_with_replacement(literals, 2):
            expr = binop(op, left, right)
            programs[expr_code(expr)] = expr

    # Two binary nodes in a bounded depth-3 shape. NOT is not allowed here,
    # because it would raise the AST depth to 4.
    for root_op, inner_op in product(BINARY_OPS, repeat=2):
        for outer_atom in atoms:
            for left, right in combinations_with_replacement(atoms, 2):
                inner = binop(inner_op, left, right)
                expr = binop(root_op, outer_atom, inner)
                if ast_depth(expr) <= 3 and ast_nodes(expr) <= 7:
                    programs[expr_code(expr)] = expr

    result = tuple(programs[k] for k in sorted(programs))
    if not result:
        raise RuntimeError("grammar enumeration unexpectedly produced no programs")
    return result


def canonical_semantic_programs(programs: Iterable[Expr] | None = None) -> tuple[Expr, ...]:
    """Return one simplest AST representative per truth function on the domain."""
    source = enumerate_programs() if programs is None else tuple(programs)
    best: dict[int, Expr] = {}
    for expr in source:
        signature = semantic_mask(expr)
        previous = best.get(signature)
        rank = (ast_depth(expr), ast_nodes(expr), expr_code(expr))
        if previous is None or rank < (ast_depth(previous), ast_nodes(previous), expr_code(previous)):
            best[signature] = expr
    return tuple(sorted(best.values(), key=lambda e: (family_id(e), expr_code(e))))


def decode_expr(data: Any) -> Expr:
    """Validate nested JSON-list AST data and return its tuple representation."""
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("AST must be a non-empty list/tuple")
    node = tuple(data)
    kind = node[0]
    if kind == "lt" and len(node) == 3:
        return atom("lt", int(node[1]), int(node[2]))
    if kind in ("even", "div3", "div5") and len(node) == 2:
        return atom(kind, int(node[1]))
    if kind == "not" and len(node) == 2:
        return neg(decode_expr(node[1]))
    if kind in BINARY_OPS and len(node) == 3:
        left, right = decode_expr(node[1]), decode_expr(node[2])
        return binop(kind, left, right)
    raise ValueError(f"invalid AST node: {node!r}")


def encode_expr(expr: Expr) -> list:
    return [encode_expr(child) if isinstance(child, tuple) else child for child in expr]
