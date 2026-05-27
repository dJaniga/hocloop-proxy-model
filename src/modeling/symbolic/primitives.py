import numpy as np
from deap import gp


def _add(a, b):
    return np.add(a, b)


def _sub(a, b):
    return np.subtract(a, b)


def _mul(a, b):
    return np.multiply(a, b)


def _neg(a):
    return np.negative(a)


def _square(a):
    return np.square(a)


def _abs(a):
    return np.abs(a)


def _protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(right) > 1e-12, left / right, left)
    return result


def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


def _random_ephemeral_constant() -> float:
    """Ephemeral constant generator for the primitive set."""
    return float(np.random.uniform(-1.0, 1.0))


def build_primitive_set(
    feature_count: int, basic_arithmetic_only: bool = False
) -> gp.PrimitiveSet:
    """Build the GP primitive set with operators suitable for symbolic regression.

    All primitives use plain Python functions (not numpy ufuncs) to ensure
    consistent return types that DEAP's type system expects.
    """
    pset = gp.PrimitiveSet("MAIN", feature_count)
    pset.addPrimitive(_add, 2)
    pset.addPrimitive(_sub, 2)
    pset.addPrimitive(_mul, 2)
    pset.addPrimitive(_protected_div, 2)

    if not basic_arithmetic_only:
        pset.addPrimitive(_protected_sqrt, 1)
        pset.addPrimitive(_square, 1)
        pset.addPrimitive(_abs, 1)
        pset.addPrimitive(_neg, 1)

    pset.addEphemeralConstant("rand", _random_ephemeral_constant)
    return pset
