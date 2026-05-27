from __future__ import annotations

import logging

from deap import base, creator, gp, tools


logger = logging.getLogger(__name__)


def build_toolbox(
    pset: gp.PrimitiveSet,
    *,
    max_tree_height: int,
    tournament_size: int,
    n_targets: int = 1,
) -> base.Toolbox:
    """Build a DEAP toolbox for symbolic regression.

    Parameters
    ----------
    pset:
        The primitive set describing the GP language.
    max_tree_height:
        Maximum depth allowed for GP trees (enforced via static limits on
        mate/mutate).
    tournament_size:
        Tournament selection pressure.
    n_targets:
        Number of regression targets.  The fitness vector has one MSE
        objective per target plus one complexity objective, giving
        ``n_targets + 1`` objectives in total, all minimised.
    """
    # Fitness weights: one −1 per target MSE + one −1 for complexity.
    # We must recreate the class whenever n_targets changes (different shape).
    fitness_weights = (-1.0,) * (n_targets + 1)

    # DEAP stores fitness classes globally; key by weight tuple to allow
    # different target counts to coexist (e.g. in tests).
    fitness_cls_name = f"SymbolicFitness_{n_targets}t"
    individual_cls_name = f"SymbolicIndividual_{n_targets}t"

    if not hasattr(creator, fitness_cls_name):
        creator.create(fitness_cls_name, base.Fitness, weights=fitness_weights)
    if not hasattr(creator, individual_cls_name):
        creator.create(
            individual_cls_name,
            gp.PrimitiveTree,
            fitness=getattr(creator, fitness_cls_name),
        )

    # Expose the canonical names used everywhere else in the codebase so that
    # callers can keep using ``creator.SymbolicIndividual`` regardless of how
    # many targets there are.  We simply alias the versioned class.
    creator.SymbolicFitness = getattr(creator, fitness_cls_name)  # type: ignore[attr-defined]
    creator.SymbolicIndividual = getattr(creator, individual_cls_name)  # type: ignore[attr-defined]

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree_height)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.SymbolicIndividual,  # type: ignore[attr-defined]
        toolbox.expr,  # type: ignore[attr-defined]
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,  # type: ignore[attr-defined]
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree_height)
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=toolbox.expr_mut,  # type: ignore[attr-defined]
        pset=pset,
    )
    max_nodes = max_tree_height * 4
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_nodes))
    logger.debug(
        "Symbolic toolbox ready",
        extra={
            "max_height": max_tree_height,
            "tournament": tournament_size,
            "n_targets": n_targets,
        },
    )
    return toolbox
