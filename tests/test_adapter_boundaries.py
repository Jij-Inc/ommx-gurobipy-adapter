import pytest
import gurobipy as gp

from ommx import (
    Bound,
    Constraint,
    DecisionVariable,
    Equality,
    Instance,
    Kind,
    Optimality,
)
from ommx_gurobipy_adapter import OMMXGurobipyAdapter


@pytest.mark.parametrize("fixed", [0, 1])
def test_binary_domain_bounds_are_preserved(fixed):
    x = DecisionVariable(7, Kind.Binary, Bound(fixed, fixed))
    instance = Instance.from_components(
        decision_variables=[x],
        objective=x if fixed == 0 else -x,
        active_constraints={},
        sense=Instance.MAXIMIZE,
    )
    solution = OMMXGurobipyAdapter.solve(instance)
    assert solution.feasible
    assert solution.state.entries[7] == fixed
    assert solution.optimality == Optimality.Optimal


@pytest.mark.parametrize("trigger", [0, 1])
@pytest.mark.parametrize(
    "equality", [Equality.EqualToZero, Equality.LessThanOrEqualToZero]
)
@pytest.mark.parametrize("constant,tolerance", [(5e-7, 1e-9), (5e-5, 1e-4)])
def test_constant_indicators_obey_backend_tolerance(
    trigger, equality, constant, tolerance
):
    instance = Instance.maximize()
    b = instance.new_binary("b")
    instance.objective = b if trigger else 1 - b
    instance.add_indicator_constraint(
        Constraint(function=constant, equality=equality).with_indicator(
            b, trigger_value=trigger
        )
    )
    adapter = OMMXGurobipyAdapter(instance)
    model = adapter.solver_input
    model.Params.FeasibilityTol = tolerance
    # Isolate feasibility tolerance from Gurobi's presolve simplifications.
    model.Params.Presolve = 0
    model.optimize()
    # Use native Gurobi as the oracle: tolerances and presolve can differ
    # between equality and inequality indicators and backend versions.
    with gp.Model() as reference:
        reference.Params.OutputFlag = 0
        reference.Params.FeasibilityTol = tolerance
        reference.Params.Presolve = 0
        variable = reference.addVar(vtype=gp.GRB.BINARY)
        reference.setObjective(variable if trigger else 1 - variable, gp.GRB.MAXIMIZE)
        reference.addGenConstrIndicator(
            variable,
            bool(trigger),
            gp.LinExpr(constant),
            gp.GRB.EQUAL if equality == Equality.EqualToZero else gp.GRB.LESS_EQUAL,
            0,
        )
        reference.optimize()
        assert model.ObjVal == pytest.approx(reference.ObjVal)
    assert model.NumGenConstrs == 1
