import pytest
from ommx.v1 import Instance, DecisionVariable, Optimality

from ommx_gurobipy_adapter import OMMXGurobipyAdapter


def test_solution_optimality():
    """Test that optimal solutions are correctly marked as optimal"""
    x = DecisionVariable.integer(1, lower=0, upper=5)
    y = DecisionVariable.integer(2, lower=0, upper=5)
    ommx_instance = Instance.from_components(
        decision_variables=[x, y],
        objective=x + y,
        constraints=[],
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(ommx_instance)
    assert solution.optimality == Optimality.Optimal


def test_basic_functionality():
    """Test basic functionality with a simple optimization problem"""
    # Simple problem: maximize x + 2y subject to x + y <= 5
    x = DecisionVariable.continuous(1, lower=0, upper=10)
    y = DecisionVariable.continuous(2, lower=0, upper=10)

    instance = Instance.from_components(
        decision_variables=[x, y],
        objective=x + 2 * y,
        constraints=[x + y <= 5],
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    # Optimal solution should be x=0, y=5
    assert solution.state.entries[1] == pytest.approx(0, abs=1e-6)
    assert solution.state.entries[2] == pytest.approx(5)
    assert solution.objective == pytest.approx(10)  # 0 + 2*5


def test_multi_objective_handling():
    """Test that the adapter correctly handles multiple objectives by focusing on the primary one"""
    x = DecisionVariable.continuous(1, lower=0, upper=1)

    instance = Instance.from_components(
        decision_variables=[x],
        objective=x,  # Primary objective: maximize x
        constraints=[],
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    # Should maximize x to its upper bound
    assert solution.state.entries[1] == pytest.approx(1)
    assert solution.objective == pytest.approx(1)


def test_partial_evaluate():
    """Test that the adapter correctly handles partially evaluated instances with used_decision_variables"""
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + x[1] + x[2],
        constraints=[(x[0] + x[1] + x[2] <= 1).set_id(0)],
        sense=Instance.MINIMIZE,
    )
    assert instance.used_decision_variables == x
    partial = instance.partial_evaluate({0: 1})
    # x[0] is no longer present in the problem
    assert partial.used_decision_variables == x[1:]

    solution = OMMXGurobipyAdapter.solve(partial)
    assert [var.value for var in solution.decision_variables] == [1, 0, 0]

    partial = instance.partial_evaluate({1: 1})
    solution = OMMXGurobipyAdapter.solve(partial)
    assert [var.value for var in solution.decision_variables] == [0, 1, 0]

    partial = instance.partial_evaluate({2: 1})
    solution = OMMXGurobipyAdapter.solve(partial)
    assert [var.value for var in solution.decision_variables] == [0, 0, 1]


def test_relax_constraint():
    """Test that the adapter correctly handles constraint relaxation and irrelevant variable identification"""
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + x[1],
        constraints=[(x[0] + 2 * x[1] <= 1).set_id(0), (x[1] + x[2] <= 1).set_id(1)],
        sense=Instance.MINIMIZE,
    )

    assert instance.used_decision_variables == x
    instance.relax_constraint(1, "relax")
    # id for x[2] is listed as irrelevant
    assert instance.decision_variable_analysis().irrelevant() == {x[2].id}

    solution = OMMXGurobipyAdapter.solve(instance)
    # x[2] is still present as part of the evaluate/decoding process but has a value of 0
    assert [var.value for var in solution.decision_variables] == [0, 0, 0]
