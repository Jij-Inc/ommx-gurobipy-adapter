# OMMX adapter for Gurobi

This package provides an adapter for [Gurobi](https://www.gurobi.com/) from [OMMX](https://github.com/Jij-Inc/ommx). It allows you to solve optimization problems defined in OMMX format using Gurobi's powerful solver.

## Features

- Support for various variable types:
  - Binary variables
  - Integer variables
  - Continuous variables
- Support for different optimization problems:
  - Linear Programming (LP)
  - Mixed Integer Linear Programming (MILP)
  - Quadratic Programming (QP)
  - Mixed Integer Quadratic Programming (MIQP)
- Support for both minimization and maximization problems

## Prerequisites

- Python >= 3.11
- Gurobi Optimizer and valid license
- gurobipy >= 12.0.1
- ommx >= 3.0.0rc1

## Installation

First, ensure you have Gurobi installed and properly licensed. Then install the OMMX Gurobi adapter using pip:

```bash
pip install ommx-gurobipy-adapter
```

## Usage

`OMMXGurobipyAdapter` is marked with `typing.final`; type checkers reject
subclasses. Implement custom adapters using `ommx.adapter.SolverAdapter`.

Here's a simple example of how to use the adapter:

```python markdown-code-runner
from ommx_gurobipy_adapter import OMMXGurobipyAdapter
from ommx import Instance, DecisionVariable

# Create decision variables
x1 = DecisionVariable.integer(1, lower=0, upper=5)
x2 = DecisionVariable.continuous(2, lower=0, upper=5)

# Create OMMX instance
instance = Instance.from_components(
    decision_variables=[x1, x2],
    objective=x1 + 2*x2,
    active_constraints={
        0: x1 + x2 <= 5,  # Linear constraint
    },
    sense=Instance.MAXIMIZE,
)

# Solve using Gurobi
solution = OMMXGurobipyAdapter.solve(instance)

# Access the results
print(f"Objective value: {solution.objective}")
print(f"x1 = {solution.state.entries[1]}")
print(f"x2 = {solution.state.entries[2]}")
```

### Automatic Preparation: OneHot Constraints

Gurobi does not accept OMMX OneHot constraints directly. For example, when
choosing exactly one of two binary variables, `solve()` converts the OneHot
constraint to the linear equality `a + b = 1` on an internal copy:

```python markdown-code-runner
from ommx import ConstraintKind, OneHotConstraint

choices = Instance.maximize()
a = choices.new_binary("a")
b = choices.new_binary("b")
choices.objective = 3 * a + 5 * b
choices.add_one_hot_constraint(OneHotConstraint(variables=[a, b]))

solution = OMMXGurobipyAdapter.solve(choices)
assert solution.feasible
assert solution.objective == 5

# The original instance still contains the OneHot constraint.
assert len(choices.active_constraints.of_kind(ConstraintKind.OneHot)) == 1
assert not OMMXGurobipyAdapter.check_applicability(choices).is_member
```

Calling `OMMXGurobipyAdapter(choices)` or
`OMMXGurobipyAdapter.solve_without_preparation(choices)` at this point would
raise `AdapterNotApplicableError`: the preceding solve did not apply its
conversion to `choices`. To use these APIs, prepare that instance explicitly:

```python markdown-code-runner
choices.prepare(
    OMMXGurobipyAdapter.INPUT_CLASS,
    OMMXGurobipyAdapter.recommended_preparation_policy(),
)
assert not choices.active_constraints.of_kind(ConstraintKind.OneHot)
solution = OMMXGurobipyAdapter.solve_without_preparation(choices)
assert solution.objective == 5
```

The recommended policy converts only OneHot constraints. Indicator and SOS1
constraints are passed to Gurobi directly; Indicators support both
`trigger_value=0` and `trigger_value=1`.

### Controlling Gurobi Parameters

If you need more control over the Gurobi solver parameters, you can use the adapter in two steps:

The constructor retains the supplied Instance without copying it. You are
responsible for keeping that Instance unchanged from adapter construction
through optimization and decoding, and for as long as you continue using
the adapter's `solver_input`. Decoding evaluates the backend result against
that same Instance; changes to it are not synchronized with the Gurobi model.

```python markdown-code-runner
from ommx_gurobipy_adapter import OMMXGurobipyAdapter

# Create adapter
adapter = OMMXGurobipyAdapter(instance)

# Get Gurobi model
model = adapter.solver_input

# Set Gurobi parameters
model.setParam('TimeLimit', 60)  # Set time limit to 60 seconds
model.setParam('MIPGap', 0.01)   # Set relative MIP gap tolerance to 1%

# Solve
model.optimize()

# Get solution
solution = adapter.decode(model)
```

## Error Handling

The adapter provides specific error types for different situations:

- `AdapterNotApplicableError`: Raised when the instance is outside the strict
  Gurobi input class in the constructor or `solve_without_preparation`
- `PreparationTargetNotReachedError`: Raised when `solve` cannot prepare its
  copy into the Gurobi input class with the recommended policy
- `OMMXGurobipyAdapterError`: Base error class for adapter-specific errors
- `InfeasibleDetected`: Raised when the problem is infeasible
- `InfeasibleOrUnboundedDetected`: Raised when Gurobi cannot distinguish
  infeasibility from unboundedness; decoding does not re-solve the model
- `UnboundedDetected`: Raised when the problem is unbounded

Example of error handling:

```python markdown-code-runner
from ommx_gurobipy_adapter import OMMXGurobipyAdapterError
from ommx import PreparationTargetNotReachedError
from ommx.adapter import (
    AdapterNotApplicableError,
    InfeasibleDetected,
    InfeasibleOrUnboundedDetected,
    UnboundedDetected,
)

try:
    solution = OMMXGurobipyAdapter.solve(instance)
except AdapterNotApplicableError:
    print("Prepare the instance or inspect the applicability report")
except PreparationTargetNotReachedError:
    print("Inspect the preparation report or choose a different preparation policy")
except InfeasibleDetected:
    print("Problem is infeasible")
except InfeasibleOrUnboundedDetected:
    print("Problem is infeasible or unbounded; further analysis is needed")
except UnboundedDetected:
    print("Problem is unbounded")
except OMMXGurobipyAdapterError as e:
    print(f"Adapter error: {e}")
```

## Testing

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Run `pytest` and `markdown-code-runner`:

    ```bash
    uv run pytest -vv
    uv run markdown-code-runner README.md
    ```

## License

This project (the source code contained in this repository) is licensed under the [Apache License 2.0](./LICENSE).

> [!WARNING]
> The package `ommx-gurobipy-adapter` depends on `gurobipy`, which is a proprietary software.
> Please make sure you have a valid license for Gurobi Optimizer when using this package.

## Reference

For more information about OMMX, please visit: https://github.com/Jij-Inc/ommx

For Gurobi documentation, please visit: https://www.gurobi.com/documentation/

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
