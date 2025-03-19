from typing import List, Optional, Dict, Union, Any

try:
    from gurobipy import GRB as _GRB
except ImportError:
    # Mock GRB for testing
    class _GRB:
        BINARY = "B"
        INTEGER = "I"
        CONTINUOUS = "C"
        MINIMIZE = -1
        MAXIMIZE = 1

from ommx.v1 import DecisionVariable, Constraint, Function, Linear, Quadratic, Polynomial, Instance
from ommx.v1.decision_variables_pb2 import Bound, DecisionVariable as _DecisionVariable
from ommx.v1.constraint_pb2 import Equality
from ommx_gurobipy_adapter import OMMXGurobipyAdapter


class GRB:
    BINARY = _GRB.BINARY
    INTEGER = _GRB.INTEGER
    CONTINUOUS = _GRB.CONTINUOUS
    MINIMIZE = _GRB.MINIMIZE
    MAXIMIZE = _GRB.MAXIMIZE
    
    # Add constraint sense constants
    LESS_EQUAL = "≤"
    GREATER_EQUAL = "≥"
    EQUAL = "="
    
    # Add optimization status constants
    OPTIMAL = 2
    INFEASIBLE = 3
    UNBOUNDED = 5

    _MAP_TYPES_GRB_TO_OMMX = {
        BINARY: _DecisionVariable.Kind.KIND_BINARY,
        INTEGER: _DecisionVariable.Kind.KIND_INTEGER,
        CONTINUOUS: _DecisionVariable.Kind.KIND_CONTINUOUS,
    }
    _MAP_TYPES_OMMX_TO_GRB = {v: k for k, v in _MAP_TYPES_GRB_TO_OMMX.items()}

    @classmethod
    def map_types_grb_to_ommx(cls, value):
        return cls._MAP_TYPES_GRB_TO_OMMX.get(value, value)

    @classmethod
    def map_types_ommx_to_grb(cls, value):
        return cls._MAP_TYPES_OMMX_TO_GRB.get(value, value)


class Var:
    def __init__(self, model, decision_variable: DecisionVariable):
        self._ommx_var = decision_variable
        self._model = model

    @property
    def VarName(self) -> str:
        return self._ommx_var.name

    @property
    def VType(self) -> str:
        return GRB.map_types_ommx_to_grb(self._ommx_var.kind)
    
    @property
    def LB(self) -> float:
        return self._ommx_var.bound.lower

    @property
    def UB(self) -> float:
        return self._ommx_var.bound.upper
        
    @property
    def X(self) -> float:
        # Return the value of the variable from the solution
        if self._model._solution is None:
            raise ValueError("Model not optimized")
        return self._model._solution.raw.state.entries.get(self._ommx_var.id, 0.0)
        
    def __add__(self, other):
        if isinstance(other, Var):
            return self._ommx_var + other._ommx_var
        return self._ommx_var + other

    def __radd__(self, other):
        return self._ommx_var + other

    def __mul__(self, other):
        if isinstance(other, Var):
            return self._ommx_var * other._ommx_var
        result = self._ommx_var * other
        # Store reference to the model and variable in the result
        result._model = self._model
        result._var = self
        # Store variable ID for easier lookup
        if hasattr(self._ommx_var, 'id'):
            result._var_id = self._ommx_var.id
        return result

    def __rmul__(self, other):
        result = self._ommx_var * other
        # Store reference to the model and variable in the result
        result._model = self._model
        result._var = self
        # Store variable ID for easier lookup
        if hasattr(self._ommx_var, 'id'):
            result._var_id = self._ommx_var.id
        return result

    def __sub__(self, other):
        if isinstance(other, Var):
            return self._ommx_var - other._ommx_var
        return self._ommx_var - other

    def __rsub__(self, other):
        return -self._ommx_var + other

    def __neg__(self):
        return -self._ommx_var

    def __le__(self, other):
        if isinstance(other, Var):
            return self._ommx_var <= other._ommx_var
        return self._ommx_var <= other

    def __ge__(self, other):
        if isinstance(other, Var):
            return self._ommx_var >= other._ommx_var
        return self._ommx_var >= other
        
    def __eq__(self, other):
        if isinstance(other, Var):
            return self._ommx_var == other._ommx_var
        return self._ommx_var == other


class Constr:
    def __init__(self, model, constraint: Constraint, name: str = ""):
        self._model = model
        self._ommx_constraint = constraint
        self._name = name

    @property
    def Sense(self):
        if self._ommx_constraint.raw.equality == Constraint.EQUAL_TO_ZERO:
            return GRB.EQUAL
        elif self._ommx_constraint.raw.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
            if self._is_greater_equal():
                return GRB.GREATER_EQUAL
            else:
                return GRB.LESS_EQUAL
        return None

    @property
    def RHS(self):
        # Extract the constant term from the function
        if self._ommx_constraint.raw.function.HasField("constant"):
            constant = -self._ommx_constraint.raw.function.constant
        elif self._ommx_constraint.raw.function.HasField("linear"):
            constant = -self._ommx_constraint.raw.function.linear.constant
        elif self._ommx_constraint.raw.function.HasField("quadratic"):
            constant = -self._ommx_constraint.raw.function.quadratic.linear.constant
        elif self._ommx_constraint.raw.function.HasField("polynomial"):
            constant = -self._ommx_constraint.raw.function.polynomial.constant
        else:
            constant = 0.0
            
        # For greater-equal constraints, we need to negate the constant
        if self.Sense == GRB.GREATER_EQUAL:
            return -constant
        return constant

    def _is_greater_equal(self):
        # Check if this is actually a >= constraint (implemented as <= with negated terms)
        # This requires examining the coefficients of the function
        # For simplicity, we'll check if all coefficients are negative
        if self._ommx_constraint.raw.function.HasField("linear"):
            linear = self._ommx_constraint.raw.function.linear
            return all(term.coefficient < 0 for term in linear.terms)
        return False


class Row:
    def __init__(self, model, constraint: Constraint):
        self._model = model
        self._ommx_constraint = constraint
        self._vars = []
        self._coeffs = []
        self._extract_vars_and_coeffs()

    def _extract_vars_and_coeffs(self):
        # Extract variables and coefficients from the constraint function
        if self._ommx_constraint.raw.function.HasField("linear"):
            linear = self._ommx_constraint.raw.function.linear
            for term in linear.terms:
                # Check if the term has an id field
                var_id = getattr(term, 'id', None)
                coeff = term.coefficient
                
                # If the term has an id, use it to get the variable
                if var_id is not None:
                    var = self._model.getVarById(var_id)
                    if var is not None:
                        self._vars.append(var)
                        # For greater-equal constraints, negate the coefficient
                        if self._ommx_constraint.raw.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO and all(t.coefficient < 0 for t in linear.terms):
                            self._coeffs.append(-coeff)
                        else:
                            self._coeffs.append(coeff)
        elif self._ommx_constraint.raw.function.HasField("quadratic"):
            # Handle quadratic terms if needed
            pass

    def size(self):
        return len(self._vars)

    def getVar(self, i):
        if i < 0 or i >= len(self._vars):
            raise IndexError("Index out of range")
        return self._vars[i]

    def getCoeff(self, i):
        if i < 0 or i >= len(self._coeffs):
            raise IndexError("Index out of range")
        return self._coeffs[i]


class LinExpr:
    def __init__(self, expr=None):
        self._vars = []
        self._coeffs = []
        self._constant = 0.0
        
        if expr is not None:
            if isinstance(expr, Var):
                self._vars.append(expr)
                self._coeffs.append(1.0)
            elif isinstance(expr, (int, float)):
                self._constant = float(expr)
            elif isinstance(expr, LinExpr):
                self._vars = expr._vars.copy()
                self._coeffs = expr._coeffs.copy()
                self._constant = expr._constant
    
    def addTerms(self, coeffs, vars):
        if not isinstance(coeffs, list):
            coeffs = [coeffs]
        if not isinstance(vars, list):
            vars = [vars]
        
        if len(coeffs) != len(vars):
            raise ValueError("Coefficients and variables must have the same length")
        
        for i in range(len(coeffs)):
            self._vars.append(vars[i])
            self._coeffs.append(coeffs[i])
        
        return self
    
    def size(self):
        return len(self._vars)
    
    def getVar(self, i):
        if i < 0 or i >= len(self._vars):
            raise IndexError("Index out of range")
        return self._vars[i]
    
    def getCoeff(self, i):
        if i < 0 or i >= len(self._coeffs):
            raise IndexError("Index out of range")
        return self._coeffs[i]
    
    def __add__(self, other):
        result = LinExpr(self)
        if isinstance(other, Var):
            result._vars.append(other)
            result._coeffs.append(1.0)
        elif isinstance(other, (int, float)):
            result._constant += float(other)
        elif isinstance(other, LinExpr):
            result._vars.extend(other._vars)
            result._coeffs.extend(other._coeffs)
            result._constant += other._constant
        # Add case for OMMX Linear objects
        elif hasattr(other, 'terms'):
            # Handle OMMX expressions (Linear, Quadratic, etc.)
            for var_tuple, coeff in other.terms.items():
                if var_tuple == ():  # Constant term
                    result._constant += coeff
                elif len(var_tuple) == 1:  # Linear term
                    var_id = var_tuple[0]
                    # Find the variable in the model
                    for var in result._vars:
                        if hasattr(var, '_ommx_var') and var._ommx_var.id == var_id:
                            # Variable already exists, update coefficient
                            idx = result._vars.index(var)
                            result._coeffs[idx] += coeff
                            break
                    else:
                        # Variable not found, try to get it from the model
                        if hasattr(other, '_var'):
                            # Use the variable reference stored during multiplication
                            result._vars.append(other._var)
                            result._coeffs.append(coeff)
                        elif hasattr(other, '_var_id') and hasattr(other, '_model'):
                            # Use the stored variable ID to find the variable in the model
                            for var in other._model.getVars():
                                if hasattr(var, '_ommx_var') and var._ommx_var.id == other._var_id:
                                    result._vars.append(var)
                                    result._coeffs.append(coeff)
                                    break
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = LinExpr(self)
            result._coeffs = [coeff * other for coeff in result._coeffs]
            result._constant *= other
            return result
        else:
            raise TypeError("Multiplication only supported with scalar values")
    
    def __rmul__(self, other):
        return self.__mul__(other)


class Model:
    def __init__(self, name: str = ""):
        self._name = name
        self._vars = []
        self._var_counter = 0
        self._constrs = []
        self._var_map = {}  # Map variable IDs to Var objects
        self._objective = None
        self._sense = GRB.MINIMIZE  # Default sense
        self._solution = None  # Store the solution after optimization
        self._status = None  # Store optimization status

    def addVar(
        self,
        lb: float = 0.0,
        ub: float = float('inf'),
        vtype: str = GRB.CONTINUOUS,
        name: str = "",
    ) -> Var:
        if ub == float('inf') and vtype == GRB.BINARY:
            ub = 1.0
        decision_variable = DecisionVariable(
            _DecisionVariable(
                id=self._var_counter,
                bound=Bound(
                    lower=lb,
                    upper=ub,
                ),
                kind=GRB.map_types_grb_to_ommx(vtype),
                name=name,
            )
        )
        var = Var(self, decision_variable)
        self._vars.append(var)
        self._var_map[self._var_counter] = var
        self._var_counter += 1
        return var

    def getVars(self) -> List[Var]:
        return self._vars
        
    def getVarById(self, var_id: int) -> Optional[Var]:
        return self._var_map.get(var_id)

    def addConstr(self, constraint, name: str = ""):
        # constraint is the result of comparison operations like x + y <= 5
        constr = Constr(self, constraint, name)
        self._constrs.append(constr)
        return constr

    def getConstrs(self) -> List[Constr]:
        return self._constrs

    def getRow(self, constr: Constr) -> Row:
        return Row(self, constr._ommx_constraint)
        
    def update(self):
        # This would update the model after changes
        pass
        
    def optimize(self):
        # Convert the model to an OMMX instance
        ommx_instance = self.as_ommx_instance()
        
        try:
            # Create an OMMXGurobipyAdapter and solve the model
            adapter = OMMXGurobipyAdapter(ommx_instance)
            model = adapter.solver_input
            model.optimize()
            
            # Store the solution
            self._solution = adapter.decode(model)
            # Set status to OPTIMAL if solution exists
            self._status = GRB.OPTIMAL
        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")
        
    def setObjective(self, objective, sense=GRB.MINIMIZE):
        # Convert Var to LinExpr if needed
        if isinstance(objective, Var):
            expr = LinExpr()
            expr.addTerms(1.0, objective)
            self._objective = expr
        elif isinstance(objective, (int, float)):
            expr = LinExpr()
            expr._constant = float(objective)
            self._objective = expr
        else:
            # Assume it's already a LinExpr or ommx expression
            if hasattr(objective, 'raw'):
                # Convert ommx expression to LinExpr
                expr = LinExpr()
                if hasattr(objective, 'terms'):
                    for var_tuple, coeff in objective.terms.items():
                        if len(var_tuple) == 1:
                            var_id = var_tuple[0]
                            var = self.getVarById(var_id)
                            if var is not None:
                                expr.addTerms(coeff, var)
                self._objective = expr
            else:
                self._objective = objective
        
        self._sense = sense
        
    def getObjective(self):
        return self._objective
        
    @property
    def modelSense(self):
        return self._sense
    
    @modelSense.setter
    def modelSense(self, sense):
        self._sense = sense
        
    @property
    def ObjVal(self):
        # Return the objective value from the solution
        if self._solution is None:
            raise ValueError("Model not optimized")
        return self._solution.objective
        
    @property
    def Status(self):
        # Return the optimization status
        if self._solution is not None:
            return GRB.OPTIMAL
        return None
        
    def as_ommx_instance(self):
        # Convert the model to an ommx.v1.Instance
        objective = 0
        sense = Instance.MINIMIZE
        
        if self._objective is not None:
            # Convert LinExpr to ommx expression
            terms = {}
            for i in range(self._objective.size()):
                var = self._objective.getVar(i)
                coeff = self._objective.getCoeff(i)
                terms[var._ommx_var.id] = coeff
            
            linear = Linear(terms=terms)
            if hasattr(self._objective, '_constant') and self._objective._constant != 0:
                linear = linear + self._objective._constant
            
            objective = linear
            
            # Set the sense
            if self._sense == GRB.MAXIMIZE:
                sense = Instance.MAXIMIZE
        
        return Instance.from_components(
            decision_variables=[var._ommx_var for var in self._vars],
            constraints=[constr._ommx_constraint for constr in self._constrs],
            objective=objective,
            sense=sense,
        )
