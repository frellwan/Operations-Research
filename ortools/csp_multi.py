import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from fractions import Fraction
from math import floor


class cutStock1D():
    def __init__(self, orders, demand, capacity):
        self.solution = None
        self.orders = orders
        self.demand = demand
        self.capacity = capacity
        self.patterns = None
        self.obj = 0

        self.initPatterns()

    def initPatterns(self):
        """ Create a diagonal matrix that will be used as the initial set of patterns.
			Initial matrix doesn't seem to have much of an effect on the time to determine
			an optimal solution, so we stick with a simple diagonal matrix meeting the demands
			given the capacity.

			Inputs: demand   - Demand for each length. ndarray of size (1,n)
					capacity - capacity of reels. ndarray of size(m)

			Output: patterns - starting point for cut patterns. ndarray(n,x)
		"""
        # TODO
        # Use Fast or Best Fit Decreasing to initialize patterns
        self.patterns = np.zeros((len(self.capacity), len(self.demand), len(self.demand)))   # Initial Pattern

        for c in range(len(self.capacity)):
            a = self.capacity[c]//self.demand
            self.patterns[c] = np.diag(a)

        self.solution = np.ones(self.patterns.shape[2])

    def addPattern(self, newColumn):
        """ Add new column to the

			Inputs: newColumn - a pattern that can enter the basis. ndarray of size (n,m) where n is the
								number of inventory items and m is the number of ordered lengths

			Output: self.patterns is updated to reflect the new patterns in the basis.
		"""
        # Add pattern p_j to the patterns matrix
        #self.patterns = np.concatenate((self.patterns, newColumn.reshape(2,13,-1)), axis=2)
        self.patterns = np.concatenate((self.patterns, newColumn[:, :, np.newaxis]), axis=2)
    #self.patterns = np.hstack((self.patterns, newColumn.reshape(2, 13, -1)))
    #self.solution = np.append(self.solution, 1)

    def solveRelaxed(self, integer=False):
        """ Solve the relaxed LP problem of minimizing sum(c*X_j) given the current patterns.

			Output: solution   - solution to the relaxed LP problem. ndarray of size(n)
		"""
        num_raws = self.patterns.shape[0]
        n = self.patterns.shape[1]
        num_patterns = self.patterns.shape[2]

        if integer:
            solver = pywraplp.Solver.CreateSolver('SCIP')

        else:
            solver = pywraplp.Solver.CreateSolver('GLOP')
            solver.SetSolverSpecificParametersAsString("solution_feasibility_tolerance:22")

        if not solver:
            return -1

        constraint = []
        # Declare an array to hold our variables.
        if integer:
            X = [[solver.IntVar(0, np.max(self.orders.astype(np.double)), f'x_{r,p}') for p in range(num_patterns)]
                 for r in range(num_raws)]
        else:
            X = [[solver.NumVar(0.0, np.max(self.orders.astype(np.double)), f'x_{r,p}') for p in range(num_patterns)]
                 for r in range(num_raws)]

        # There are a couple of ways to frame the cost. We can look at minimizing the number of patterns
        # In the case of a single sized length of fiber will be the same as minimizing waste (knapobj <= 1).
        # We can also look at minimizing the waste, represented by the size of the length of the spool * the
        # number of patterns minus the amount of the ordered lengths (knapobj <= capacity).
        cost1 = sum([X[i][j] for i in range(num_raws) for j in range(num_patterns)])
        # cost2 = self.capacity[c] - sum(patterns[c, k, p] * self.demand[k] for k in range(n))
        cost3 = sum(sum(X[r][p]*self.capacity[r] - X[r][p]*sum(self.patterns[r, i, p] * self.demand[i] for i in range(n)) for p in range(num_patterns)) for r in range(num_raws))
        # cost4 = self.capacity[0] * sum(X[j] for j in range(num_patterns)) - sum(self.demand[k]*self.orders[k] for k in range(n))
        solver.Minimize(100000*cost1 + cost3)

        # Create the constraints, one per row in patterns - sum(A_ij*X_j) <= orders_i
        # Constraint requires type double, so need to cast to type double
        for i in range(n):
            if integer:
                constraint.append(solver.Add(sum([X[r][p] * self.patterns[r, i, p] for r in range(num_raws)
                                                  for p in range(num_patterns)]) >= self.orders[i]))
            else:
                constraint.append(solver.Add(sum([X[r][p] * self.patterns[r, i, p] for r in range(num_raws)
                                                  for p in range(num_patterns)]) == self.orders[i].astype(np.double)))

        status = solver.Solve()

        # Create array of solution values
        solution = np.array([[X[r][j].SolutionValue() for j in range(num_patterns)] for r in range(num_raws)])

        if integer:
            dual = None
        else:
            dual = np.array([constraint[i].DualValue() for i in range(n)])
        self.obj = solver.Objective().Value()

        self.solution = solution
        return solution, dual, status, self.obj, solver

    def solveKnapsack(self, yi):
        col = np.zeros((len(self.capacity), len(yi)))

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if solver is None:
            return -2

        n = len(yi)

        new_pattern = [solver.IntVar(0, self.orders[i].astype(np.double), '') for i in range(n)]
        u = solver.IntVar(0, np.max(self.orders.astype(np.double)), 'u')
        l = solver.IntVar(0, np.max(self.orders.astype(np.double)), 'l')
        d = [solver.IntVar(0, 1, f'd{i}') for i in range(len(self.capacity))]

        # maximizes the sum of the values times the number of occurrence of that roll in a pattern
        Cost1 = solver.Sum(yi[i] * new_pattern[i] for i in range(n))
        solver.Minimize(l - u - Cost1)

        # ensuring that the pattern stays within the total length of the large reel
        # solver.Add(sum(self.demand[i] * new_pattern[i] for i in range(n)) <= self.capacity[c])
        solver.Add(u == sum(self.demand[i] * new_pattern[i] for i in range(n)))
        solver.Add(l == sum(self.capacity[k]*d[k] for k in range(len(self.capacity))))
        solver.Add(sum(d[k] for k in range(len(self.capacity))) == 1)
        solver.Add(u <= l)

        status = solver.Solve()
        obj = solver.Objective().Value()
        row = [item.SolutionValue() for item in d]
        col[np.where(row == 1)[0][0]] = np.array([item.SolutionValue() for item in new_pattern])

        return obj, col

orders = np.array([22, 25, 12, 14, 18, 18, 20, 10, 12, 14, 16, 18, 20])
demand = np.array([1380, 1520, 1560, 1710, 1820, 1880, 1930, 2000, 2050, 2100, 2140, 2150, 2200])
# capacity = np.array([5600, 5400, 5300, 5200, 5400, 5300, 5400, 5300, 5400, 5350, 5375, 5450, 5475])
capacity = np.array([5600, 5400])
time = 0
cut = cutStock1D(orders, demand, capacity)
sol, dual, status, obj, _ = cut.solveRelaxed()
knapobj, col = cut.solveKnapsack(dual)


i = 0
for i in range(50):
    cut.addPattern(col)

    sol, dual, status, obj, solver = cut.solveRelaxed()

    # Check that the problem has an optimal solution.
    if status != solver.OPTIMAL:
        print('The problem does not have an optimal solution!')
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found.')
            break
        else:
            print('The solver could not solve the problem.')
            break

    knapobj, col = cut.solveKnapsack(dual)
    print(knapobj)

    if all(b <= 1 for b in knapobj):
        print("FOUND SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("I: ", i)
        break

#sol, dual, status, obj, solver = cut.solveRelaxed(integer=True)

# Check that the problem has an optimal solution.
if status != solver.OPTIMAL:
    print('The problem does not have an optimal solution!')
    if status == solver.FEASIBLE:
        print('A potentially suboptimal solution was found.')
    else:
        print('The solver could not solve the problem.')

print("SOL FP: ", sol, i)
print("DUAL: ", dual)
print(cut.patterns@sol[0])
print(cut.patterns@sol[1])

# Find the indexes where sol has non-zero values for the patterns
print(sol.shape, cut.patterns.shape)
print(sol[1].shape)
print(cut.patterns[0, :, sol[0] != 0].T.shape)
print(cut.patterns[1, :, sol[1] != 0].T.shape)

# Find the remnant amounts
print(sum((5600 - (demand@cut.patterns[0, :, sol[0] != 0].T)) * sol[0][sol[0] != 0]))
print(sum((5400 - (demand@cut.patterns[1, :, sol[1] != 0].T)) * sol[1][sol[1] != 0]))
# print(sum((5600 - (demand@cut.patterns[0, :, sol[1] != 0]))*sol[1][sol[1] != 0]))
print("=====Stats:======")
# print(solver.SolutionInfo())
print("=====Response:======")
# print(solver.ResponseStats())


#%%
