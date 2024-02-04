import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from fractions import Fraction
from math import floor

class cutStock1D():
    def __init__(self, orders, demand, capacity, c):
        self.solution = None
        self.orders = orders
        self.demand = demand
        self.capacity = capacity
        self.c = c
        self.A = None
        self.obj = 0
        self.z = None
        self.patterns = self.initPatterns(self.demand, self.capacity)
        self.forbidden = None

    def initPatterns(self, demand, capacity):
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

        i = 0
        j = 0
        #patterns = np.zeros((len(demand), 1))   # Initial Pattern
        a = capacity[0]/demand
        a = a.astype(int)
        patterns = np.diag(a)

        # Add 1's for each demand in a pattern until all demands are accounted for
        '''while i < len(demand):
            if (capacity[0] - patterns[:, j]@demand - demand[i]) > 0:
                # Still capacity left in current pattern to fill this demand
                patterns[i, j] = 1
                i += 1
            else:
                # Current pattern filled, add another pattern
                j += 1
                new_col = np.zeros((len(demand), 1))
                patterns = np.append(patterns, new_col, 1)
        '''
        self.solution = np.ones(patterns.shape[1])

        return patterns

    def swapPattern(self, p_j):
        """ With a pattern p_j that can enter the basis, determine which pattern
            will leave the basis. Need to calculate Theta and find the min value
            of theta. The minimum value of Theta will be the index of the pattern
            that will leave the basis.
    
            Inputs: p_j   - a pattern that can enter the basis. ndarray of size(n,1).
    
            Output: self.patterns is updated to reflect the new patterns in the basis.
       """  # Solve the LP with current set of patterns
        z_bar, l, _, _, _ = self.solveRelaxed()

        # Determine theta
        p_bar_j = np.matmul(np.linalg.inv(self.patterns), p_j)

        # Find the minimum value (exclusive of 0's) of z_bar/p_barj to determine which pattern leaves the basis
        z = np.divide(z_bar, p_bar_j, out=np.zeros_like(z_bar), where=p_bar_j > 0)

        # Determine the index of this minimum value to put new pattern in that column
        idx = np.where(z == np.min(z[np.nonzero(z)]))[0][0]

        #self.patterns[:, idx] = p_j

    def addPattern(self, newColumn):
        """ Add new column to the

            Inputs: newColumn - a pattern that can enter the basis. ndarray of size (n,1).

            Output: self.patterns is updated to reflect the new patterns in the basis.
       """
        # Add pattern p_j to the patterns matrix
        self.patterns = np.hstack((self.patterns, newColumn.reshape(-1, 1)))
        self.solution = np.append(self.solution, 1)

    def removePattern(self):
        """ With a pattern p_j that can enter the basis, determine which pattern
            will leave the basis. Need to calculate Theta and find the min value
            of theta. The minimum value of Theta will be the index of the pattern
            that will leave the basis.

            Inputs: p_j   - a pattern that can enter the basis. ndarray of size (n,1).

            Output: self.patterns is updated to reflect the new patterns in the basis.
       """
        # Add pattern p_j to the patterns matrix
        self.patterns = np.delete(self.patterns, -1, axis=1)

    def solveRelaxed(self, integer=False):
        """ Solve the relaxed LP problem of minimizing sum(c*X_j) given the current patterns.

            Output: solution   - solution to the relaxed LP problem. ndarray of size(n)
        """
        n = self.patterns.shape[0]
        num_patterns = self.patterns.shape[1]

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
            X = [solver.IntVar(0, np.max(self.orders.astype(np.double)), f'x_{i}') for i in range(num_patterns)]
        else:
            X = [solver.NumVar(0.0, np.max(self.orders.astype(np.double)), f'x_{i}') for i in range(num_patterns)]

        # There are a couple of ways to frame the cost. We can look at minimizing the number of paterns
        # In the case of a single sized length of fiber will be the same as minimizing waste.
        # We can also look at minimizing the waste, represented by the size of the length of the spool * the
        # number of patterns minus the amount of the ordered lengths.
        #cost1 = sum(X[j] for j in range(num_patterns))
        cost2 = self.capacity[0] * sum(X[j] for j in range(num_patterns)) - sum(self.demand[k]*self.orders[k] for k in range(n))
        solver.Minimize(cost2)

        # Create the constraints, one per row in patterns - sum(A_ij*X_j) <= orders_i
        # Constraint requires type double, so need to cast to type double
        for i in range(n):
            if integer:
                constraint.append(solver.Add(sum(X[j] * self.patterns[i, j] for j in range(num_patterns)) >=
                                             self.orders[i]))
            else:
                constraint.append(solver.Add(sum(X[j] * self.patterns[i, j] for j in range(num_patterns)) ==
                                             self.orders[i].astype(np.double)))

        status = solver.Solve()

        # Check that the problem has an optimal solution.
        if status != solver.OPTIMAL:
            print('The problem does not have an optimal solution!')
            if status == solver.FEASIBLE:
                print('A potentially suboptimal solution was found.')
            else:
                print('The solver could not solve the problem.')

        # Create array of solution values
        solution = np.array([X[i].SolutionValue() for i in range(num_patterns)])

        if integer:
            dual = None
        else:
            dual = np.array([constraint[i].DualValue() for i in range(n)])
        self.obj = solver.Objective().Value()

        self.solution = solution
        return solution, dual, status, self.obj, solver

    def solveKnapsack(self, yi):
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            return -2

        n = len(yi)
        new_pattern = [solver.IntVar(0, self.orders[i].astype(np.double), '') for i in range(n)]

        # maximizes the sum of the values times the number of occurrence of that roll in a pattern
        Cost1 = solver.Sum(yi[i] * new_pattern[i] for i in range(n))
        solver.Maximize(Cost1)

        # ensuring that the pattern stays within the total length of the large reel
        solver.Add(sum(self.demand[i] * new_pattern[i] for i in range(n)) <= self.capacity[0])

        status = solver.Solve()
        col = np.array([item.SolutionValue() for item in new_pattern])

        return solver.Objective().Value(), col


orders = np.array([22, 25, 12, 14, 18, 18, 20, 10, 12, 14, 16, 18, 20])
demand = np.array([1380, 1520, 1560, 1710, 1820, 1880, 1930, 2000, 2050, 2100, 2140, 2150, 2200])
capacity = np.array([5600])
c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

time = 0
cut = cutStock1D(orders, demand, capacity, c)
sol, dual, status, obj, _ = cut.solveRelaxed()
knapobj, col = cut.solveKnapsack(dual)


i = 0
for i in range(53):
    cut.addPattern(col)

    sol, dual, status, obj, _ = cut.solveRelaxed()
    if status != 0:
        cut.swapPattern(col)
        # cut.removePattern()
        print("REMOVED: ", cut.patterns)
        break

    knapobj, col = cut.solveKnapsack(dual)

    if knapobj <= 5600:
        print("FOUND SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("I: ", i)
        break

sol, dual, status, obj, solver = cut.solveRelaxed(integer=True)
print("SOL FP: ", sol, i)
print("DUAL: ", dual)
print(cut.patterns@sol)

# Find the indexes where sol has non-zero values for the patterns
print(cut.patterns[:, sol != 0].shape)

# Find the remnant amounts
print(5600 - (demand@cut.patterns))
print(sum((5600 - (demand@cut.patterns[:, sol != 0]))*sol[sol != 0]))
print("=====Stats:======")
# print(solver.SolutionInfo())
print("=====Response:======")
# print(solver.ResponseStats())
#%%
