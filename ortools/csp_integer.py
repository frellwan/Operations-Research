import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from fractions import Fraction
from math import floor


class CutStock1D(object):
    """
    Class for solving a one-dimensional cutting stock problem.

    Parameters:
    -----------
    orders : list
        A list of integers representing the demand for each length.
    lengths : list
        A list of integers representing the possible lengths of stock.
    capacity : int
        An integer representing the capacity of the reels used to cut the stock.
    cost : list or None, optional
        A list of integers representing the cost of each reel. If None, the cost is set to 1.

    Attributes:
    -----------
    patterns : list
        A list of lists representing the initial set of patterns used to solve the problem.
    c : list
        A list of floats representing the cost of each pattern.

    Methods:
    --------
    initPatterns(W)
        Create a matrix that will be used as the initial set of patterns.
    addColumn(column, idx)
        Add a new column to the matrix of patterns.
    solveRMP(integer=False)
        Solve the relaxed master problem given a set of patterns and orders.

    """
    def __init__(self, orders, demand, capacity, cost):
        self.orders = orders
        self.demand = demand
        self.capacity = capacity
        self.c = cost
        self.A = None
        self.obj = 0
        self.z = None
        self.patterns = self.initPatterns(self.demand, self.capacity)
        self.forbidden = None

    def initPatterns(self, demand, capacity):
        """ Create a diagonal matrix that will be used as the initial set of patterns

            Inputs: demand   - Demand for each length. ndarray of size (n,n)
                    capacity - capacity of reels. ndarray of size(m)

            Output: patterns - starting point for cut patterns
       """
        # TODO
        # Use Fast or Best Fit Decreasing to initialize patterns

        patterns = np.zeros((len(capacity), len(demand), np.sum(demand)//np.max(capacity)+2))

        # Add 1's in each pattern until the capacity is used up
        for c in capacity:
            i = 0
            j = 0
            while i < len(demand) and j < len(patterns[1]):
                if (capacity[c] - patterns[c, :, j]@demand - demand[i]) > 0:
                    # Still capacity left to fill next demand
                    patterns[i, j] = 1
                    i += 1
                    j = 0
                else:
                    # Current pattern filled, use next pattern
                    j += 1

            return patterns


    def swapPattern(self, p_j):
        """ With a pattern p_j that can enter the basis, determine which pattern
            will leave the basis. Need to calculate Theta and find the min value
            of theta. The minimum value of Theta will be the index of the pattern
            that will leave the basis.
   
            Inputs: p_j   - a pattern that can enter the basis. ndarray of size (n).
   
            Output: self.patterns is updated to reflect the new patterns in the basis.
       """  # Solve the LP with current set of patterns
        z_bar, l, _, _ = self.solveRelaxed()

        # Determine theta
        p_bar_j = np.matmul(np.linalg.inv(self.patterns), p_j)

        # Find the minimum value (exclusive of 0's) of z_bar/p_barj to determine which pattern leaves the basis
        print("SWAP Z: ", z_bar, z_bar.shape)
        print("SWAP P: ", p_bar_j, p_bar_j.shape)
        z = np.divide(z_bar, p_bar_j, out=np.zeros_like(z_bar), where=p_bar_j > 0)
        print('SWAP z: ', z)

        # Determine the index of this minimum value to put new pattern in that column
        idx = np.where(z == np.min(z[np.nonzero(z)]))[0][0]
        print('SWAP idx to change: ', idx)
        #self.patterns[:, idx] = p_j

    def addPattern(self, newColumn):
        """ With a pattern p_j that can enter the basis, determine which pattern
            will leave the basis. Need to calculate Theta and find the min value
            of theta. The minimum value of Theta will be the index of the pattern
            that will leave the basis.

            Inputs: p_j   - a pattern that can enter the basis. ndarray of size (n,1).

            Output: self.patterns is updated to reflect the new patterns in the basis.
       """
        # Add pattern p_j to the patterns matrix
        self.patterns = np.hstack((self.patterns, newColumn.reshape(-1, 1)))

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

        cost = sum(X[j] for j in range(num_patterns))
        solver.Minimize(cost)

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

        return solution, dual, status, self.obj

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


cut = CutStock1D(orders, demand, capacity, c)
sol, dual, status, obj = cut.solveRelaxed()
knapobj, col = cut.solveKnapsack(dual)

for i in range(54):
    cut.addPattern(col)

    sol, dual, status, obj = cut.solveRelaxed()
    if status != 0:
        cut.swapPattern(col)
        #cut.removePattern()
        print("REMOVED: ", cut.patterns)
        break

    knapobj, col = cut.solveKnapsack(dual)

    if knapobj <= 1.000000001:
        print("FOUND SOLUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("I: ", i)
        break

sol, dual, status, obj = cut.solveRelaxed(integer=True)
print("SOL FP: ", sol)
print("SOL: ", np.floor(sol))
print("DUAL: ", dual)
print(cut.patterns@sol)
print((demand@cut.patterns) - 5600)