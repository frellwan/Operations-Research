import numpy as np
from ortools.linear_solver import pywraplp


class cutStock1D():
    def __init__(self, orders, demand, capacity, c):
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
        """ Create a diagonal matrix that will be used as the initial set of patterns

           Inputs: demand   - Demand for each length. ndarray of size (n,n)
                 capacity - capacity of reels. ndarray of size(m)

           Output: patterns - starting point for cut patterns
        """
        # TODO
        # Use Fast or Best Fit Decreasing to initialize patterns

        i = 0
        j = 0
        patterns = np.zeros((len(demand), np.sum(demand) // capacity[0] + 2))

        # Add 1's in each pattern until the capacity is used up
        for c in range(len(capacity)):
            while i < len(demand) and j < len(patterns[1]):
                if (capacity[c] - patterns[c, :, j] @ demand - demand[i]) > 0:
                    # Still capacity left to fill next demand
                    patterns[i, j] = 1
                    i += 1
                    j = 0
                else:
                    # Current pattern filled, use next pattern
                    j += 1

        return patterns

    def addPattern(self, newColumn):
        """ With a pattern p_j that can enter the basis, determine which pattern
           will leave the basis. Need to calculate Theta and find the min value
           of theta. The minimum value of Theta will be the index of the pattern
           that will leave the basis.

           Inputs: newColumn - a pattern that can enter the basis. ndarray of size (n,1).

           Output: patterns is updated to reflect the new patterns in the basis.
        """
        # Add pattern p_j to the patterns matrix
        for c in range(len(self.capacity)):
            self.patterns = np.hstack((self.patterns[c], newColumn.reshape(-1, 1)))

    def solveRelaxed(self, integer=False):
        """ Solve the relaxed LP problem of minimizing sum(c*X_j) given the current patterns.

           Output: solution   - solution to the relaxed LP problem. ndarray of size(n)
        """
        page = self.patterns.shape[0]
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
            X = [[solver.IntVar(0, np.max(self.orders.astype(np.double)), f'x_{c}_{i}') for i in range(num_patterns)]
                 for c in range(page)]
        else:
            X = [[solver.NumVar(0.0, np.max(self.orders.astype(np.double)), f'x_{c}_{i}') for i in range(num_patterns)]
                 for c in range(page)]

        cost = sum(X[c][j] for j in range(num_patterns) for c in range(page))
        solver.Minimize(cost)

        # Create the constraints, one per row in patterns - sum(A_ij*X_j) <= orders_i
        # Constraint requires type double, so need to cast to type double
        for i in range(n):
            if integer:
                constraint.append(solver.Add(sum(X[c][j] * self.patterns[i, j] for j in range(num_patterns)
                                                 for c in range(page)) >= self.orders[i]))
            else:
                constraint.append(solver.Add(sum(X[c][j] * self.patterns[i, j] for j in range(num_patterns)
                                                 for c in range(page)) == self.orders[i].astype(np.double)))

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