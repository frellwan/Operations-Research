import pulp as pl
import numpy as np
import logging

class cutStock1D():
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
    def __init__(self, orders, lengths, capacity, cost=None):
        self.orders = orders
        self.lengths = lengths
        self.capacity = capacity
        if cost == None:
            self.cost = [1]*len(capacity)
        else:
            self.cost = cost
        self.patterns, self.c = self.initPatterns(max(capacity))

    def initPatterns(self, W):
        """
        Create a matrix that will be used as the initial set of patterns.

        Parameters:
        -----------
        W : int
            An integer representing the largest capacity of the reels used to cut the stock.

        Returns:
        --------
        patterns : list
            A list of lists of integers representing the initial set of patterns.
        c : list
            A list of floats representing the cost of each pattern.
        """
        # Number of rows for the different lengths
        n = len(self.lengths)

        # Sort w in descending order
        w_sort = sorted(self.lengths, reverse=True)
        w_idx = sorted(range(n), key=lambda k: -self.lengths[k])

        # Initialize R, ib and A
        R = set(range(n))
        ib = np.array(self.lengths, dtype=float)
        A = np.zeros((n, n))

        # Find initial solution
        for j in range(n):
            a = np.zeros(n)
            for i in R:
                if w_sort[i] <= W - np.dot(w_sort, a):
                    a[i] = np.floor((W - np.dot(w_sort, a))/w_sort[i])

            A[:, j] = a[w_idx]
            result = np.divide(ib, a, out=np.zeros_like(a), where=a!=0)
            x = np.min(result[np.nonzero(result)])
            idxR = np.where(result == x)[0][0]
            ib -= x*a
            R.remove(idxR)

            if len(R)==0:
                break

        if len(capacity) == 1:
            c = [self.cost[0]]*n
        else:
            idxC = np.where(np.array(self.capacity)==W)[0][0]
            c = [self.cost[idxC] for i in range(n)]
            reel = np.zeros((len(self.capacity), n))
            reel[idxC, :] = 1
            A = np.vstack((A, reel))

        return A.tolist(), c

    def addColumn(self, column, idx):
        """
        Add a new column to the matrix of patterns.

        Parameters:
        -----------
        column : list
            A list of integers representing the new column to add.
        idx : int
            An integer index indicating the reel this column corresponds to.

        Returns:
        --------
        patterns : list
            A list of lists of integers representing the updated patterns.
        c : list
            A list of floats representing the updated cost vector.
        """
        # Add column to pattern matrix
        for i in range(len(self.patterns)):
            if i < len(self.lengths):
                # Add pattern to top part of patterns
                self.patterns[i].append(column[i])
            elif i == len(self.lengths) + idx:
                # Add 1 to correspond to correct raw
                self.patterns[i].append(1.0)
            else:
                # Add 0 to all other elements
                self.patterns[i].append(0.0)

        # Add cost of the particular raw this column corresponds to
        self.c.append(self.cost[idx])

        return self.patterns, self.c

    def solveRMP(self, integer=False):
        """Solve the relaxed master problem given a set of patterns and orders.

        Args:
            integer: If True, solve the integer programming problem, otherwise solve
                the linear programming problem. Default is False.

        Returns:
            A tuple containing:
            - solution: A list containing the solution to the relaxed master problem.
            - dual: A list containing the dual variables.
            - obj: The value of the objective function.
            - status: The status of the optimization (optimal, infeasible, etc.).
        """
        solver = pl.COIN_CMD(msg=0)
        if integer:
            vartype = pl.LpInteger
        else:
            vartype = pl.LpContinuous

        model = pl.LpProblem("RMP", pl.LpMinimize)

        n = len(self.lengths)
        num_patterns = len(self.patterns[0])

        # Declare an array to hold our variables.
        X = [pl.LpVariable(f'x_{i}', lowBound=0, cat=vartype) for i in range(num_patterns)]

        model += pl.lpSum(self.c[i]*X[i] for i in range(num_patterns))

        # Create the constraints, one per row in patterns - sum(A_ij*X_j) == orders_i
        # Constraint requires type double, so need to cast to type double
        if integer:
            for i in range(n):
                model += pl.lpSum(X[j] * self.patterns[i][j] for j in range(num_patterns)) == self.orders[i]
        else:
            for i in range(n):
                model += pl.lpSum(X[j] * self.patterns[i][j] for j in range(num_patterns)) == self.orders[i]

        model.solve(solver)

        # Create array of solution values
        solution = [X[i].varValue for i in range(num_patterns)]
        obj = pl.value(model.objective)
        status = pl.LpStatus[model.status]

        dual = []
        for name, constr in list(model.constraints.items()):
            dual.append(constr.pi)

        return solution, dual, obj, status

    def pricing(self, dual, capacity):
        """Solves the pricing subproblem to generate a new pattern that will improve
        the objective function of the master problem.

        Inputs:
        - dual: a list of dual variables, corresponding to the constraints of the master problem
        - capacity: capacity of the reel

        Returns:
        - new_pattern: a list representing the new pattern generated by the pricing subproblem
        - obj: the objective value of the pricing subproblem
        - status: the status of the optimization (e.g. Optimal, Infeasible, etc.)
        """
        solver = pl.COIN_CMD(msg=0)
        model = pl.LpProblem("Pricing", pl.LpMaximize)

        n = len(self.lengths)

        # Declare an array to hold our variables.
        newPattern = [pl.LpVariable(f'y_{i}', lowBound=0, cat=pl.LpInteger) for i in range(n)]


        # Define the Objective function
        model += pl.lpSum(dual[i] * newPattern[i] for i in range(n))

        # Add contraints
        model += pl.lpSum(self.lengths[i] * newPattern[i] for i in range(n)) <= capacity

        # Solve the model
        model.solve(solver)

        # Create array of solution values
        solution = [newPattern[i].varValue for i in range(n)]
        obj = pl.value(model.objective)
        status = pl.LpStatus[model.status]

        return solution, obj, status




# define the input parameters
n = 6 # number of orders
m = 3 # number of stock material lengths
capacity = [10, 6, 4] # lengths of the stock materials
cost = [1.9, 1, 0.6]
lengths   = [2, 3, 5, 8] # widths of the orders
orders   = [400, 350, 200, 150] # demands for each order

#capacity=[100]
#lengths=[45,36,31,14]
#orders=[97, 610, 395, 211]
#cost=[1]

cut = cutStock1D(orders, lengths, capacity, cost)


for i in range(10):
    sol, dual, obj, status = cut.solveRMP()
    print("STATUS: ", sol)
    print("OBJ:     ", obj)
    print("DUAL:   ", dual)
    print(status == 'Optimal')

    patterns = []
    objs = []
    for i in range(len(cut.capacity)):
        solk, objk, statk = cut.pricing(dual, cut.capacity[i])
        patterns.append(solk)
        objs.append(objk)

    print(patterns)
    print(objs)
    reducedCost = np.array(objs)-np.array(cut.cost)
    idx = np.argmax(reducedCost)
    print("IDX: ", idx)
    newPattern = patterns[idx]
    maxCost = reducedCost[idx]
    if maxCost > 0.00001:
        print("Adding Column")
        A, c = cut.addColumn(newPattern, idx)
    else:
        #A = cut.addColumn(newPattern, idx)
        print("SOLUTION FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        break

    print("KOBJ: ", objk)
    print("COL: ", solk)
    print("STATUSK: ", statk)

sol, dual, obj, status = cut.solveRMP(integer=True)


