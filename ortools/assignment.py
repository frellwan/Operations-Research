from ortools.sat.python import cp_model
import pandas as pd
from threading import Timer, Lock


class EarlyStoppingLogger(cp_model.CpSolverSolutionCallback):
    """
    Stop solving if the solution does not improve within time bound

    Arguments:
    - variables: list of variables to display at each solution
    - early_stopping_timeout: stop solving if this many seconds pass without improvement
    """

    def __init__(self, variables, early_stopping_timeout=10):
        super().__init__()
        self.__variables = variables
        self.__solution_count = 0
        self.early_stopping_timeout = early_stopping_timeout
        self.timer = None
        self.lock = Lock()

    def on_solution_callback(self):
        """Log solution and set next timeout"""
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = Timer(self.early_stopping_timeout, self.stop)
            self.timer.start()
        self.__solution_count += 1
        for v in self.__variables:
            # print(f"{v}={self.Value(v)}", end=" ")
            print('Solution %i, time = %f s, objective = %i' % (self.__solution_count, self.WallTime(), self.ObjectiveValue()))

    def stop(self):
        """Stop solving"""
        self.StopSearch()

    def clean(self):
        """Clean up timer"""
        if self.timer:
            self.timer.cancel()

    def solution_count(self):
        return self.__solution_count


def solve_assignment(cost, groups, early_stopping=None):
    """
    Solve assignment problem of reducing cost of assigning reel to groups
    Arguments:  cost: cost matrix of assigning item
                groups: list of groups
                early_stopping: If this is not None, this is the timeout for early stopping

    Returns:    solution_list: list of dictionaries [Reel, Color, Group, Cost] minimizing the costs,
                status:  status of solver - string (OPTIMAL, FEASIBLE, INFEASIBLE)
    """
    maxL = max([max(map(int, i)) for i in cost])
    num_colors = len(cost)
    num_reels = len(cost[1])
    num_groups = len(groups)
    all_colors = range(num_colors)
    all_reels = range(num_reels)
    all_groups = range(num_groups)
    UB = cp_model.INT32_MAX

    # _____________________
    # Create Model Object
    # ---------------------
    model = cp_model.CpModel()

    # -----------------------
    # Create Model Variables
    # -----------------------
    # objective is our objective variable
    objective = model.NewIntVar(0, UB, 'objective')

    # Helper variables to find minimum non-zero value
    y = [model.NewIntVar(0, maxL, f'yi_{g}') for g in all_groups]

    # min_length[g] * M (length of group[g] (used for objective)
    m_min_length = [model.NewIntVar(0, maxL*num_groups, f'max_length_{g}') for g in all_groups]
    sum_g = [model.NewIntVar(0, maxL*num_colors, f'sum_{g}') for g in all_groups]

    # Indicator Variables for color/reel in a group x[group, color, reel]
    # a = model.NewBoolVar('a')
    # b = model.NewBoolVar('b')

    X = [[[model.NewBoolVar(f'x[{g},{c},{r}]') for r in all_reels] for c in all_colors] for g in all_groups]
    # G = [[model.NewIntVar(0, maxL, f'G[{c}{r}') for r in all_reels] for c in all_colors]

    # -------------------------
    # Create Model Constraints
    # -------------------------
    # Each group is assigned at most one reel from each color
    for g in all_groups:
        for c in all_colors:
            model.Add(cp_model.LinearExpr.Sum([X[g][c][r] for r in all_reels]) <= 1)

    # Each reel is assigned to exactly one group (0 length reels don't get assigned)
    for c in all_colors:
        for r in all_reels:
            if cost[c][r] != 0:
                model.Add(cp_model.LinearExpr.Sum([X[g][c][r] for g in all_groups]) == 1)

    # Must have the correct number of reels in each group
    for g in all_groups:
        model.Add(cp_model.LinearExpr.Sum([X[g][c][r] for r in all_reels for c in all_colors]) == groups[g])

    # Determine the smallest non-zero length in the group - implication enforces only looking at non-zero reels
    for g in all_groups:
        for c in all_colors:
            for r in all_reels:
                model.Add(y[g] <= cost[c][r]).OnlyEnforceIf(X[g][c][r])

    # Determine M*min_length in a group (to use in cost function)
    for g in all_groups:
        model.Add(sum_g[g] == cp_model.LinearExpr.Sum([cost[c][r]*X[g][c][r] for c in all_colors for r in all_reels]))
        model.AddMultiplicationEquality(m_min_length[g], y[g], groups[g])

    # --------------
    # Break symmetry
    # --------------
    # TODO - Find ways to break symmetry
    # Can swap if abs(y[g1][c1][r1] - y[g2][c2][r2]) <= y[g2][c2][r2]-min_g[2] and y[g1][c1][r1]-min_g[1]
    """
    abs_values = [[model.NewIntVar(0, maxL, f'abs_values[{c},{a}') for a in range(num_reels*(num_reels-1)//2)] for c in all_colors]
    for c in all_colors:
        i = 0
        for r1 in all_reels:
            for r2 in range(r1+1, num_reels):
                if groups[r1] == groups[r2]:
                    v = model.NewIntVar(-maxL, maxL, "v")     # Temporary variable
                    model.Add(v == G[c][r1] - G[c][r2])
                    model.AddAbsEquality(abs_values[c][i], v)
                    model.Add(abs_values[c][i] <= G[c][r1]-y[r1]).OnlyEnforceIf(a)
                    model.Add(abs_values[c][i] <= G[c][r2]-y[r2]).OnlyEnforceIf(b)
                    model.Add(G[c][r1] >= G[c][r2]).OnlyEnforceIf([a, b])
                
                i += 1
    """
    # ----------
    # Objective
    # ----------
    # Objective cost - Minimize the differences between reel lengths in each group (same as min(sum_g-m*min))
    model.Add(objective == cp_model.LinearExpr.Sum([sum_g[g] - m_min_length[g] for g in all_groups]))
    model.Minimize(objective)

    # Could also use the following - different objective value, but same end result (doesn't appear to effect speed)
    # model.Add(objective == sum(m_min_length[g] for g in all_groups))
    # model.Maximize(objective)

    # ---------
    # Add Hint
    # ---------
    '''
    r = 0
    for g in all_groups:
        for c in all_colors:
            if cost[c][r] != 0:
                model.AddHint(X[g][c][r], 1)
        r += 1
    '''

    # Setup Early Stopping when solutions don't improve within time limit (s)
    if early_stopping is not None:
        solution_printer = EarlyStoppingLogger([objective], early_stopping)

    # -------------------------------------
    # Create a solver and solve the model
    # -------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.min_num_lns_workers = 2
    # solver.parameters.keep_all_feasible_solutions_in_presolve = True

    # Sets a time limit of 5 minutes.
    solver.parameters.max_time_in_seconds = 300000.0

    # solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
    solver.parameters.log_search_progress = True
    # solver.parameters.keep_all_feasible_solutions_in_presolve = True

    # Solve the Model
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    solution_printer.clean()

    solution_list = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for g in all_groups:
            for c in all_colors:
                for r in all_reels:
                    if solver.Value(X[g][c][r]) == 1:
                        solution_row = {'Reel': r, 'Color': c, 'Group': g, 'Cost': cost[c][r]}
                        solution_list.append(solution_row)
    else:
        solution_row = {'Reel': -1, 'Color': -1, 'Group': -1, 'Cost': -1}
        solution_list.append(solution_row)

    print(solver.StatusName())
    print('Statistics')
    print("Min Length: ", [solver.Value(m_min_length[i])/groups[i] for i in range(len(groups))])
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())

    return solution_list, solver.StatusName()


if __name__ == '__main__':

    '''
    cost =  [[90, 76, 75, 70, 50, 74, 12, 68, 70],
             [35, 85, 55, 65, 48, 101, 70, 83, 0],
             [125, 95, 90, 105, 59, 120, 36, 73, 59],
             [45, 110, 95, 115, 104, 83, 37, 71, 0],
             [60, 105, 80, 75, 59, 62, 93, 88, 59],
             [45, 65, 110, 95, 47, 31, 81, 34, 0],
             [38, 51, 107, 41, 69, 99, 115, 48, 51],
             [47, 85, 57, 71, 92, 77, 109, 36, 0],
             [39, 63, 97, 49, 118, 56, 92, 61, 56],
             [47, 101, 71, 60, 88, 109, 52, 90, 52]]

    groups = [10, 10, 10, 10, 10, 10, 10, 10, 6]
    '''
    '''
    cost =  [[90, 89, 76, 76, 76, 75, 75, 74, 73, 71, 70, 70, 68, 50, 12],
             [101, 90, 85, 83, 76, 74, 73, 70, 70, 65, 55, 48, 35, 35, 0],
             [125, 120, 105, 95, 90, 80, 73, 59, 59, 55, 43, 36, 0, 0, 0],
             [115, 110, 104, 95, 90, 83, 81, 73, 71, 70, 56, 45, 37, 32, 0],
             [105, 99, 93, 88, 88, 80, 80, 75, 67, 62, 60, 59, 59, 43, 0],
             [110, 107, 105, 95, 91, 81, 81, 81, 65, 47, 45, 34, 31, 0, 0],
             [115, 107, 99, 90, 89, 88, 69, 63, 51, 51, 48, 45, 41, 38, 0],
             [115, 109, 92, 85, 77, 73, 72, 71, 57, 47, 36, 0, 0, 0, 0],
             [118, 111, 105, 97, 92, 65, 63, 61, 56, 56, 49, 39, 0, 0, 0],
             [109, 101, 90, 88, 71, 67, 60, 54, 52, 52, 47, 33, 0, 0, 0],
             [115, 107, 100, 99, 85, 73, 62, 51, 49, 0, 0, 0, 0, 0, 0],
             [105, 99, 99, 99, 88, 85, 83, 72, 69, 67, 55, 46, 0, 0, 0]]
    
    groups = [12, 12, 12, 12, 12, 12, 12, 12, 9, 9, 9, 8, 7, 7, 7]
    '''

    cost = [[ 90,  89,  76, 76, 76, 75, 75, 74, 73, 71, 70, 70, 68, 50, 12],
            [101,  90,  85, 83, 76, 74, 73, 70,  0, 70, 65, 55, 48, 35, 35],
            [125, 120, 105, 95, 90, 80, 73, 59, 59,  0,  0,  0, 55, 43, 36],
            [115, 110, 104, 95, 90, 83, 81, 73,  0, 71, 70, 56, 45, 37, 32],
            [105,  99,  93, 88, 88, 80, 80, 75, 67,  0, 62, 60, 59, 59, 43],
            [110, 107, 105, 95, 91, 81, 81, 81,  0, 65,  0, 47, 45, 34, 31],
            [115, 107,  99, 90, 89, 88, 69, 63, 51, 51, 48,  0, 45, 41, 38],
            [115, 109,  92, 85, 77, 73, 72, 71,  0, 57, 47, 36,  0,  0,  0],
            [118, 111, 105, 97, 92, 65, 63, 61, 56, 56, 49, 39,  0,  0,  0],
            [109, 101,  90, 88, 71, 67, 60, 54, 52, 52, 47, 33,  0,  0,  0],
            [115, 107, 100, 99, 85, 73, 62, 51, 49,  0,  0,  0,  0,  0,  0],
            [105,  99,  99, 99, 88, 85, 83, 72, 69, 67, 55, 46,  0,  0,  0]]

    groups = [12,  12,  12,  12,  12,  12,  12, 12,   8,   9,   9,   9,  7,  7,   7]

    '''
    cost = [
        [90, 80, 75, 70, 50],
        [35, 85, 55, 65, 48],
        [125, 95, 90, 105, 59],
        [45, 110, 95, 115, 104],
        [60, 105, 80, 75, 59],
        [45, 65, 110, 95, 47]
    ]
    groups = [6, 6, 6, 6, 6]
    '''
    solution, sol = solve_assignment(cost, groups, early_stopping=3000)
    d = pd.DataFrame(solution)

    # Calculate the remnant associated with each reel (length - min of group)
    d['Remnant'] = d['Cost'] - d.groupby('Group')['Cost'].transform('min')
    print(sum(d['Remnant']))
#%%
