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
    Solve assignment problem of reducing cost of assigning boxwe to groups
    Arguments:  cost: cost matrix of assigning item
                groups: list of groups

    Returns:    solution_list: list of dictionaries [Box, Color, Group, Cost] minimizing the costs,
                status:  status of solver - string (OPTIMAL, FEASIBLE, INFEASIBLE)
    """
    maxL = max([max(map(int, i)) for i in cost]) + 1
    num_colors = len(cost)
    num_boxes = len(cost[1])
    num_groups = len(groups)
    all_colors = range(num_colors)
    all_boxes = range(num_boxes)
    all_groups = range(num_groups)

    # _____________________
    # Create Model Object
    # ---------------------
    model = cp_model.CpModel()

    # -----------------------
    # Create Model Variables
    # -----------------------
    # objective is our objective variable
    objective = model.NewIntVar(0, 1000000, 'objective')

    # Helper variables to find minimum non-zero value
    y = [model.NewIntVar(0, 200000, f'yi_{g}') for g in all_groups]

    # min_length[g] * M (length of group[g] (used for objective)
    mmin_length = [model.NewIntVar(0, 200000, f'max_length_{g}') for g in all_groups]
    sum_g = [model.NewIntVar(0, 2000000, f'sum_{g}') for g in all_groups]

    # Indicator Variables for color/reel in a group x[group, color, reel]
    X = [[[model.NewBoolVar(f'x[{g},{c},{b}]') for b in all_boxes] for c in all_colors] for g in all_groups]

    # -------------------------
    # Create Model Constraints
    # -------------------------
    # Each group is assigned at most one box from each color (0 length reels don't need to be assigned).
    for g in all_groups:
        for c in all_colors:
            model.Add(sum(X[g][c][b] for b in all_boxes) <= 1)

    # Each box from every color is assigned to at most one group
    for b in all_boxes:
        for c in all_colors:
            model.Add(sum(X[g][c][b] for g in all_groups) == 1)

    # Must have the correct number of boxes in each group
    for g in all_groups:
        model.Add(sum(X[g][c][b] for b in all_boxes for c in all_colors) == groups[g])

    # Don't use boxes that have 0 items
    for g in all_groups:
        for b in all_boxes:
            for c in all_colors:
                if cost[c][b] == 0:
                    model.Add(X[g][c][b] == 0)
                    print(f"Not using {g},{c},{b}")

    # Determine smallest non-zero item count in group - Because some X[g][c][r]'s will be zero
    for g in all_groups:
        for c in all_colors:
            for b in all_boxes:
                # smallest[g][c][r] == True if X[g][c][r] == 1
                #model.Add(X[g][c][b] > 0).OnlyEnforceIf(smallest[g][c][b])

                # smallest[g][c][r] == False if X[g][c][r] == 0
                #model.Add(X[g][c][b] == 0).OnlyEnforceIf(smallest[g][c][b].Not())

                # yi[g][c][r] == cost[c][r] if smallest[g][c][r] == True
                model.Add(y[g] <= cost[c][b]).OnlyEnforceIf(X[g][c][b])

                # Set yi[g][c][r] to a large enough value that it can't be min if smallest[g][c][r] == False
                #model.Add(yi[g][c][b] == maxL).OnlyEnforceIf(smallest[g][c][b].Not())

        # min_length[g] = min(yi[g][c][r] for c in all_colors for r in all_reels)
        #model.AddMinEquality(min_length[g], [yi[g][c][b] for c in all_colors for b in all_boxes])

    # Determine M*min_length in a group (to use in cost function)
    # sum(C[g][c][b] - min_length[g]) is same as sum(sum_g[g] - group_length[g]*min_length[g])
    for g in all_groups:
        model.Add(sum_g[g] == sum(cost[c][b]*X[g][c][b] for c in all_colors for b in all_boxes))
        model.AddMultiplicationEquality(mmin_length[g], y[g], groups[g])

    # --------------
    # Break symmetry
    # --------------
    # TODO - Find ways to break symmetry
    # Can swap if y[g1][c1][r1] - y[g2][c2][r2] <= y[g2][c2][r2]-min_g[2] and y[g1][c1][r1]-min_g[1]

    # Objective cost - Minimize the differences between box item counts in each group (same as min(sum_g-m*min))
    model.Add(objective == sum(sum_g[g] - mmin_length[g] for g in all_groups))
    model.Minimize(objective)

    # ---------
    # Add Hint
    # ---------


    # Setup Early Stopping when solutions don't improve within time limit (s)
    solution_printer = EarlyStoppingLogger([objective], early_stopping)

    # -------------------------------------
    # Create a solver and solve the model
    # -------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8

    # Sets a time limit of approx 48 hours.
    solver.parameters.max_time_in_seconds = 200000.0
    #solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    # Solve the Model
    #status = solver.Solve(model)
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    solution_printer.clean()

    solution_list = []
    print(solver.StatusName(status))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for g in all_groups:
            for c in all_colors:
                for b in all_boxes:
                    if solver.Value(X[g][c][b]) == 1:
                        solution_row = {'Box': b, 'Color': c, 'Group': g, 'Cost': cost[c][b]}
                        solution_list.append(solution_row)
    print('Statistics')
    print("Min Length: ", [solver.Value(y[g]) for g in range(len(groups))])
    print('  - conflicts : %i' % solver.NumConflicts())
    print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())

    return solution_list, solver.StatusName()


if __name__ == '__main__':

    cost = [[90, 76, 75, 74, 68, 61, 50, 12, 70],
            [101, 85, 83, 70, 65, 55, 48, 35, 0],
            [125, 120, 105, 95, 90, 73, 73, 36, 59],
            [115, 110, 104, 95, 83, 71, 45, 37, 0],
            [105, 93, 88, 81, 80, 75, 62, 60, 59],
            [110, 95, 81, 65, 47, 45, 34, 31, 0],
            [115, 107, 99, 69, 48, 47, 41, 38, 51],
            [109, 92, 85, 77, 71, 57, 47, 36, 0],
            [118, 97, 92, 63, 61, 60, 49, 39, 56],
            [109, 101, 90, 88, 88, 71, 60, 47, 52]]

    groups = [10, 10, 10, 10, 10, 10, 10, 10, 10]
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
    cost = [
        [70, 75, 80, 90],
        [35, 55, 65, 85],
        [90, 95, 95, 125],
        [45, 95, 110, 115],
        [50, 90, 100, 100],
    ]

    groups = [5, 5, 5, 5]
    '''
    solution, sol = solve_assignment(cost, groups, early_stopping=180000)
    d = pd.DataFrame(solution)

    # Calculate the remnant associated with each reel (length - min of group)
    d['Remnant'] = d['Cost'] - d.groupby('Group')['Cost'].transform('min')
    print(sum(d['Remnant']))


#%%
