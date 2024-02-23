from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
import pandas as pd
from threading import Timer, Lock

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
    mins = [min(map(int, i)) for i in cost]
    minL = min(i for i in mins if i != 0)
    num_colors = len(cost)
    num_reels = len(cost[1])
    num_groups = len(groups)
    all_colors = range(num_colors)
    all_reels = range(num_reels)
    all_groups = range(num_groups)

    # ---------------------------------------------
    # Create the mip solver with the SCIP backend.
    # ---------------------------------------------
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # -----------------------
    # Create Model Variables
    # -----------------------
    # Helper variables to find minimum non-zero value
    y = [solver.IntVar(0, maxL, f'yi_{g}') for g in all_groups]
    d = [[[solver.BoolVar(f'd_{g}{c}{r}') for r in all_reels] for c in all_colors] for g in all_groups]

    # min_length[g] * M (length of group[g] (used for objective)
    m_min_length = [solver.IntVar(0, maxL*num_groups, f'max_length_{g}') for g in all_groups]
    sum_g = [solver.IntVar(0, maxL*num_colors, f'sum_{g}') for g in all_groups]

    # Indicator Variables for color/reel in a group x[group, color, reel]
    X = [[[solver.BoolVar(f'x[{g},{c},{r}]') for r in all_reels] for c in all_colors] for g in all_groups]

    # Each group is assigned at most one reel from each color (0 length reels don't need to be assigned).
    for g in all_groups:
        for c in all_colors:
            solver.Add(solver.Sum(X[g][c][r] for r in all_reels) <= 1)

    # Each reel is assigned to exactly one group (0 length reels don't get assigned)
    for c in all_colors:
        for r in all_reels:
            if cost[c][r] != 0:
                solver.Add(solver.Sum(X[g][c][r] for g in all_groups) == 1)

    # Must have the correct number of reels in each group
    for g in all_groups:
        solver.Add(solver.Sum(X[g][c][r] for r in all_reels for c in all_colors) == groups[g])

    # Determine the smallest non-zero length in the group - implication enforces only looking at non-zero reels
    for g in all_groups:
        for c in all_colors:
            for r in all_reels:
                solver.Add(y[g] >= cost[c][r]*X[g][c][r] - (maxL-minL)*(1-d[g][c][r]))
                solver.Add(y[g] <= cost[c][r]*X[g][c][r] + maxL*(1-X[g][c][r]))
                solver.Add(d[g][c][r] <= cost[c][r]*X[g][c][r])
        solver.Add(solver.Sum(d[g][c][r] for c in all_colors for r in all_reels) == 1)

    for g in all_groups:
        solver.Add(sum_g[g] == solver.Sum(cost[c][r]*X[g][c][r] for c in all_colors for r in all_reels))
        solver.Add(m_min_length[g] == y[g]*groups[g])

    for g1 in range(1, num_groups-1):
        for g2 in range(g1+1, num_groups):
            if groups[g1] == groups[g1-1]:
                solver.Add(y[g] >= y[g-1])

    # ----------
    # Objective
    # ----------
    # Objective cost - Minimize the differences between reel lengths in each group (same as min(sum_g-m*min))
    # solver.Add(objective == sum(sum_g[g] - m_min_length[g] for g in all_groups))
    # solver.Minimize(solver.Sum(y[g] for g in all_groups))
    solver.Maximize(solver.Sum(m_min_length[g] for g in all_groups))

    # Sets a time limit of 5 minutes.
    solver.SetTimeLimit(1200000)
    solver.EnableOutput()
    #solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    # Solve the Model
    status = solver.Solve()

    solver.Maximize(solver.Sum(m_min_length[g] for g in all_groups))
    #status = solver.SolveWithSolutionCallback(model, solution_printer)
    #solution_printer.clean()

    solution_list = []
    if status == solver.OPTIMAL or status == solver.FEASIBLE:
        for g in all_groups:
            for c in all_colors:
                for r in all_reels:
                    if X[g][c][r].solution_value() == 1:
                        print(f"Min {g},{c},{r} = {d[g][c][r].solution_value()*cost[c][r]}")
                        solution_row = {'Reel': r, 'Color': c, 'Group': g, 'Cost': cost[c][r]}
                        solution_list.append(solution_row)

    print('Statistics')
    print("Min Length: ", [y[i].solution_value() for i in range(len(groups))])
    #print('  - conflicts : %i' % solver.NumConflicts())
    #print('  - branches  : %i' % solver.NumBranches())
    print('  - wall time : %f s' % solver.WallTime())

    return solution_list, status


if __name__ == '__main__':
    '''
    cost =  [[90, 76, 75, 70, 50, 74, 12, 68, 70, 71, 75, 76, 89, 76, 73],
             [35, 85, 55, 65, 48, 101, 70, 83, 0, 70, 76, 73, 90, 74, 35],
             [125, 95, 90, 105, 59, 120, 36, 73, 59, 0, 0, 0, 80, 55, 43],
             [45, 110, 95, 115, 104, 83, 37, 71, 0, 73, 81, 90, 70, 56, 32],
             [60, 105, 80, 75, 59, 62, 93, 88, 59, 0, 88, 99, 80, 67, 43],
             [45, 65, 110, 95, 47, 31, 81, 34, 0, 81, 0, 105, 81, 91, 107],
             [38, 51, 107, 41, 69, 99, 115, 48, 51, 88, 90, 0, 63, 45, 89],
             [47, 85, 57, 71, 92, 77, 109, 36, 0, 115, 72, 73, 0, 0, 0],
             [39, 63, 97, 49, 118, 56, 92, 61, 56, 111, 105, 65, 0, 0, 0],
             [47, 101, 71, 60, 88, 109, 52, 90, 52, 67, 54, 33, 0, 0, 0],
             [51, 100, 73, 62, 85, 107, 49, 99, 115, 0, 0, 0, 0, 0, 0],
             [55, 99, 69, 67, 83, 105, 99, 99, 85, 88, 46, 72, 0, 0, 0]]
    
    groups = [12, 12, 12, 12, 12, 12, 12, 12, 8, 9, 9, 9, 7, 7, 7]
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

    solution, sol = solve_assignment(cost, groups, early_stopping=300000)
    print(solution)
    d = pd.DataFrame(solution)

    # Calculate the remnant associated with each reel (length - min of group)
    d['Remnant'] = d['Cost'] - d.groupby('Group')['Cost'].transform('min')
    print(sum(d['Remnant']))

#%%
