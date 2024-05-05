# Import packages
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import sys
from functools import partial

# Sets P and D, respectively
# When we code sets we can be more descriptive in the name
production = ['Baltimore','Cleveland','Little Rock','Birmingham','Charleston']
distribution = ['Columbia','Indianapolis','Lexington','Nashville','Richmond','St. Louis']

class CallbackData:
    def __init__(self, modelvars):
        self.modelvars = modelvars
        self.lastiter = -GRB.INFINITY
        self.lastnode = -GRB.INFINITY


def mycallback(model, where, *, cbdata, logfile):
    """
    Callback function. 'model' and 'where' arguments are passed by gurobipy
    when the callback is invoked. The other arguments must be provided via
    functools.partial:
      1) 'cbdata' is an instance of CallbackData, which holds the model
         variables and tracks state information across calls to the callback.
      2) 'logfile' is a writeable file handle.
    """

    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print(f"{cdels} columns and {rdels} rows are removed")
    elif where == GRB.Callback.SIMPLEX:
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT)
        if itcnt - cbdata.lastiter >= 100:
            cbdata.lastiter = itcnt
            obj = model.cbGet(GRB.Callback.SPX_OBJVAL)
            ispert = model.cbGet(GRB.Callback.SPX_ISPERT)
            pinf = model.cbGet(GRB.Callback.SPX_PRIMINF)
            dinf = model.cbGet(GRB.Callback.SPX_DUALINF)
            if ispert == 0:
                ch = " "
            elif ispert == 1:
                ch = "S"
            else:
                ch = "P"
            print(f"{int(itcnt)} {obj:g}{ch} {pinf:g} {dinf:g}")
    elif where == GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if nodecnt - cbdata.lastnode >= 100:
            cbdata.lastnode = nodecnt
            actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            print(
                f"{nodecnt:.0f} {actnodes:.0f} {itcnt:.0f} {objbst:g} "
                f"{objbnd:g} {solcnt} {cutcnt}"
            )
        if abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst)):
            print("Stop early - 10% gap achieved")
            model.terminate()
        if nodecnt >= 10000 and solcnt:
            print("Stop early - 10000 nodes explored")
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(cbdata.modelvars)
        print(
            f"**** New solution at node {nodecnt:.0f}, obj {obj:g}, "
            f"sol {solcnt:.0f}, x[0] = {x[0]:g} ****"
        )
    elif where == GRB.Callback.MIPNODE:
        # MIP node callback
        print("**** New node ****")
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(cbdata.modelvars)
            model.cbSetSolution(cbdata.modelvars, x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)
        print(f"{itcnt:.0f} {primobj:g} {dualobj:g} {priminf:g} {dualinf:g} {cmpl:g}")
    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        logfile.write(msg)


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

    # _____________________
    # Create Model Object
    # ---------------------
    model = gp.Model("assignment")

    # Set cuts to be aggressive
    #model.setParam(GRB.Param.Heuristics, 0)
    model.setParam(GRB.Param.Cuts, 2)
    model.setParam(GRB.Param.MIPFocus, 2)
    model.setParam(GRB.Param.PreDepRow, 0)
    model.setParam(GRB.Param.SimplexPricing, 1)
    #model.setParam(GRB.Param., 0)

    # -----------------------
    # Create Model Variables
    # -----------------------
    # Minimum non-zero value for each group
    y = model.addVars(num_groups, lb=0, ub=maxL, vtype=GRB.INTEGER, name='min')

    # Indicator variables
    x = model.addVars(num_groups, num_colors, num_reels, vtype=GRB.BINARY, name='x')

    # ----------
    # Objective
    # ----------
    # Objective cost - Minimize the differences between reel lengths in each group (same as min(sum_g-m*min))
    # Define objective function
    obj = gp.quicksum(y[g] * groups[g] for g in all_groups)

    # Could also use the following - different objective value, but same end result (doesn't appear to effect speed)
    model.setObjective(obj, GRB.MAXIMIZE)

    # -------------------------
    # Create Model Constraints
    # -------------------------
    # Each group is assigned at most one reel from each color
    for g in all_groups:
        for c in all_colors:
            model.addConstr(x.sum(g, c, '*') <= 1)

    # Each reel is assigned to exactly one group (0 length reels don't get assigned)
    for c in all_colors:
        for r in all_reels:
            if cost[c][r] != 0:
                model.addConstr(x.sum('*', c, r) == 1)

    # Must have the correct number of reels in each group
    for g in all_groups:
        model.addConstr(x.sum(g, '*', '*') == groups[g])

    # Determine the smallest non-zero length in the group - implication enforces only looking at non-zero reels
    for g in all_groups:
        for c in all_colors:
            for r in all_reels:
                model.addGenConstrIndicator(x[g, c, r], True, y[g] <= cost[c][r])

    # --------------
    # Break symmetry
    # --------------
    # TODO - Find ways to break symmetry
    # Can swap if abs(y[g1][c1][r1] - y[g2][c2][r2]) <= y[g2][c2][r2]-min_g[2] and y[g1][c1][r1]-min_g[1]

    # ---------
    # Add Hint
    # ---------

    # Setup Early Stopping when solutions don't improve within time limit (s)

    # -------------------------------------
    # Create a solver and solve the model
    # -------------------------------------
    '''
    model.tune()

    for i in range(model.tuneResultCount):
        model.getTuneResult(i)
        model.write('tune'+str(i)+'.prm')
    '''

    model.optimize()

    solution_list = []

    print(model.Status)
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        for g in all_groups:
            for c in all_colors:
                for r in all_reels:
                    if x[g, c, r].x == 1:
                        solution_row = {'Reel': r, 'Color': c, 'Group': g, 'Cost': cost[c][r]}
                        solution_list.append(solution_row)

    print('Statistics')
    print("Min Length: ", [y[i].x for i in range(len(groups))])
    print('  - wall time : %f s' % model.Runtime)

    return solution_list, model.Status


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
    solution, status = solve_assignment(cost, groups, early_stopping=300000)
    d = pd.DataFrame(solution)

    # Calculate the remnant associated with each reel (length - min of group)
    d['Remnant'] = d['Cost'] - d.groupby('Group')['Cost'].transform('min')
    print(sum(d['Remnant']))

#%%
