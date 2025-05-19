"""

solve.py
--------
This code solves the model.

"""

#%% Imports from Python
from numpy import argmax,squeeze,zeros,seterr
from numpy.linalg import norm
from types import SimpleNamespace
import time
seterr(divide='ignore')
seterr(invalid='ignore')

#%% Solve the model using Value Function Iteration.
def do_business(myClass):
    '''
    
    This function solves the dynamic model of firm investment.
    
    Input:
        myClass : Model class with parameters, grids, cost function, and profit function.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Solving the Model by Value Function Iteration')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for optimal policy funtions.
    setattr(myClass,'sol',SimpleNamespace())
    sol = myClass.sol

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    
    beta = par.beta # Discount factor.
    alpha = par.alpha # Capital's share of income.
    delta = par.delta # Depreciation rate
    gamma = par.gamma # Speed of adjustment; cost function coefficient.

    klen = par.klen # Grid size for k.
    kgrid = par.kgrid # Grid for k (state and choice).

    Alen = par.Alen # Grid size for A.
    Agrid = par.Agrid[0] # Grid for A.
    pmat = par.pmat # Grid for A.

    plen = par.plen # Grid size for A.
    pgrid = par.pgrid[0] # Grid for A.
    ppmat = par.ppmat # Grid for A.

    production = par.production # Revenue function.
    total_cost = par.total_cost # Cost function.

    # Value Function Iteration.
    v0 = zeros((klen,Alen,plen)) # Guess of value function for each value of k.
                    
    crit = 1e-6;
    maxiter = 10000;
    diff = 1;
    iter = 0;

    t0 = time.time()

    while (diff > crit) and (iter < maxiter): # Iterate on the Bellman Equation until convergence.
    
        v1 = zeros((klen,Alen,plen)) # Container for V.
        k1 = zeros((klen,Alen,plen)) # Container for k'.
        i1 = zeros((klen,Alen,plen)) # Container for i.
        r1 = zeros((klen,Alen,plen)) # Container for revenue.
        e1 = zeros((klen,Alen,plen)) # Container for investment expenditure.
        p1 = zeros((klen,Alen,plen)) # Container for profit.

        for q in range(0,klen): # Loop over the k-states.
            for j in range(0,Alen): # Loop over the A-states.
                for h in range(0,plen): # Loop over the p-states.

                    # Macro variables.
                    rev = production(Agrid[j],kgrid[q],alpha) # Revenue given A and K.
                    expend = total_cost(kgrid[q],pgrid[h],kgrid,delta,gamma) # Total investment expenditure given K.
                    prof = rev-expend # Profit.
                    invest = kgrid-(1.0-delta)*kgrid[q] # Investment in new capital.
    
                    # Solve the maximization problem.
                    ev = 0.0 
                    
                    for hh in range(0,plen):
                        ev += squeeze(squeeze(v0[:,:,hh])@pmat[j,:].T)*ppmat[h,hh] #  The next-period value function is the expected value function over each possible next-period A, conditional on the current state j.
                    
                    vall = prof + beta*ev # Compute the value function for each choice of k', given k.
                    v1[q,j,h] = max(vall) # Maximize: vmax is the maximized value function; ind is where it is in the grid.
                    k1[q,j,h] = kgrid[argmax(vall)] # Optimal k'.
                    i1[q,j,h] = invest[argmax(vall)] # Optimal i.
                    r1[q,j,h] = rev # Total revenue.
                    e1[q,j,h] = expend[argmax(vall)] # Total cost.
                    p1[q,j,h] = prof[argmax(vall)] # Profits.
        
        diff = norm(v1-v0) # Check convergence.
        v0 = v1; # Update guess.

        iter = iter + 1; # Update counter.
        
        # Print counter.
        if iter%25 == 0:
            print('Iteration: ',iter,'.\n')

    t1 = time.time()
    print('Elapsed time is ',t1-t0,' seconds.')
    print('Converged in ',iter,' iterations.')

    # Macro variables, value, and policy functions.
    sol.v = v1 # Firm value.
    sol.k = k1 # Capital policy function.
    sol.i = i1 # Investment policy function.
    sol.r = r1 # Revenue function.
    sol.e = e1 # Investment expenditure function.
    sol.p = p1 # Profit function.