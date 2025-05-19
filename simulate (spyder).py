"""

simulate.py
-----------
This code simulates the model.

"""

#%% Imports from Python
from numpy import cumsum,empty,linspace,nan,squeeze,where,zeros
from numpy.random import choice,rand,seed
from numpy.linalg import matrix_power
from types import SimpleNamespace
from numpy import mean 

#%% Simulate the model.
def earn_profit(myClass):
    '''
    
    This function simulates the dynamic model of firm investment.
    
    Input:
        myClass : Model class with parameters, grids, cost and revenue functions, and policy functions.
        
    '''

    print('\n--------------------------------------------------------------------------------------------------')
    print('Simulate the Model')
    print('--------------------------------------------------------------------------------------------------\n')
    
    # Namespace for simulation.
    setattr(myClass,'sim',SimpleNamespace())
    sim = myClass.sim

    # Model parameters, grids and functions.
    
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.

    par.seed_sim # Seed for simulation.
    plen = par.plen
    klen = par.klen # Capital grid size.
    Alen = par.Alen # Productivity grid size.
    kgrid = par.kgrid # Capital today (state).
    Agrid = par.Agrid[0] # Productivity today (state).
    pgrid = par.pgrid[0]
    ppmat = par.ppmat
    pmat = par.pmat # Productivity today (state).

    vpol = sol.v # Firm value.
    kpol = sol.k # Policy function for capital.
    ipol = sol.i # Policy function for investment.
    rpol = sol.r # Optimal revenue.
    epol = sol.e # Optimal total investment expenditure.
    ppol = sol.p # Optimal profit.

    T = par.T # Time periods.
    Asim = zeros((T*2,1)) # Container for simulated productivity.
    vsim = zeros((T*2,1)) # Container for simulated firm value.
    rsim = zeros((T*2,1)) # Container for simulated output.
    ksim = zeros((T*2,1)) # Container for simulated capital stock.
    isim = zeros((T*2,1)) # Container for simulated investment.
    esim = zeros((T*2,1)) # Container for simulated investment expenditure.
    psim = zeros((T*2,1)) # Container for simulated profit.
    prsim = zeros((T*2,1)) # Container for simulated profit.
    # Begin simulation.
    
    seed(par.seed_sim)

    pmat0 = matrix_power(pmat,1000)
    pmat0 = pmat0[0,:] # % Stationary distribution.
    cmat = cumsum(par.pmat,axis=1) # CDF matrix.
    
    ppmat0 = matrix_power(ppmat,1000)
    ppmat0 = ppmat0[0,:] # % Stationary distribution.
    pcmat = cumsum(par.ppmat,axis=1) # CDF matrix.

    A0_ind = choice(linspace(0,Alen,Alen,endpoint=False,dtype=int),1,p=pmat0) # Index for initial productivity.
    k0_ind = choice(linspace(0,klen,klen,endpoint=False,dtype=int),1) # Index for initial capital stock.
    p0_ind = choice(linspace(0,plen,plen,endpoint=False,dtype=int),1,p=ppmat0) # Index for initial price.
    Asim[0] = Agrid[A0_ind] # Productivity in period 1.
    vsim[0] = vpol[k0_ind,A0_ind,p0_ind] # Firm value in period 1 given k0 and A0.
    ksim[0] = kpol[k0_ind,A0_ind,p0_ind] # Capital choice for period 2 given k0 and A0.
    isim[0] = ipol[k0_ind,A0_ind,p0_ind] # Investment in period 1 given k0 and A0.
    rsim[0] = rpol[k0_ind,A0_ind,p0_ind] # Revenue in period 1 given k0 and A0.
    esim[0] = epol[k0_ind,A0_ind,p0_ind] # Investment ependiture in period 1 given k0 and A0.
    psim[0] = ppol[k0_ind,A0_ind,p0_ind] # Profit in period 1 given k0 and A0.
    prsim[0] = pgrid[p0_ind] # Profit in period 1 given k0 and A0.
    
    A1_ind = where(rand(1)<=squeeze(cmat[A0_ind,:])) # Draw productivity for next period.
    At_ind = A1_ind[0][0]
    p1_ind = where(rand(1)<=squeeze(pcmat[p0_ind,:])) # Draw productivity for next period.
    pt_ind = p1_ind[0][0]
    # Simulate endogenous variables.
    
    for j in range(1,T*2): # Time loop.
        kt_ind = where(ksim[j-1]==kgrid); # Capital choice in the previous period is the state today. Find where the latter is on the grid.
        Asim[j] = Agrid[At_ind] # Productivity in period t.
        prsim[j] = pgrid[pt_ind] # Productivity in period t.
        vsim[j] = vpol[kt_ind,At_ind,pt_ind] # Firm value in period t.
        ksim[j] = kpol[kt_ind,At_ind,pt_ind] # Capital stock for period t+1.
        isim[j] = ipol[kt_ind,At_ind,pt_ind] # Investment in period t.
        rsim[j] = rpol[kt_ind,At_ind,pt_ind] # Revenue in period t.
        esim[j] = epol[kt_ind,At_ind,pt_ind] # Investment expenditure in period t.
        psim[j] = ppol[kt_ind,At_ind,pt_ind] # Profit in period t.
        A1_ind = where(rand(1)<=squeeze(cmat[At_ind,:])) # Draw next state.
        At_ind = A1_ind[0][0] # State next period.
        p1_ind = where(rand(1)<=squeeze(pcmat[pt_ind,:])) # Draw next state.
        pt_ind = p1_ind[0][0] # State next period.
    
    # Simulated model.
    sim.Asim = Asim[T:2*T] # Simulated productivity.
    sim.vsim = vsim[T:2*T] # Simulated firm value.
    sim.ksim = ksim[T:2*T] # Simulated capital choice.
    sim.isim = isim[T:2*T] # Simulated investment.
    sim.rsim = rsim[T:2*T] # Simulated revenue.
    sim.esim = esim[T:2*T] # Simulated investment expenditure.
    sim.psim = psim[T:2*T] # Simulated profit.
    sim.prsim = prsim[T:2*T] # Simulated profit.
    
    sim.Kbar = mean(sim.ksim)
    sim.Ibar = mean(sim.isim)
    print(sim.Kbar)
    print(sim.Ibar)
    print('Simulation done.\n')
    print('--------------------------------------------------------------------------------------------------\n')