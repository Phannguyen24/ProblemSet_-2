"""

model.py
--------
This code sets up the model.

"""

#%% Imports from Python
from numpy import count_nonzero,exp,expand_dims,linspace,tile
from scipy import stats
from types import SimpleNamespace

#%% Firm Investment Model.
class firm():
    '''
    
    Methods:
        __init__(self,**kwargs) -> Set the firm's attributes.
        setup(self,**kwargs) -> Sets parameters.
        
    '''
    
    #%% Constructor.
    def __init__(self,**kwargs):
        '''        
        
        This initializes the model.
        
        Optional kwargs:
            All parameters changed by setting kwarg.
            
        '''

        print('--------------------------------------------------------------------------------------------------')
        print('Model')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The model is a dynamic model of firm investment and is solved via Value Function Iteration.')
        
        print('\n--------------------------------------------------------------------------------------------------')
        print('Household')
        print('--------------------------------------------------------------------------------------------------\n')
        print('   The firm is infintely-lived.')
        print('   It chooses investment to maximize profit.')
        
    #%% Set up model.
    def setup(self,**kwargs):
        '''
        
        This sets the parameters and creates the grids for the model.
        
            Input:
                self : Model class.
                kwargs : Values for parameters if not using the default.
                
        '''
        
        # Namespace for parameters, grids, and utility function.
        setattr(self,'par',SimpleNamespace())
        par = self.par

        print('\n--------------------------------------------------------------------------------')
        print('Parameters:')
        print('--------------------------------------------------------------------------------\n')
        
        # Technology.
        par.beta = 0.96 # Discount factor.
        par.alpha = 0.33 # Capital's share of income.
        par.delta = 0.25 # Depreciation rate.

        # Prices and Income.
        par.gamma = 0.08 # Speed of adjustment; cost function coefficient.

        par.sigma_eps = 0.07 # Std. dev of productivity shocks and prices.
        par.rho = 0.85 # Persistence of AR(1) process and prices.
        par.mu = 0.0 # Intercept of AR(1) processes.

        par.ylen = 7 # Grid size for y.
        par.m = 3 # Scaling parameter for Tauchen.
            
        # Simulation parameters.
        par.seed_sim = 2025 # Seed for simulation.
        par.T = 1000 # Number of time periods.

        # Set up asset grid.
        par.klen = 300 # Grid size for k.
        par.kmax = 30.0 # Upper bound for k.
        par.kmin = 0.0001 # Minimum k.
          
        # Discretized productivity.
        par.Alen = 7 # Grid size for A.
        par.m = 3 # Scaling parameter for Tauchen.
        
        # Discretized prices.
              
        par.plen = 7 # Grid size for p.
        
        # Update parameter values to kwarg values if you don't want the default values.
        for key,val in kwargs.items():
            setattr(par,key,val)
        
        assert par.beta > 0.0 and par.beta < 1.0
        assert par.alpha > 0.0 and par.alpha < 1.0
        assert par.delta >= 0.0 and par.delta <= 1.0
        assert par.sigma_eps > 0.00
        assert abs(par.sigma_eps) < 1.00
        assert par.gamma >= 0.0
        assert par.klen > 5
        assert par.kmax > par.kmin
        
        # Set up asset grid.
        par.kgrid = linspace(par.kmin,par.kmax,par.klen) # Equally spaced, linear grid for k (and k').

        # Discretize productivity.
        Agrid,pmat = tauchen(par.mu,par.rho,par.sigma_eps,par.Alen,par.m) # Tauchen's Method to discretize the AR(1) process for log productivity.
        par.Agrid = exp(Agrid) # The AR(1) is in logs so exponentiate it to get A.
        par.pmat = pmat # Transition matrix.
    
        # Discretize prices.
        pgrid,ppmat = tauchen(par.mu,par.rho,par.sigma_eps,par.plen,par.m) # Tauchen's Method to discretize the AR(1) process for log productivity.
        par.pgrid = exp(pgrid) # The AR(1) is in logs so exponentiate it to get A.
        par.ppmat = ppmat # Transition matrix.
    
        # Revenue and cost functions.
        par.production = production
        par.total_cost = total_cost
        
        print('beta: ',par.beta)
        print('alpha: ',par.alpha)
        print('delta: ',par.delta)
        print('kmin: ',par.kmin)
        print('kmax: ',par.kmax)
        print('gamma: ',par.gamma)

#%% Revenue Function.
def production(A,k,alpha):
    #Revenue function.
    
    output = A*k**alpha # Cobb-Douglas production.
    
    return output

#%% Cost function.
        
def total_cost(k,p,kgrid,delta,gamma):
    # Convex adjustment cost.
    
    invest = kgrid-(1.0-delta)*k
    adj_cost = (gamma/2.0)*((invest/k)**2.0)*k # Convex adjustment cost.
    cost = adj_cost + p*invest # Total investment cost.
    
    return cost

             
#%% Tauchen's Method.
def tauchen(mu,rho,sigma,N,m):
    """
    
    This function discretizes an AR(1) process.
    
            y(t) = mu + rho*y(t-1) + eps(t), eps(t) ~ NID(0,sigma^2)
    
    Input:
        mu    : Intercept of AR(1).
        rho   : Persistence of AR(1).
        sigma : Standard deviation of error term.
        N     : Number of states.
        m     : Parameter such that m time the unconditional std. dev. of the AR(1) is equal to the largest grid point.
        
    Output:
        y    : Grid for the AR(1) process.
        pmat : Transition probability matrix.
        
    """
    
    #%% Construct equally spaced grid.
    
    ar_mean = mu/(1.0-rho) # The mean of a stationary AR(1) process is mu/(1-rho).
    ar_sd = sigma/((1.0-rho**2.0)**(1/2)) # The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
    
    y1 = ar_mean-(m*ar_sd) # Smallest grid point is the mean of the AR(1) process minus m*std.dev of AR(1) process.
    yn = ar_mean+(m*ar_sd) # Largest grid point is the mean of the AR(1) process plus m*std.dev of AR(1) process.
     
    y,d = linspace(y1,yn,N,endpoint=True,retstep=True) # Equally spaced grid. Include endpoint (endpoint=True) and record stepsize, d (retstep=True).
    
    #%% Compute transition probability matrix from state j (row) to k (column).
    
    ymatk = tile(expand_dims(y,axis=0),(N,1)) # Container for state next period.
    ymatj = mu+rho*ymatk.T # States this period.
    
    # In the following, loc and scale are the mean and std used to standardize the variable. # For example, norm.cdf(x,loc=y,scale=s) is the standard normal CDF evaluated at (x-y)/s.
    pmat = stats.norm.cdf(ymatk,loc=ymatj-(d/2.0),scale=sigma)-stats.norm.cdf(ymatk,loc=ymatj+(d/2.0),scale=sigma) # Transition probabilities to state 2, ..., N-1.
    pmat[:,0] = stats.norm.cdf(y[0],loc=mu+rho*y-(d/2.0),scale=sigma) # Transition probabilities to state 1.
    pmat[:,N-1] = 1.0-stats.norm.cdf(y[N-1],loc=mu+rho*y+(d/2.0),scale=sigma) # Transition probabilities to state N.
    
    #%% Output.
    
    y = expand_dims(y,axis=0) # Convert 0-dimensional array to a row vector.
    
    if count_nonzero(pmat.sum(axis=1)<0.999999) > 0:
        raise Exception("Some columns of transition matrix don't sum to 1.") 

    return y,pmat