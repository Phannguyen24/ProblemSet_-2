"""

my_graph.py
-----------
This code plots the value and policy functions.

"""

#%% Imports from Python
from matplotlib.pyplot import figure,plot,xlabel,ylabel,title,show
from numpy import linspace,squeeze,zeros
from copy import deepcopy # Changed from numpy.copy
#%% Plot the model functions and simulations.
def track_profit(myClass):
    '''
    
    This function plots the model functions and simulations.
    
    Input:
        myClass : Model class with parameters, grids, cost and revenue functions, policy functions, and simulations.
        
    '''

    # Model parameters, policy and value functions, and simulations.
    par = myClass.par # Parameters.
    sol = myClass.sol # Policy functions.
    sim = myClass.sim # Simulations.
    
    #Plot capital policy function.
    
    figure(1)
    plot(par.kgrid,squeeze(sol.k[:,:,0]))
    xlabel('$k_t$')
    ylabel('$k_{t+1}$') 
    title('Capital Policy Function')

    
    # Plot investment policy function.
    
    figure(2)
    plot(par.kgrid,squeeze(sol.i[:,:,0]))
    xlabel('$k_t$')
    ylabel('$i_t$') 
    title('Investment Policy Function')
    
    # Plot revenue function.
    
    figure(3)
    plot(par.kgrid,squeeze(sol.r[:,:,0]))
    xlabel('$k_t$')
    ylabel('$r_t$') 
    title('Revenue Function')
    
    # Plot expenditure function.
    
    figure(4)
    plot(par.kgrid,squeeze(sol.e[:,:,0]))
    xlabel('$k_t$')
    ylabel('$C(k_{t+1},A_t,k_t)+pi_t$') 
    title('Expenditure Function')
    
    # Plot profit function.
    
    figure(5)
    plot(par.kgrid,squeeze(sol.p[:,:,0]))
    xlabel('$k_t$')
    ylabel('$C(k_{t+1},A_t,k_t)+pi_t$') 
    title('Profit Function')
    
    # Plot value function.
    
    figure(6)
    plot(par.kgrid,sol.v[:,:,0])
    xlabel('$k_t$')
    ylabel('$v_t$') 
    title('Value Function')
    
    # Plot simulated revenue shocks.

    tgrid = linspace(1,par.T,par.T,dtype=int)

    figure(7)
    plot(tgrid,sim.Asim)
    xlabel('Time')
    ylabel('$A^sim_t$') 
    title('Simulated Revenue Shocks')

    # Plot simulated capital choice.

    figure(8)
    plot(tgrid,sim.ksim)
    xlabel('Time')
    ylabel('$k^sim_t$') 
    title('Simulated Capital Choice')

    # Plot simulated investment expenditure.

    figure(9)
    plot(tgrid,sim.esim)
    xlabel('Time')
    ylabel('$C(k^{sim}_{t+1},A^{sim}_t,k^{sim}_t)+pi^{sim}_t$') 
    title('Simulated Investment Expenditure')

    # Plot simulated investment.

    figure(10)
    plot(tgrid,sim.isim)
    xlabel('Time')
    ylabel('$i^{sim}_t$') 
    title('Simulated Investment')

    # Plot simulated revenue.

    figure(11)
    plot(tgrid,sim.rsim)
    xlabel('Time')
    ylabel('$y^{sim}_t$') 
    title('Simulated Revenue')

    # Plot simulated profit.

    figure(12)
    plot(tgrid,sim.psim)
    xlabel('Time')
    ylabel('$\pi^{sim}_t$') 
    title('Simulated Profit')

    # Plot simulated value function.

    figure(13)
    plot(tgrid,sim.vsim)
    xlabel('Time')
    ylabel('$v^{sim}_t$') 
    title('Simulated Firm Value')

    show()