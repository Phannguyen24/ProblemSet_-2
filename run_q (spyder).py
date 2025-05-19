"""

run_slcm.py
-----------
This code solves the dynamic model of firm investment using value function iteration.

"""

#%% Import from folder
from model import firm
from solve import do_business
from simulate import earn_profit
from my_graph import track_profit

#%% Stochastic Growth Model.
some_company = firm()

# Set the parameters, state space, and utility function.
some_company.setup() # You can set the parameters here or use the defaults.

# Solve the model.
do_business(some_company) # Obtain the policy functions for consumption and savings.

# Simulate the model.
earn_profit(some_company) # Simulate forward in time.

# Graphs.
track_profit(some_company) # Plot policy functions and simulations.

 