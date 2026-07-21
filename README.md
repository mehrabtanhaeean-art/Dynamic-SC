## How to Use

The three scripts in this repository are executed in sequence. Each stage's 
output becomes the input to the next stage.

### Step 1: Run the simulation

Run `1- Simulation-All- Disruption.py` once for each of the 16 candidate 
minimum-inventory actions (see the paper for the action 
definitions), by adjusting the minimum-inventory multipliers for 
manufacturers and distribution centers before each run. Each run simulates 
the supply chain over a 20-year horizon under capacity, demand, and 
lead-time disruptions.

From each run, extract the weekly records occurring during the disruption 
period and classify each week into one of six service-level states. Use 
these classified records to build the empirical state-transition probability matrix for that action.

Repeat this for all 16 actions to obtain the full set of Reward and 
Transition matrices required by the Q-learning module.

### Step 2: Run Q-learning

Provide the Reward matrix (6x16) and the 16 Transition matrices (6x6 each) 
produced in Step 1 as input to `2- Q_Learning.py`. This script trains a 
tabular Q-learning agent and outputs the learned policy: the recommended 
minimum-inventory level at manufacturers and DCs for each service-level 
state.

### Step 3: Run the mathematical optimization model

Provide the minimum-inventory levels recommended by Q-learning (or by an 
alternative resilience strategy) as input to `3- Optimization.py`, which 
determines production, storage, and distribution decisions subject to a 
target service level. This step also requires defining the supply chain's 
operational parameters not produced by the earlier steps, including 
customer demand, production and transportation costs, capacity limits, 
and backup supplier parameters (see the paper).

### Status


These three stages are provided as separate, modular scripts rather than 
a single fixed pipeline, so that each can be adapted independently to the 
structure and requirements of a specific supply chain or industry. This 
modular design also allows the components to be combined into a single 
automated pipeline with minimal effort, should a fully integrated workflow 
be preferred for a given application.
