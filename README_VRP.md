# Vehicle Routing Problem with Forward and Reverse Logistics

A mixed integer linear programming (MILP) implementation of a Vehicle Routing Problem (VRP) that handles forward and reverse logistics with time windows, multiple depots, emission constraints, and workload balancing.

## Features

- **Forward and Reverse Logistics**: Handles both delivery and pickup operations
- **Time Windows**: Customers have specific service time windows
- **Multiple Depots**: Vehicles can start and end at different depot locations
- **Emission Constraints**: Aggregate CO2 emission budget constraints
- **Workload Balancing**: Per-route distance caps for fair driver workload distribution
- **Capacity Constraints**: Vehicle capacity limits for combined deliveries and pickups

## Problem Formulation

The model solves two variants for each instance:

1. **Base Model**: Minimizes total distance with only core VRP constraints
2. **Constrained Model**: Adds emission budget (10% tighter than base) and workload balancing caps (20% above average route length)

## Requirements

- Python 3.10 or later
- PuLP optimization library (includes CBC solver)

## Installation

1. Make sure you have Python 3.10 or later installed
2. Install the required dependency:

```bash
pip install pulp
```

## Usage

1. Copy the script into a file called `vrp_forward_reverse.py`
2. Run the script from the command line:

```bash
python vrp_forward_reverse.py
```

## What the Script Does

The script automatically:

1. **Generates three synthetic instances** with:
   - 10 customers, 2 depots, 4 vehicles (capacity 25)
   - 15 customers, 2 depots, 5 vehicles (capacity 28)  
   - 20 customers, 3 depots, 6 vehicles (capacity 30)

2. **Solves each instance twice**:
   - Base model: Minimizes total distance with core constraints only
   - Constrained model: Adds emission budget and workload balancing constraints

3. **Outputs detailed results**:
   - Route-by-route breakdown for each vehicle
   - Summary statistics (distance, emissions, vehicles used, solve time)
   - Compact comparison table across all instances

## Model Details

### Decision Variables
- `x[i,j,k]`: Binary variable indicating if vehicle k travels from node i to node j
- `t[i,k]`: Continuous variable for arrival time of vehicle k at node i
- `load[i,k]`: Continuous variable for vehicle k's load when leaving node i
- `y[i,k]`: Binary variable indicating if vehicle k serves customer i

### Key Constraints
- Customer service requirements (each customer visited exactly once)
- Flow conservation at all nodes
- Vehicle capacity limits
- Time window compliance
- Load propagation (deliveries reduce load, pickups increase load)
- Emission budget (constrained model only)
- Route length balancing (constrained model only)

### Objective
Minimize total distance traveled by all vehicles.

## Expected Results

The constrained model typically:
- Uses one additional vehicle compared to the base model
- Reduces total distance to meet the emission budget
- Balances workload more evenly across drivers
- Increases solve time due to additional constraints

For the provided test instances:
- Small instances (10-15 customers): Solve to optimality in under 10 minutes
- Larger instances (20 customers): May reach time limit with small optimality gaps

## Customization

You can modify the script to:
- Change instance parameters (number of customers, depots, vehicles)
- Adjust constraint tightness (emission cap percentage, route balancing factor)
- Modify time limits and solver parameters
- Add new constraints or objectives

## Algorithm

The implementation uses:
- **Solver**: CBC (Coin-or Branch and Cut) mixed integer solver via PuLP
- **Formulation**: Arc-based MILP with time and load variables
- **Time Limit**: 1 hour per solve (configurable)

## Author

Course: MIS41100 Hot Topics in Analytics  
Date: 25 July 2025

## Repository

[Add your GitHub repository link here]
