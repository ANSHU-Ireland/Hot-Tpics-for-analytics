# Vehicle Routing Problem with Forward and Reverse Logistics

A Python implementation of a Vehicle Routing Problem (VRP) that handles both forward (deliveries) and reverse (pickups) logistics, with time windows, multiple depots, emission constraints, and workload balancing. The model is formulated as a Mixed Integer Linear Program (MILP) using PuLP and solved with the CBC solver.

## Features

- **Forward and Reverse Logistics:** Each customer has delivery and pickup demands.
- **Time Windows:** Customers have earliest and latest service times.
- **Multiple Depots:** Vehicles can start and end at different depot locations.
- **Emission Constraints:** Aggregate CO₂ emission budget (constrained model).
- **Workload Balancing:** Per-route distance cap for fair workload (constrained model).
- **Capacity Constraints:** Vehicle capacity limits for combined deliveries and pickups.

## Problem Formulation

For each instance, the script solves two variants:
1. **Base Model:** Minimizes total distance with core VRP constraints.
2. **Constrained Model:** Adds an emission cap (10% tighter than base) and a per-route distance cap (20% above average route length).

## Requirements

- Python 3.10 or later
- PuLP optimization library (includes CBC solver)

Install PuLP with:
```bash
pip install pulp
```

## Usage

1. Ensure the script is named `vrp.py`.
2. Run the script:
   ```bash
   python vrp.py
   ```

## What the Script Does

- **Generates three synthetic instances**:
  - 10 customers, 2 depots, 4 vehicles (capacity 25)
  - 15 customers, 2 depots, 5 vehicles (capacity 28)
  - 20 customers, 3 depots, 6 vehicles (capacity 30)
- **Solves each instance twice** (base and constrained models).
- **Prints results** for each instance:
  - Status and total distance for both models
  - Emission and route caps for the constrained model

## Model Details

### Decision Variables

- `x[i,j,k]`: 1 if vehicle k travels from node i to node j, 0 otherwise
- `t[i,k]`: Arrival time of vehicle k at node i
- `load[i,k]`: Load of vehicle k after leaving node i
- `y[i,k]`: 1 if vehicle k serves customer i, 0 otherwise

### Constraints

- Each customer is visited exactly once.
- Flow conservation at all nodes.
- Vehicle capacity limits.
- Time window compliance.
- Load propagation (deliveries reduce load, pickups increase load).
- Emission cap (constrained model only).
- Per-route distance cap (constrained model only).

### Objective

Minimize total distance traveled by all vehicles.

## Customization

You can modify the script to:
- Change the number of customers, depots, vehicles, or vehicle capacity.
- Adjust emission and route cap percentages.
- Change the random seed for instance generation.

## Author

Course: MIS41100 Hot Topics in Analytics  
Date: 25 July 2025

## Repository

https://github.com/ANSHU-Ireland/Hot-Tpics-for-analytics.git
