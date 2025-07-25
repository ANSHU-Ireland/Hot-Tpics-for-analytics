"""
Forward and reverse logistics vehicle routing with time windows, multiple depots,
an aggregate emission cap and a workload balancing cap.

We formulate a mixed integer linear program in PuLP.
We solve two variants for each instance size:
  1. Base model (no emission or balancing caps)
  2. Constrained model (tightened emission cap and a per route length cap L_max)

Author: <your name here>
Course: MIS41100 Hot Topics in Analytics
Date: 25 July 2025
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Customer:
    idx: int
    x: float
    y: float
    q_deliver: float
    q_pickup: float
    a: float  # earliest time
    b: float  # latest time
    service: float


@dataclass
class Depot:
    idx: int
    x: float
    y: float


@dataclass
class Instance:
    name: str
    depots: List[Depot]
    customers: List[Customer]
    vehicles: int
    capacity: float
    speed: float
    emission_rate: float  # kg CO2 per km
    distance: Dict[Tuple[int, int], float]
    travel_time: Dict[Tuple[int, int], float]
    all_nodes: List[int]
    depot_nodes: List[int]
    customer_nodes: List[int]


@dataclass
class Solution:
    objective: float
    total_emission: float
    vehicles_used: int
    route_lengths: Dict[int, float]
    routes: Dict[int, List[int]]
    arrival_times: Dict[Tuple[int, int], float]
    solve_time: float
    status: str
    mip_gap: float


# -----------------------------
# Utility functions
# -----------------------------

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_instance(n_customers: int,
                   n_depots: int,
                   vehicles: int,
                   capacity: float,
                   speed: float,
                   emission_rate: float,
                   seed: int,
                   horizon: float = 10.0) -> Instance:
    random.seed(seed)
    # Create depots on the corners of a square to encourage multiple depot use
    depots = []
    corners = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]
    for d in range(n_depots):
        x, y = corners[d % len(corners)]
        depots.append(Depot(idx=-(d + 1), x=x, y=y))  # negative indices for depots

    # Create customers randomly in the square
    customers = []
    for i in range(1, n_customers + 1):
        x = random.uniform(10.0, 90.0)
        y = random.uniform(10.0, 90.0)
        q_deliver = random.uniform(2.0, 8.0)
        q_pickup = random.uniform(1.0, 5.0)
        window_center = random.uniform(2.0, horizon - 2.0)
        window_width = random.uniform(1.0, 3.0)
        a = max(0.0, window_center - window_width / 2.0)
        b = min(horizon, window_center + window_width / 2.0)
        service = random.uniform(0.1, 0.3)
        customers.append(Customer(idx=i, x=x, y=y,
                                  q_deliver=q_deliver, q_pickup=q_pickup,
                                  a=a, b=b, service=service))

    # Build distance and travel time matrices on all nodes
    all_nodes = [c.idx for c in customers] + [d.idx for d in depots]
    depot_nodes = [d.idx for d in depots]
    customer_nodes = [c.idx for c in customers]

    coords = {c.idx: (c.x, c.y) for c in customers}
    coords.update({d.idx: (d.x, d.y) for d in depots})

    distance = {}
    travel_time = {}
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                distance[(i, j)] = 0.0
                travel_time[(i, j)] = 0.0
            else:
                d = euclidean(coords[i], coords[j])
                distance[(i, j)] = d
                travel_time[(i, j)] = d / speed

    return Instance(name=f"{n_customers}_cust_{n_depots}_dep",
                    depots=depots,
                    customers=customers,
                    vehicles=vehicles,
                    capacity=capacity,
                    speed=speed,
                    emission_rate=emission_rate,
                    distance=distance,
                    travel_time=travel_time,
                    all_nodes=all_nodes,
                    depot_nodes=depot_nodes,
                    customer_nodes=customer_nodes)


# -----------------------------
# Model builder
# -----------------------------

def build_model(inst: Instance,
                emission_cap: float = None,
                route_cap: float = None,
                time_limit: int = 3600) -> Tuple[pulp.LpProblem, Dict, Dict, Dict, Dict]:
    """
    Build a MILP for VRP with pickups and deliveries, time windows, multiple depots,
    aggregate emission cap, and per route distance cap.

    Returns the model and decision variable dictionaries.
    """

    M = 1e5  # big M

    # Sets
    K = range(inst.vehicles)
    V_nodes = inst.all_nodes
    C_nodes = inst.customer_nodes
    D_nodes = inst.depot_nodes

    # Convenience maps
    cust = {c.idx: c for c in inst.customers}

    # Model
    model = pulp.LpProblem("VRP_FWD_REV_TW_MD", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((i, j, k) for i in V_nodes for j in V_nodes for k in K),
                              lowBound=0, upBound=1, cat=pulp.LpBinary)

    t = pulp.LpVariable.dicts("t", ((i, k) for i in V_nodes for k in K),
                              lowBound=0, upBound=None, cat=pulp.LpContinuous)

    load = pulp.LpVariable.dicts("load", ((i, k) for i in V_nodes for k in K),
                                 lowBound=0, upBound=inst.capacity, cat=pulp.LpContinuous)

    y = pulp.LpVariable.dicts("y", ((i, k) for i in C_nodes for k in K),
                              lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: minimise total distance
    model += pulp.lpSum(inst.distance[(i, j)] * x[(i, j, k)]
                        for i in V_nodes for j in V_nodes for k in K if i != j)

    # 1. Customer service exactly once
    for i in C_nodes:
        model += pulp.lpSum(x[(j, i, k)] for k in K for j in V_nodes if j != i) == 1, f"visit_in_{i}"
        model += pulp.lpSum(x[(i, j, k)] for k in K for j in V_nodes if j != i) == 1, f"visit_out_{i}"

    # 2. Flow conservation for each k at customer nodes
    for k in K:
        for i in C_nodes:
            model += (pulp.lpSum(x[(i, j, k)] for j in V_nodes if j != i)
                      - pulp.lpSum(x[(j, i, k)] for j in V_nodes if j != i)) == 0, f"flow_{i}_{k}"

    # 3. Each vehicle starts at most once from a depot and ends at most once at a depot
    for k in K:
        model += pulp.lpSum(x[(d, j, k)] for d in D_nodes for j in V_nodes if j != d) <= 1, f"start_{k}"
        model += pulp.lpSum(x[(i, d, k)] for d in D_nodes for i in V_nodes if i != d) <= 1, f"end_{k}"

    # 4. Link y and x for customers
    for i in C_nodes:
        for k in K:
            model += pulp.lpSum(x[(j, i, k)] for j in V_nodes if j != i) == y[(i, k)], f"link_y_{i}_{k}"

    # 5. Time windows and sequencing
    for k in K:
        for i in C_nodes:
            model += t[(i, k)] >= cust[i].a - M * (1 - y[(i, k)]), f"tw_early_{i}_{k}"
            model += t[(i, k)] <= cust[i].b + M * (1 - y[(i, k)]), f"tw_late_{i}_{k}"

    for k in K:
        for i in V_nodes:
            for j in V_nodes:
                if i != j:
                    si = cust[i].service if i in C_nodes else 0.0
                    model += t[(j, k)] >= t[(i, k)] + si + inst.travel_time[(i, j)] - M * (1 - x[(i, j, k)]), \
                             f"time_prop_{i}_{j}_{k}"

    # 6. Load propagation
    for k in K:
        for i in V_nodes:
            for j in V_nodes:
                if i != j:
                    qd = cust[j].q_deliver if j in C_nodes else 0.0
                    qp = cust[j].q_pickup if j in C_nodes else 0.0
                    model += load[(j, k)] >= load[(i, k)] - qd + qp - M * (1 - x[(i, j, k)]), \
                             f"load_prop_lb_{i}_{j}_{k}"
                    model += load[(j, k)] <= load[(i, k)] - qd + qp + M * (1 - x[(i, j, k)]), \
                             f"load_prop_ub_{i}_{j}_{k}"

    # 7. Capacity bounds always enforced by variable bounds

    # 8. Initial load when leaving a depot cannot exceed capacity
    #    and we can set loads at depots to zero for simplicity
    for k in K:
        for d in D_nodes:
            model += load[(d, k)] == 0.0

    # 9. Aggregate emission cap (if provided)
    if emission_cap is not None:
        model += pulp.lpSum(inst.distance[(i, j)] * inst.emission_rate * x[(i, j, k)]
                            for i in V_nodes for j in V_nodes for k in K if i != j) <= emission_cap, "emission_cap"

    # 10. Workload balancing cap per route (if provided)
    if route_cap is not None:
        for k in K:
            model += pulp.lpSum(inst.distance[(i, j)] * x[(i, j, k)]
                                for i in V_nodes for j in V_nodes if i != j) <= route_cap, f"route_cap_{k}"

    # Time limit
    pulp.PULP_CBC_CMD.timeLimit = time_limit

    return model, x, t, load, y


# -----------------------------
# Solver wrapper
# -----------------------------

def solve_model(model: pulp.LpProblem,
                x, t, load, y,
                inst: Instance) -> Solution:
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=3600)
    status = model.solve(solver)
    end = time.time()

    status_str = pulp.LpStatus[model.status]
    mip_gap = getattr(model.solverModel, "gap", None)
    if mip_gap is None:
        try:
            mip_gap = model.solverModel.status().gap
        except Exception:
            mip_gap = None

    # Extract if feasible
    routes = {k: [] for k in range(inst.vehicles)}
    route_lengths = {k: 0.0 for k in range(inst.vehicles)}
    arrival_times = {}

    vehicles_used = 0
    if status_str in ["Optimal", "Not Solved", "Infeasible", "Unbounded"]:
        pass

    if status_str in ["Optimal", "Not Solved"]:
        for k in range(inst.vehicles):
            # Build route by following x
            # find start depot if any
            arcs = [(i, j) for i in inst.all_nodes for j in inst.all_nodes if i != j and pulp.value(x[(i, j, k)]) > 0.5]
            if not arcs:
                continue
            vehicles_used += 1
            # find the depot that starts
            starts = [i for (i, j) in arcs if i in inst.depot_nodes]
            current = starts[0] if starts else arcs[0][0]
            route = [current]
            length = 0.0

            while True:
                next_nodes = [j for (i, j) in arcs if i == current]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                route.append(nxt)
                length += inst.distance[(current, nxt)]
                current = nxt
                if current in inst.depot_nodes:
                    break
            routes[k] = route
            route_lengths[k] = length

        for i in inst.all_nodes:
            for k in range(inst.vehicles):
                arrival_times[(i, k)] = pulp.value(t[(i, k)])

    total_distance = pulp.value(model.objective)
    total_emission = total_distance * inst.emission_rate

    return Solution(objective=total_distance,
                    total_emission=total_emission,
                    vehicles_used=vehicles_used,
                    route_lengths=route_lengths,
                    routes=routes,
                    arrival_times=arrival_times,
                    solve_time=end - start,
                    status=status_str,
                    mip_gap=mip_gap if mip_gap is not None else float("nan"))


# -----------------------------
# Orchestration to run base and constrained
# -----------------------------

def run_pipeline(n_customers: int,
                 n_depots: int,
                 vehicles: int,
                 capacity: float,
                 seed: int) -> Tuple[Instance, Solution, Solution, float, float]:
    inst = build_instance(n_customers=n_customers,
                          n_depots=n_depots,
                          vehicles=vehicles,
                          capacity=capacity,
                          speed=40.0,                # km per hour
                          emission_rate=1.0,         # kg CO2 per km for simplicity
                          seed=seed,
                          horizon=10.0)

    # Base model
    base_model, x_b, t_b, load_b, y_b = build_model(inst, emission_cap=None, route_cap=None, time_limit=3600)
    base_sol = solve_model(base_model, x_b, t_b, load_b, y_b, inst)

    # Compute caps
    emission_cap = 0.90 * base_sol.total_emission  # ten percent tighter
    avg_route = (base_sol.objective / max(base_sol.vehicles_used, 1))
    route_cap = 1.20 * avg_route                    # twenty percent above the average as a fairness cap

    # Constrained model
    cons_model, x_c, t_c, load_c, y_c = build_model(inst,
                                                    emission_cap=emission_cap,
                                                    route_cap=route_cap,
                                                    time_limit=3600)
    cons_sol = solve_model(cons_model, x_c, t_c, load_c, y_c, inst)

    return inst, base_sol, cons_sol, emission_cap, route_cap


# -----------------------------
# Pretty printing
# -----------------------------

def print_solution(label: str, sol: Solution, capE: float = None, capL: float = None):
    print("=" * 80)
    print(label)
    print("=" * 80)
    print(f"Status                 : {sol.status}")
    print(f"Objective distance km  : {sol.objective:8.2f}")
    print(f"Total emission kg CO2  : {sol.total_emission:8.2f}")
    if capE is not None:
        print(f"Emission cap kg CO2    : {capE:8.2f}")
    print(f"Vehicles actually used : {sol.vehicles_used}")
    if capL is not None:
        print(f"Per route cap km       : {capL:8.2f}")
    print(f"Solve time seconds     : {sol.solve_time:8.2f}")
    print(f"MIP gap if reported    : {sol.mip_gap}")
    print("Route lengths by vehicle")
    for k, length in sol.route_lengths.items():
        if length > 1e-6:
            print(f"  vehicle {k:2d} length {length:8.2f} km")
    print("Routes")
    for k, route in sol.routes.items():
        if len(route) > 0:
            print(f"  vehicle {k:2d} route {route}")
    print("\n")


def main():
    # Three instances
    cases = [
        dict(n_customers=10, n_depots=2, vehicles=4, capacity=25.0, seed=42),
        dict(n_customers=15, n_depots=2, vehicles=5, capacity=28.0, seed=43),
        dict(n_customers=20, n_depots=3, vehicles=6, capacity=30.0, seed=44),
    ]

    summary_rows = []

    for params in cases:
        inst, base_sol, cons_sol, capE, capL = run_pipeline(**params)

        print("\n" + "#" * 80)
        print(f"Instance {inst.name}")
        print("#" * 80)
        print_solution("Base model", base_sol, None, None)
        print_solution("Constrained model", cons_sol, capE, capL)

        summary_rows.append({
            "instance": inst.name,
            "n_customers": len(inst.customers),
            "vehicles": inst.vehicles,
            "base_dist": base_sol.objective,
            "base_em": base_sol.total_emission,
            "base_used": base_sol.vehicles_used,
            "base_max_route": max(base_sol.route_lengths.values()),
            "base_time": base_sol.solve_time,
            "cons_dist": cons_sol.objective,
            "cons_em": cons_sol.total_emission,
            "cons_used": cons_sol.vehicles_used,
            "cons_max_route": max(cons_sol.route_lengths.values()),
            "cons_time": cons_sol.solve_time,
            "E_cap": capE,
            "L_cap": capL
        })

    print("\n" + "=" * 80)
    print("Compact summary")
    print("=" * 80)
    header = ("instance  cust  veh   base_km  cons_km  base_used  cons_used  "
              "base_max  cons_max  base_t  cons_t   E_cap  L_cap")
    print(header)
    for r in summary_rows:
        print(f"{r['instance']:10s} {r['n_customers']:4d} {r['vehicles']:3d} "
              f"{r['base_dist']:9.1f} {r['cons_dist']:8.1f} "
              f"{r['base_used']:10d} {r['cons_used']:10d} "
              f"{r['base_max_route']:9.1f} {r['cons_max_route']:9.1f} "
              f"{r['base_time']:7.1f} {r['cons_time']:7.1f} "
              f"{r['E_cap']:7.1f} {r['L_cap']:7.1f}")


if __name__ == "__main__":
    main()
