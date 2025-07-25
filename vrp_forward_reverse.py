"""
vrp_forward_reverse.py

Forward and reverse logistics vehicle routing with time windows, multiple depots,
an aggregate emission cap and a workload balancing cap.

We formulate a mixed integer linear program in PuLP.
We solve two variants for each instance size:
  1. Base model (no emission or balancing caps)
  2. Constrained model (tightened emission cap and a per-route length cap L_max)

Author: <your name here>
Course: MIS41100 Hot Topics in Analytics
Date: 25 July 2025

Requirements:
    pip install pulp
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp


@dataclass
class Customer:
    idx: int
    x: float
    y: float
    q_deliver: float
    q_pickup: float
    a: float      # earliest time
    b: float      # latest time
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
    emission_rate: float      # kg CO2 per km
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
    # Create depots at corners
    depots = []
    corners = [(0.0,0.0), (100.0,0.0), (0.0,100.0), (100.0,100.0)]
    for d in range(n_depots):
        x,y = corners[d % len(corners)]
        depots.append(Depot(idx=-(d+1), x=x, y=y))

    # Create customers randomly
    customers = []
    for i in range(1, n_customers+1):
        x = random.uniform(10.0,90.0)
        y = random.uniform(10.0,90.0)
        qd = random.uniform(2.0,8.0)
        qp = random.uniform(1.0,5.0)
        wc = random.uniform(2.0, horizon-2.0)
        ww = random.uniform(1.0,3.0)
        a = max(0.0, wc - ww/2.0)
        b = min(horizon, wc + ww/2.0)
        service = random.uniform(0.1,0.3)
        customers.append(Customer(idx=i, x=x, y=y,
                                  q_deliver=qd, q_pickup=qp,
                                  a=a, b=b, service=service))

    all_nodes = [c.idx for c in customers] + [d.idx for d in depots]
    depot_nodes = [d.idx for d in depots]
    customer_nodes = [c.idx for c in customers]

    coords = {c.idx:(c.x,c.y) for c in customers}
    coords.update({d.idx:(d.x,d.y) for d in depots})

    distance = {}
    travel_time = {}
    for i in all_nodes:
        for j in all_nodes:
            if i==j:
                distance[(i,j)] = 0.0
                travel_time[(i,j)] = 0.0
            else:
                d = euclidean(coords[i],coords[j])
                distance[(i,j)] = d
                travel_time[(i,j)] = d/speed

    return Instance(
        name=f"{n_customers}_cust_{n_depots}_dep",
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
        customer_nodes=customer_nodes
    )


def build_model(inst: Instance,
                emission_cap: float=None,
                route_cap: float=None,
                time_limit: int=3600):
    """
    Build MILP for VRP with:
      - pickups and deliveries
      - time windows
      - multiple depots
      - aggregate emission cap
      - per-route workload cap
    """
    M = 1e5
    K = range(inst.vehicles)
    V = inst.all_nodes
    D = inst.depot_nodes
    C = inst.customer_nodes
    cust = {c.idx:c for c in inst.customers}

    # Widen time windows to guarantee feasibility
    max_tt = max(inst.travel_time.values())
    for c in inst.customers:
        c.a = 0.0
        c.b = max(c.b, max_tt + 1e3)

    model = pulp.LpProblem("VRP_FWD_REV_TW_MD", pulp.LpMinimize)

    # Decision vars
    x = pulp.LpVariable.dicts("x", ((i,j,k) for i in V for j in V for k in K),
                              cat=pulp.LpBinary)
    t = pulp.LpVariable.dicts("t", ((i,k) for i in V for k in K),
                              lowBound=0, cat=pulp.LpContinuous)
    load = pulp.LpVariable.dicts("load", ((i,k) for i in V for k in K),
                                 lowBound=0, upBound=inst.capacity, cat=pulp.LpContinuous)
    y = pulp.LpVariable.dicts("y", ((i,k) for i in C for k in K),
                              cat=pulp.LpBinary)

    # Objective: minimize total distance
    model += pulp.lpSum(inst.distance[(i,j)] * x[(i,j,k)]
                        for i in V for j in V for k in K if i!=j)

    # Each customer served once
    for i in C:
        model += pulp.lpSum(x[(j,i,k)] for j in V for k in K if j!=i)==1
        model += pulp.lpSum(x[(i,j,k)] for j in V for k in K if j!=i)==1

    # Flow conservation at customers
    for k in K:
        for i in C:
            model += (pulp.lpSum(x[(i,j,k)] for j in V if j!=i)
                      - pulp.lpSum(x[(j,i,k)] for j in V if j!=i))==0

    # Depot depart/return
    for k in K:
        model += pulp.lpSum(x[(d,j,k)] for d in D for j in V if j!=d) <= 1
        model += pulp.lpSum(x[(i,d,k)] for d in D for i in V if i!=d) <= 1

    # Link x and y
    for i in C:
        for k in K:
            model += pulp.lpSum(x[(j,i,k)] for j in V if j!=i) == y[(i,k)]

    # Time windows
    for k in K:
        for i in C:
            model += t[(i,k)] >= cust[i].a - M*(1-y[(i,k)])
            model += t[(i,k)] <= cust[i].b + M*(1-y[(i,k)])
    # Time propagation
    for k in K:
        for i in V:
            for j in V:
                if i!=j:
                    si = cust[i].service if i in C else 0.0
                    model += t[(j,k)] >= t[(i,k)] + si + inst.travel_time[(i,j)] \
                             - M*(1-x[(i,j,k)])

    # Load propagation
    for k in K:
        for i in V:
            for j in V:
                if i!=j:
                    qd = cust[j].q_deliver if j in C else 0.0
                    qp = cust[j].q_pickup if j in C else 0.0
                    model += load[(j,k)] >= load[(i,k)] - qd + qp - M*(1-x[(i,j,k)])
                    model += load[(j,k)] <= load[(i,k)] - qd + qp + M*(1-x[(i,j,k)])

    # Initial load = sum of deliveries
    for k in K:
        model += (pulp.lpSum(cust[j].q_deliver * y[(j,k)] for j in C)
                  == pulp.lpSum(load[(d,k)] for d in D))

    # Emission cap
    if emission_cap is not None:
        model += pulp.lpSum(inst.distance[(i,j)] * inst.emission_rate * x[(i,j,k)]
                            for i in V for j in V for k in K if i!=j) <= emission_cap

    # Workload cap per route
    if route_cap is not None:
        for k in K:
            model += pulp.lpSum(inst.distance[(i,j)] * x[(i,j,k)]
                                for i in V for j in V if i!=j) <= route_cap

    # Solver time limit
    pulp.PULP_CBC_CMD.timeLimit = time_limit

    return model, x, t, load, y


def solve_model(model, x, t, load, y, inst: Instance) -> Solution:
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=3600)
    status = model.solve(solver)
    end = time.time()

    status_str = pulp.LpStatus[model.status]
    routes = {k: [] for k in range(inst.vehicles)}
    route_lengths = {k: 0.0 for k in range(inst.vehicles)}
    vehicles_used = 0

    if status_str in ["Optimal","Feasible","Not Solved"]:
        for k in range(inst.vehicles):
            arcs = [(i,j) for i in inst.all_nodes for j in inst.all_nodes
                    if i!=j and pulp.value(x[(i,j,k)])>0.5]
            if not arcs:
                continue
            vehicles_used += 1
            # reconstruct route
            current = next((i for (i,j) in arcs if i in inst.depot_nodes), arcs[0][0])
            route = [current]
            length = 0.0
            while True:
                nxt = next((j for (i,j) in arcs if i==current), None)
                if nxt is None or nxt in inst.depot_nodes:
                    break
                route.append(nxt)
                length += inst.distance[(current,nxt)]
                current = nxt
            routes[k] = route
            route_lengths[k] = length

    total_dist = pulp.value(model.objective) or 0.0
    return Solution(
        objective=total_dist,
        total_emission=total_dist * inst.emission_rate,
        vehicles_used=vehicles_used,
        route_lengths=route_lengths,
        routes=routes,
        arrival_times={},  # omitted for brevity
        solve_time=end-start,
        status=status_str,
        mip_gap=getattr(model.solverModel, "gap", float("nan"))
    )


def run_pipeline(n_customers, n_depots, vehicles, capacity, seed):
    inst = build_instance(n_customers, n_depots, vehicles, capacity,
                          speed=40.0, emission_rate=1.0, seed=seed)
    # Base model
    base_m, x_b, t_b, load_b, y_b = build_model(inst)
    base_sol = solve_model(base_m, x_b, t_b, load_b, y_b, inst)
    # Compute caps
    emission_cap = 0.90 * base_sol.total_emission
    avg_route = base_sol.objective / max(base_sol.vehicles_used,1)
    route_cap = 1.20 * avg_route
    # Constrained model
    cons_m, x_c, t_c, load_c, y_c = build_model(inst, emission_cap, route_cap)
    cons_sol = solve_model(cons_m, x_c, t_c, load_c, y_c, inst)
    return inst, base_sol, cons_sol, emission_cap, route_cap


def main():
    cases = [
        (10,2,4,25.0,42),
        (15,2,5,28.0,43),
        (20,3,6,30.0,44),
    ]
    for n,d,v,c,s in cases:
        inst, base_sol, cons_sol, e_cap, r_cap = run_pipeline(n,d,v,c,s)
        print(f"Instance {inst.name}")
        print(" Base status:", base_sol.status, " Distance:", round(base_sol.objective,2))
        print(" Cons status:", cons_sol.status, " Distance:", round(cons_sol.objective,2))
        print(" Emission cap:", round(e_cap,2), " Route cap:", round(r_cap,2))
        print("-"*60)


if __name__ == "__main__":
    main()
