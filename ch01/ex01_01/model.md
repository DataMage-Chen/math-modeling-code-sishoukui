# Ex 1.1 Model

## Problem restatement
Produce two machine tools (A and B). Profit per unit: 4 (thousand yuan) for machine A, 3 (thousand yuan) for machine B. Machine A requires 2 hours on machine A and 1 hour on machine B. Machine B requires 1 hour on each of machines A, B, and C. Daily available hours: A = 10, B = 8, C = 7. Determine production quantities to maximize total profit.

## Decision variables
- x: daily quantity of machine A
- y: daily quantity of machine B

## Objective (maximize profit)
Maximize:

- 4x + 3y

## Constraints (machine-hour capacities)
- Machine A: 2x + y <= 10
- Machine B: x + y <= 8
- Machine C: y <= 7

## Non-negativity
- x >= 0
- y >= 0

## Integrality (optional)
If production quantities must be integers, enforce x, y as integers. Otherwise treat them as continuous (linear programming).
