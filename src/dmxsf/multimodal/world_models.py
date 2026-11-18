"""Plan and replan in a simulated grid for /wiki/multimodal/world-models/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/multimodal/world-models/
  English: https://dmxsf.com/en/wiki/multimodal/world-models/
"""

from __future__ import annotations

from collections import deque

Cell = tuple[int, int]
MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


def step(cell: Cell, move: Cell) -> Cell:
    return cell[0] + move[0], cell[1] + move[1]


def plan(
    start: Cell,
    goal: Cell,
    walls: set[Cell],
    shape: Cell,
) -> tuple[Cell, ...] | None:
    """Find a shortest sequence, or None if the goal is blocked."""
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        cell, actions = queue.popleft()
        if cell == goal:
            return actions
        for move in MOVES:
            row, col = target = step(cell, move)
            inside = 0 <= row < shape[0] and 0 <= col < shape[1]
            if inside and target not in walls and target not in seen:
                seen.add(target)
                queue.append((target, (*actions, move)))
    return None


def traverse(replan: bool, shape: Cell = (2, 4)) -> tuple[Cell, ...]:
    """Execute with an incorrect map and optional feedback."""
    start = (0, 0)
    goal = (0, shape[1] - 1)
    actual_walls = {(0, 2)}
    modeled_walls: set[Cell] = set()
    cell = start
    trace = [start]
    actions = plan(start, goal, modeled_walls, shape)
    while actions and cell != goal:
        move, *rest = actions
        target = step(cell, move)
        row, col = target
        inside = 0 <= row < shape[0] and 0 <= col < shape[1]
        if inside and target not in actual_walls:
            cell = target
        trace.append(cell)
        if replan and cell != target:
            modeled_walls.add(target)
            actions = plan(cell, goal, modeled_walls, shape)
        else:
            actions = tuple(rest)
    return tuple(trace)
