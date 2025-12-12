"""
Tetris AI Module
================

This module implements the AI logic for the Sand Tetris game.
It calculates the best move (rotation and target X position) for the current piece
based on a scoring system that evaluates potential board states.

Strategies:
    1. Potential Energy Strategy: Used when the target color exists on the board.
       Encourages placing the piece near existing same-colored sand to facilitate flow.
    2. Valley Filling Strategy: Used when the target color is not present.
       Encourages filling gaps and keeping the surface flat.

Functions:
    compute_best_move: Main entry point to calculate the best move.
    evaluate_position: Scores a specific board state.
"""

import copy
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from typing import TypedDict

    from tetris_core import Pixel, SandtrisCore, ShapeCells

    class AIPlan(TypedDict):
        rotation_count: int
        target_x: int


# --- Weight Settings ---

# 1. Potential Energy Strategy (When same color exists on board)
W_OFFSET_X = 2.5  # Horizontal spread reward (within range, further is better)
W_POTENTIAL_H = 3.5  # Vertical potential reward (higher is better)
W_OVER_RANGE = 5.0  # Out of range penalty (coefficient should be larger than reward to drop score sharply)

# Parameter Settings
MAX_FLOW_RANGE = 4  # Maximum search radius
MAX_FLOW_RANGE_PPC = MAX_FLOW_RANGE * 4

# 2. Valley Filling Strategy (When no same color exists on board)
W_VALLEY = 6.0  # Valley filling reward

# 3. Basic Physical Constraints
W_HEIGHT = -1.0
W_HOLES = -4.0
W_BUMPINESS = -0.5


def compute_best_move(game: "SandtrisCore") -> "AIPlan":
    """
    Calculates the best move for the current piece.

    Simulates all possible rotations and horizontal positions to find the state
    with the highest score.

    Args:
        game (SandtrisCore): The current game state.

    Returns:
        AIPlan: A dictionary containing 'rotation_count' and 'target_x'.
    """
    best_score = -float("inf")
    best_move = (0, game.piece_x_px)

    original_shape = copy.deepcopy(game.current_shape_cells)
    current_test_shape = original_shape

    # Pre-scan for target color
    target_pixels = []
    for x in range(game.width_px):
        for y in range(game.height_px):
            if game.grid[y][x] != 0:
                if game.grid[y][x] == game.piece_color:
                    target_pixels.append((x, y))
                break  # Take only the top one for each column

    target_center_x = -1
    target_avg_y = -1
    has_target = False

    if target_pixels:
        has_target = True
        sum_x = sum(p[0] for p in target_pixels)
        sum_y = sum(p[1] for p in target_pixels)
        count = len(target_pixels)
        target_center_x = sum_x / count
        target_avg_y = sum_y / count

    # Change MAX_FLOW_RANGE by ppc
    global MAX_FLOW_RANGE_PPC
    MAX_FLOW_RANGE_PPC = MAX_FLOW_RANGE * game.ppc

    # Simulation
    for rot_idx in range(4):
        pixels_shape = _get_shape_pixels_relative(current_test_shape, game.ppc)
        min_x = min(p[0] for p in pixels_shape)
        max_x = max(p[0] for p in pixels_shape)

        start_x = -min_x
        end_x = game.width_px - 1 - max_x
        base_mod = game.piece_x_px % game.ppc
        valid_x_range = range(start_x, end_x + 1, game.ppc)

        for tx in valid_x_range:
            if (tx % game.ppc) != base_mod:
                continue

            land_y = _simulate_drop_y(game, tx, current_test_shape)
            if land_y is None:
                continue

            score = evaluate_position(
                game, tx, land_y, current_test_shape, has_target, target_center_x, target_avg_y
            )

            if score > best_score:
                best_score = score
                best_move = (rot_idx, tx)

        # Prepare for next rotation
        current_test_shape = _rotate_shape_simulate(current_test_shape)

    return {"rotation_count": best_move[0], "target_x": best_move[1]}


def evaluate_position(
    game: "SandtrisCore",
    x: int,
    y: int,
    shape_cells: "ShapeCells",
    has_target: bool,
    target_center_x: float,
    target_avg_y: float,
) -> float:
    """
    Evaluates a specific board position after a simulated drop.

    Args:
        game (SandtrisCore): The game state.
        x (int): The target X position (pixels).
        y (int): The landing Y position (pixels).
        shape_cells (ShapeCells): The shape cells configuration.
        has_target (bool): Whether the target color exists on the board.
        target_center_x (float): Center X of the target color mass.
        target_avg_y (float): Average Y of the target color mass.

    Returns:
        float: The calculated score for this position.
    """
    score = 0
    pixels = game.get_projected_pixels(x, y, shape_cells)
    my_pixels_set = set(pixels)

    sum_y = 0
    bottom_pixels = {}

    for px, py in pixels:
        if py < 0:
            return -999999
        sum_y += py
        if px not in bottom_pixels or py > bottom_pixels[px]:
            bottom_pixels[px] = py

        if py + 1 < game.height_px:
            if game.grid[py + 1][px] == 0 and (px, py + 1) not in my_pixels_set:
                score += W_HOLES

    piece_xs = [p[0] for p in pixels]
    current_piece_center_x = sum(piece_xs) / len(piece_xs)
    current_piece_y = y

    # ==========================================
    # Strategy Branch
    # ==========================================

    if has_target:
        # --- Strategy A: Potential Energy Impact (Limited Range Version) ---

        # 1. Horizontal Offset - Peak Scoring
        dist_x = abs(current_piece_center_x - target_center_x)

        if dist_x <= MAX_FLOW_RANGE_PPC:
            # Within range: further distance gets higher score (encourage spreading)
            score += dist_x * W_OFFSET_X
        else:
            # Out of range: score drops sharply
            # Base score is MAX_FLOW_RANGE * W_OFFSET_X (highest point)
            # Penalty term is (excess distance * penalty coefficient)
            base_max_score = MAX_FLOW_RANGE_PPC * W_OFFSET_X
            excess_dist = dist_x - MAX_FLOW_RANGE_PPC

            # Here we let the score decrease rapidly, even to negative values
            penalty = excess_dist * W_OVER_RANGE
            score += base_max_score - penalty

        # 2. Vertical Potential
        diff_h = target_avg_y - current_piece_y
        if diff_h > 0:
            # Optimization: If distance is too far (exceeds MAX_FLOW_RANGE), vertical potential reward should be discounted
            # Because if it's too far, it won't flow over even if it's high
            if dist_x > MAX_FLOW_RANGE_PPC:
                score += diff_h * W_POTENTIAL_H * 0.2  # Significantly reduce potential reward when far away
            else:
                score += diff_h * W_POTENTIAL_H

    else:
        # --- Strategy B: Valley Filling Defense ---
        valley_score = 0
        for bx, by in bottom_pixels.items():
            left_solid = (bx == 0) or (game.grid[by][bx - 1] != 0)
            right_solid = (bx == game.width_px - 1) or (game.grid[by][bx + 1] != 0)
            if left_solid and right_solid:
                valley_score += W_VALLEY
        score += valley_score

    # --- General Metrics ---
    avg_y = sum_y / len(pixels)
    score += avg_y * -(W_HEIGHT)

    bumpiness = _calculate_local_bumpiness(game, pixels)
    score += bumpiness * abs(W_BUMPINESS)

    return score


# --- Helper Functions ---


def _calculate_local_bumpiness(game: "SandtrisCore", pixels: List["Pixel"]) -> int:
    """
    Calculates local bumpiness penalty (negative value).
    Only checks the affected X range.
    """
    xs = [p[0] for p in pixels]
    min_x, max_x = min(xs), max(xs)

    # Extend check range by one to compare edges
    check_start = max(0, min_x - 1)
    check_end = min(game.width_px - 1, max_x + 1)

    # Get heights of these columns
    col_heights = {}

    # 1. Get original heights
    for cx in range(check_start, check_end + 1):
        h = 0
        for cy in range(game.height_px):
            if game.grid[cy][cx] != 0:
                h = game.height_px - cy
                break
        col_heights[cx] = h

    # 2. Simulate height after adding new block
    for px, py in pixels:
        new_h = game.height_px - py
        if new_h > col_heights[px]:
            col_heights[px] = new_h

    # 3. Calculate adjacent differences
    total_bumpiness = 0
    for cx in range(check_start, check_end):  # Compare pairwise
        total_bumpiness += abs(col_heights[cx] - col_heights[cx + 1])

    # Return negative score
    return -total_bumpiness


def _get_shape_pixels_relative(shape_cells: "ShapeCells", ppc: int) -> List["Pixel"]:
    """Cell coordinates -> Relative Pixel coordinates"""
    pixels = []
    for cx, cy in shape_cells:
        for i in range(ppc):
            for j in range(ppc):
                pixels.append((cx * ppc + i, cy * ppc + j))
    return pixels


def _rotate_shape_simulate(shape_cells: "ShapeCells") -> "ShapeCells":
    if not shape_cells:
        return []
    cx = sum(c[0] for c in shape_cells) / len(shape_cells)
    cy = sum(c[1] for c in shape_cells) / len(shape_cells)
    center = (round(cx), round(cy))
    new_shape = []
    for x, y in shape_cells:
        rx = center[0] - (y - center[1])
        ry = center[1] + (x - center[0])
        new_shape.append((int(rx), int(ry)))
    min_x = min(p[0] for p in new_shape)
    min_y = min(p[1] for p in new_shape)
    return [(x - min_x, y - min_y) for x, y in new_shape]


def _simulate_drop_y(game: "SandtrisCore", px_x: int, shape_cells: "ShapeCells") -> Optional[int]:
    # Start trying from current actual height (or slightly higher) to save time
    # For safety, start searching from -10 (assuming block won't be higher than -10)
    # Or if Core already spawns around -10, start directly from game.piece_y_px
    y = game.piece_y_px

    # Safety check: if current y already collides (should not happen in AI prediction, but just in case)
    if game.check_collision(px_x, y, shape_cells):
        return None

    while True:
        if game.check_collision(px_x, y + 1, shape_cells):
            return y
        y += 1
        if y >= game.height_px:
            return game.height_px
