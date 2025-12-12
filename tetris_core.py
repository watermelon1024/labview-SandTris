"""
Tetris Core Module
==================

This module implements the core logic of the Sand Tetris game, including:
- Game state management (grid, score, active piece).
- Physics simulation (sand falling, shattering).
- Collision detection.
- Line clearing logic (BFS-based).
- Rendering utilities.

Classes:
    SandtrisCore: The main game engine class.

Functions:
    init: Factory function to create a new game instance.
    update: Updates the game state by one step.
    get_view: Returns the current game view as a grid of colors.
    get_statistics: Returns current game statistics.
"""

import itertools
import random
import time
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple

from tetris_ai import compute_best_move

if TYPE_CHECKING:
    from tetris_ai import AIPlan


# --- Type Definitions ---
Pixel = Tuple[int, int]
ShapeCells = List[Pixel]
Grid = List[List[int]]
Statistics = Tuple[int, str, int, bool]

# --- Constants ---
DEFAULT_COLS = 10
DEFAULT_ROWS = 20
DEFAULT_PPC = 4  # Pixels Per Cell (Each block cell consists of 4x4 pixels)

# Actions
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_DOWN = 4  # Soft drop
ACTION_DROP = 5  # Hard drop
ACTION_AI = 6  # Let AI decide

# Hardness Levels
HARDNESS_EASY = 1
HARDNESS_MEDIUM = 2
HARDNESS_HARD = 3

HARDNESS_COLOR_MAPPING = {
    HARDNESS_EASY: 4,  # 4 colors
    HARDNESS_MEDIUM: 6,  # 6 colors
    HARDNESS_HARD: 7,  # 7 colors
}

# Colors
# 0 is empty
COLORS = [1, 2, 3, 4, 5, 6, 7]
COLOR_TO_RGB_MAPPING = {
    0: (210, 210, 210),  # Gray (Background)
    1: (255, 49, 49),  # Red
    2: (0, 191, 99),  # Green
    3: (0, 37, 204),  # Blue
    4: (255, 117, 31),  # Orange
    5: (56, 182, 255),  # Cyan
    6: (203, 108, 230),  # Magenta
    7: (255, 255, 255),  # White
}
COLOR_TO_24BIT = {color: (r << 16) | (g << 8) | b for color, (r, g, b) in COLOR_TO_RGB_MAPPING.items()}

# Tetromino Shapes (Defined in Cells)
SHAPES = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
}
SPACE_TO_INDEX_MAPPING = {shape: index for index, shape in enumerate(SHAPES.keys())}


class SandtrisCore:
    """
    The main game engine for Sand Tetris.

    Attributes:
        cols (int): Number of columns in cells.
        rows (int): Number of rows in cells.
        ppc (int): Pixels per cell.
        width_px (int): Total width in pixels.
        height_px (int): Total height in pixels.
        grid (Grid): The physical grid storing static sand (0 for empty, color ID for sand).
        score (int): Current game score.
        game_over (bool): Whether the game has ended.
        hardness (int): Current difficulty level.
    """

    def __init__(
        self,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        ppc: int = DEFAULT_PPC,
        hardness: int = HARDNESS_MEDIUM,
    ):
        """
        Initialize the game engine.

        Args:
            cols (int): Number of columns.
            rows (int): Number of rows.
            ppc (int): Pixels per cell.
            hardness (int): Difficulty level (1-3).
        """
        self.cols = cols
        self.rows = rows
        self.ppc = ppc
        self.width_px = cols * ppc
        self.height_px = rows * ppc

        # Physical Grid (Stores static sand)
        self.grid: Grid = [[0 for _ in range(self.width_px)] for _ in range(self.height_px)]

        self.score: int = 0
        self.continuous_bonus: float = 1.0
        self.start_time: float = time.time()
        self.game_over: bool = False
        self._just_shattered: bool = False
        self.gravity: int = 1  # in pixels per tick
        self.hardness: int = hardness

        # Active Piece (Rigid Body State)
        self.current_shape_cells: ShapeCells = []
        # List of (x, y) in CELL coordinates relative to piece center
        self.piece_x_px: int = 0  # Top-left X in PIXELS
        self.piece_y_px: int = 0  # Top-left Y in PIXELS
        self.piece_color: int = 1
        self.ai_plan: Optional[AIPlan] = None
        self.next_shape: str = ""
        self.next_piece_color: int = 1

        # Set colors based on hardness
        COLORS[:] = COLORS[: HARDNESS_COLOR_MAPPING.get(hardness, 5)]

        self.generate_next_piece()
        self.spawn_piece()

    def set_hardness(self, hardness: int) -> None:
        """Sets the game difficulty and updates available colors."""
        self.hardness = hardness
        COLORS[:] = COLORS[: HARDNESS_COLOR_MAPPING.get(hardness, 5)]

    def generate_next_piece(self) -> None:
        """Generates the next tetromino shape and color."""
        self.next_shape = random.choice(list(SHAPES.keys()))
        self.next_piece_color = random.choice(COLORS)

    def spawn_piece(self) -> None:
        """
        Spawns a new active piece at the top of the board.
        Resets the AI plan and continuous score bonus.
        """
        if not self.next_shape:
            self.generate_next_piece()
        self.current_shape_cells = SHAPES[self.next_shape]
        self.piece_color = self.next_piece_color
        self.generate_next_piece()

        # Initial Position (Center Top)
        # Calculate width in cells to center it
        min_cx = min(x for x, y in self.current_shape_cells)
        max_cx = max(x for x, y in self.current_shape_cells)
        piece_width_cells = max_cx - min_cx + 1
        start_col = (self.cols - piece_width_cells) // 2

        # Start just above the visible area
        max_cy = max(y for x, y in self.current_shape_cells)
        start_row = -(max_cy + 1)

        self.piece_x_px = start_col * self.ppc
        self.piece_y_px = start_row * self.ppc

        # Reset score bonus
        self.continuous_bonus = 1.0

        # --- Reset AI Plan ---
        self.ai_plan = None

    def get_projected_pixels(self, px_x: int, px_y: int, shape_cells: ShapeCells) -> List[Pixel]:
        """
        Converts "Cell Coordinates" to a list of real "Pixel Coordinates".

        This is the key to "Shattering": turning large blocks into small sand grains.

        Args:
            px_x (int): Top-left X coordinate in pixels.
            px_y (int): Top-left Y coordinate in pixels.
            shape_cells (ShapeCells): List of cell coordinates for the shape.

        Returns:
            List[Pixel]: A list of (x, y) tuples representing all pixels occupied by the shape.
        """
        pixels = []
        for cx, cy in shape_cells:
            # Each Cell converts to ppc * ppc pixels
            base_px = px_x + cx * self.ppc
            base_py = px_y + cy * self.ppc

            for i in range(self.ppc):
                for j in range(self.ppc):
                    pixels.append((base_px + i, base_py + j))
        return pixels

    def check_collision(self, px_x: int, px_y: int, shape_cells: ShapeCells) -> bool:
        """
        Checks if the rigid body block collides with boundaries or existing sand.

        Args:
            px_x (int): X coordinate in pixels.
            px_y (int): Y coordinate in pixels.
            shape_cells (ShapeCells): The shape to check.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        # Optimization: Check only boundary pixels of each Cell, or check all converted pixels
        # For accuracy, we check all projected pixels
        pixels = self.get_projected_pixels(px_x, px_y, shape_cells)

        for x, y in pixels:
            # Wall / Floor Collision
            if x < 0 or x >= self.width_px or y >= self.height_px:
                return True

            # Sand Collision (Check grid only when y >= 0 to avoid errors during spawning)
            if y >= 0 and self.grid[y][x] != 0:
                return True

        return False

    def rotate_piece(self) -> None:
        """
        Rotates the current piece 90 degrees clockwise.
        Handles wall kicks to prevent the piece from rotating into walls.
        """
        # Standard Rotation: (x, y) -> (-y, x) around center?
        # Simplified: Rotate around first block or center of bounding box
        # Let's try rotating relative to (1,1) roughly
        new_shape = []

        # Find center of rotation (approximate)
        cx = sum(c[0] for c in self.current_shape_cells) / len(self.current_shape_cells)
        cy = sum(c[1] for c in self.current_shape_cells) / len(self.current_shape_cells)
        center = (round(cx), round(cy))

        for x, y in self.current_shape_cells:
            # Rotate 90 deg clockwise
            rx = center[0] - (y - center[1])
            ry = center[1] + (x - center[0])
            new_shape.append((int(rx), int(ry)))

        # Re-normalize to top-left 0,0 to prevent drifting away from piece_x_px
        min_x = min(p[0] for p in new_shape)
        min_y = min(p[1] for p in new_shape)
        new_shape = [(x - min_x, y - min_y) for x, y in new_shape]

        # Wall Kick (Wall push correction)
        # Calculate absolute pixel boundaries of the new shape at current position
        # Since each cell width is ppc, we need to calculate the leftmost and rightmost pixel points
        # min_cell_x is definitely 0 (normalized above), so the leftmost is self.piece_x_px
        # We only need to calculate if the rightmost side exceeds the boundary
        max_cell_x = max(p[0] for p in new_shape)
        # Predicted leftmost and rightmost pixel X coordinates
        current_min_px = self.piece_x_px
        current_max_px = self.piece_x_px + (max_cell_x * self.ppc) + (self.ppc - 1)

        offset_x = 0

        if current_min_px < 0:  # Check Left Wall
            # If less than 0, push right (positive value) to bring it back to 0
            offset_x = -current_min_px
        elif current_max_px >= self.width_px:  # Check Right Wall
            # If exceeds width, push left (negative value)
            offset_x = (self.width_px - 1) - current_max_px

        # Calculate corrected target X coordinate
        target_x = self.piece_x_px + offset_x

        if not self.check_collision(target_x, self.piece_y_px, new_shape):
            self.piece_x_px = target_x
            self.current_shape_cells = new_shape

    def shatter_and_lock(self) -> None:
        """
        Key Mechanism: Shattering.

        Writes the rigid body's pixel positions into the grid; from then on, they become independent sand.
        Also checks for Game Over condition (if locked above the screen).
        """
        pixels = self.get_projected_pixels(self.piece_x_px, self.piece_y_px, self.current_shape_cells)
        for x, y in pixels:
            # --- Game Over Check ---
            # If any pixel is still at y < 0 (above screen area) when the block locks
            # It means the stack is too high, and the block cannot fully enter
            if y < 0:
                self.game_over = True
                return  # End, stop writing sand and spawning new blocks

            if 0 <= x < self.width_px and y < self.height_px:
                self.grid[y][x] = self.piece_color

        self._just_shattered = True
        self.check_lines()
        self.spawn_piece()

    def update_sand_physics(self) -> bool:
        """
        Simulates sand physics.

        Iterates through the grid from bottom to top and moves pixels down or diagonally
        if there is empty space.

        Returns:
            bool: True if any sand particle moved, False otherwise.
        """
        changes = False
        # Scan from bottom to top (Bottom-Up)
        for y in range(self.height_px - 2, -1, -1):
            # Randomize X-axis traversal order for more natural diffusion
            x_indices = list(range(self.width_px))
            if random.random() > 0.5:
                x_indices.reverse()

            for x in x_indices:
                pixel = self.grid[y][x]
                if pixel == 0:
                    continue

                # Check directly below
                if self.grid[y + 1][x] == 0:
                    self.grid[y + 1][x] = pixel
                    self.grid[y][x] = 0
                    changes = True
                else:
                    # Check diagonals (Slide)
                    can_left = x > 0 and self.grid[y + 1][x - 1] == 0
                    can_right = x < self.width_px - 1 and self.grid[y + 1][x + 1] == 0

                    if can_left and can_right:
                        direction = random.choice([-1, 1])
                    elif can_left:
                        direction = -1
                    elif can_right:
                        direction = 1
                    else:
                        direction = 0

                    if direction != 0:
                        self.grid[y + 1][x + direction] = pixel
                        self.grid[y][x] = 0
                        changes = True
        return changes

    def check_lines(self) -> bool:
        """
        Detects and clears connected lines of the same color.

        Uses BFS to find connected components of the same color that touch both
        the left and right walls.

        Returns:
            bool: True if any lines were cleared, False otherwise.
        """
        visited = set()
        pixels_to_clear = set()
        cleared_groups = 0

        for y in range(self.height_px):
            for x in range(self.width_px):
                # Skip empty pixels or already processed pixels
                if self.grid[y][x] == 0 or (x, y) in visited:
                    continue

                # --- Start BFS for a new color block ---
                target_color = self.grid[y][x]
                queue = deque([(x, y)])
                visited.add((x, y))

                current_cluster = []  # Record all coordinates of the current block
                touches_left = False
                touches_right = False

                while queue:
                    cx, cy = queue.popleft()
                    current_cluster.append((cx, cy))

                    # Check if touching boundaries
                    if cx == 0:
                        touches_left = True
                    if cx == self.width_px - 1:
                        touches_right = True

                    # Check neighbors in four directions (up, down, left, right)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy

                        # Boundary check
                        if 0 <= nx < self.width_px and 0 <= ny < self.height_px:
                            # Key condition: Not visited AND same color
                            if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

                # --- BFS finished, determine if clearing is needed ---
                if touches_left and touches_right:
                    cleared_groups += 1
                    for px, py in current_cluster:
                        pixels_to_clear.add((px, py))

        # Execute clearing
        if pixels_to_clear:
            # Score calculation: Cleared pixels * Base score + Extra bonus
            points = len(pixels_to_clear)
            self.score += int(points * self.continuous_bonus)

            for px, py in pixels_to_clear:
                self.grid[py][px] = 0  # Set to empty

            # Increase continuous clearing bonus
            self.continuous_bonus *= 1.6

            return True  # Indicates clearing occurred

        return False

    def step(self, action: int) -> None:
        """
        Advances the game state by one tick.

        Handles:
        1. AI decision making (if enabled).
        2. Active piece movement (Left, Right, Rotate).
        3. Gravity for the active piece.
        4. Sand physics simulation.
        5. Line clearing checks.

        Args:
            action (int): The action to perform (see ACTION_* constants).
        """
        if self.game_over:
            return

        # check for AI action
        if action == ACTION_AI:
            # 1. If no plan yet, call external function to compute (only once)
            if self.ai_plan is None:
                self.ai_plan = compute_best_move(self)
            # 2. Execute plan (Override action)
            # Priority: Rotate -> Move Horizontally -> Drop
            if self.ai_plan["rotation_count"] != 0:
                action = ACTION_ROTATE
                self.ai_plan["rotation_count"] -= 1
            else:
                # Handle X-axis movement
                # Allow slight error due to floating point or ppc alignment issues
                diff = self.ai_plan["target_x"] - self.piece_x_px

                if abs(diff) <= self.ppc:  # Aligned
                    # Reached target, execute drop
                    action = ACTION_DOWN
                elif diff > 0:
                    action = ACTION_RIGHT
                elif diff < 0:
                    action = ACTION_LEFT

        # --- 1. Control Active Piece (Rigid Body) ---
        move_dist = self.ppc  # Move by one full cell width horizontally

        if action == ACTION_LEFT:
            if not self.check_collision(
                self.piece_x_px - move_dist, self.piece_y_px, self.current_shape_cells
            ):
                self.piece_x_px -= move_dist
        elif action == ACTION_RIGHT:
            if not self.check_collision(
                self.piece_x_px + move_dist, self.piece_y_px, self.current_shape_cells
            ):
                self.piece_x_px += move_dist
        elif action == ACTION_ROTATE:
            self.rotate_piece()

        # --- 2. Gravity (Active Piece) ---
        # Block drop speed (Pixels per tick)
        if action == ACTION_DOWN:  # soft drop
            drop_speed = self.ppc
            self.score += 5  # soft drop bonus
        elif action == ACTION_DROP:  # hard drop
            drop_speed = self.height_px - self.piece_y_px  # Instant
            self.score += drop_speed  # hard drop bonus
        else:
            drop_speed = self.gravity

        for _ in range(drop_speed):
            if not self.check_collision(self.piece_x_px, self.piece_y_px + 1, self.current_shape_cells):
                self.piece_y_px += 1
            else:
                # Collision detected -> SHATTER
                # Check collision at spawn position for game over
                if self.piece_y_px < 0:
                    self.game_over = True
                    break
                self.shatter_and_lock()
                break

        # --- 3. Physics (Sand) ---
        # Keep background sand flowing
        sand_moved = False
        if self.update_sand_physics():
            sand_moved = True

        # 4. Global Line Check
        # --- Key Modification ---
        # Only run BFS when "sand moved" or "block just shattered" to save performance
        if sand_moved or self._just_shattered:
            self.check_lines()

        self._just_shattered = False

    def get_render_grid(self) -> Grid:
        """
        Generates the current display grid.

        Combines the static sand grid with the active rigid piece.

        Returns:
            Grid: A 2D list representing the game board colors.
        """
        # Copy static grid
        display = [row[:] for row in self.grid]

        # Overlay Active Piece (Rigid)
        pixels = self.get_projected_pixels(self.piece_x_px, self.piece_y_px, self.current_shape_cells)
        for x, y in pixels:
            if 0 <= x < self.width_px and 0 <= y < self.height_px:
                display[y][x] = self.piece_color

        return display

    def get_play_time(self) -> float:
        """Returns the elapsed play time in seconds."""
        return time.time() - self.start_time

    def get_play_time_formatted(self) -> str:
        """Returns the elapsed play time formatted as MM:SS."""
        minutes, seconds = divmod(int(self.get_play_time()), 60)
        return f"{minutes:02}:{seconds:02}"


def _render_grid_to_24bit(grid: Grid, scalar: int = 1) -> List[List[int]]:
    """
    Converts the grid color IDs to 24-bit RGB integers and optionally scales it up.

    Args:
        grid (Grid): The game grid.
        scalar (int): Scaling factor (e.g., 2 means 2x zoom).

    Returns:
        List[List[int]]: A 2D list of 24-bit RGB integers.
    """
    # If scalar <= 1, convert color directly and return
    if scalar <= 1:
        return [[COLOR_TO_24BIT.get(cell, 0) for cell in row] for row in grid]

    # If scalar > 1, perform scaling
    return [
        r
        for row in grid
        for r in itertools.repeat(
            [c for cell in row for c in itertools.repeat(COLOR_TO_24BIT.get(cell, 0), scalar)], scalar
        )
    ]


# --- Interface for LabVIEW ---
def init(cols: int = 10, rows: int = 20, ppc: int = 4, hardness: int = HARDNESS_MEDIUM) -> SandtrisCore:
    return SandtrisCore(cols, rows, ppc, hardness)


def update(game: SandtrisCore, action: int, return_statistics: bool = True) -> Optional["Statistics"]:
    game.step(action)
    if return_statistics:
        return get_statistics(game)
    return None


def get_view(game: SandtrisCore, scalar: int = 1) -> Grid:
    return _render_grid_to_24bit(game.get_render_grid(), scalar)


def get_statistics(game: SandtrisCore) -> "Statistics":
    return (
        game.score,
        game.get_play_time_formatted(),
        SPACE_TO_INDEX_MAPPING[game.next_shape] + (game.next_piece_color - 1) * 7,
        game.game_over,
    )


# If run as main, demo loop (text only)
if __name__ == "__main__":
    s = SandtrisCore(8, 12, 4)  # small board for demo
    ai_mode = input("Enable AI mode? (y/n): ").lower() == "y"
    print("Started demo: press Ctrl+C to quit.")
    for i in itertools.count(0):
        print("=" * 40, "Tick", i, "=" * 40)
        state = s.step(ACTION_NONE)
        if s.game_over:
            print("Game over at tick", i, "score", s.score)
            break
        if ai_mode:
            action = ACTION_AI
            time.sleep(0.2)  # slow down for demo
        else:
            ipt = input("Press Enter to continue, or type 'q' to quit: ").strip()
            if ipt.lower() == "q":
                break
            action = int(ipt) if ipt and ipt.isdigit() else ACTION_NONE
        s.step(action)
        print(
            "\x1b[0m\n".join("".join(f"\x1b[{40 + p}m{p}" for p in row) for row in s.get_render_grid()),
            end="\x1b[0m\n",
        )
        print("Score:", s.score, "Time:", s.get_play_time_formatted())
        print("=" * 38, "Tick", i, "END", "=" * 38)

    print("Demo finished. Score:", s.score)
