"""
tetris_core.py

Sandtris-style Tetris core (light prototype)

API:
    init_game(cols=10, rows=20, pixels_per_cell=6, seed=None)
    step(action) -> returns (pixel_grid, score_delta, game_over)
    spawn_piece(piece_id=None)
    get_state() -> dict { 'pixel_grid': ..., 'score': ..., 'game_over': ... }

Actions (int or str):
    0 / 'none'
    1 / 'left'
    2 / 'right'
    3 / 'rotate'
    4 / 'soft_drop'
    5 / 'hard_drop'
    6 / 'ai'  (optional)
"""

import itertools
import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

# Action constants
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3  # not implemented yet
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5

# Piece definitions in cell coordinates (4x4 prototypes)
# Each piece: list of rotation states; each state = list of (x,y) cells (0-based)
Pos = Tuple[int, int]
Pixel = Tuple[int, int]
PixelWithColor = Tuple[int, int, int]

PIECES_TXT = {
    "I": [
        "#",
        "#",
        "#",
        "#",
    ],
    "J": [
        " #",
        " #",
        "##",
    ],
    "L": [
        "# ",
        "# ",
        "##",
    ],
    "O": [
        "##",
        "##",
    ],
    "S": [
        " ##",
        "## ",
    ],
    "T": [
        " # ",
        "###",
    ],
    "Z": [
        "## ",
        " ##",
    ],
}

# build piece to position from text
PIECES: Dict[str, List[Pos]] = {
    k: sorted(
        itertools.chain.from_iterable(
            (itertools.chain((j, i) for j, c in enumerate(s) if c == "#")) for i, s in enumerate(v)
        )
    )
    for k, v in PIECES_TXT.items()
}
print("Loaded pieces:", PIECES)

# map piece -> color id (int)
PIECE_COLOR = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}


class Piece:
    def __init__(self, shape: str, rotation: int = 0) -> None:
        self.shape = shape
        self.rotation = rotation
        self.cells = PIECES[shape]
        self.color = PIECE_COLOR[shape]
        self.width = max(p[0] for p in self.cells) + 1
        self.height = max(p[1] for p in self.cells) + 1


class Particle:
    __slots__ = ("x", "y", "color", "stable", "stuck_ticks")

    def __init__(self, x: int, y: int, color: int) -> None:
        self.x: int = x  # pixel coord, 0..w-1
        self.y: int = y  # pixel coord, 0..h-1
        self.color: int = color
        self.stable: bool = False  # true when it hasn't moved for settle_threshold ticks
        self.stuck_ticks: int = 0


class SandTetrisCore:
    def __init__(
        self,
        cols: int = 10,
        rows: int = 20,
        pixels_per_cell: int = 6,
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        cols, rows : logical tetris cells (standard: 10x20)
        pixels_per_cell : number of pixels per cell side (square)
        """
        self.cols: int = cols
        self.rows: int = rows
        self.ppc: int = int(pixels_per_cell)
        """ pixels per cell """
        self.pw: int = cols * self.ppc
        """ pixel grid width """
        self.ph: int = rows * self.ppc
        """ pixel grid height """
        self.score: int = 0
        self.game_over: bool = False

        self.pixel_grid: List[List[int]] = [[0 for _ in range(self.pw)] for _ in range(self.ph)]
        # settled pixels are recorded in pixel_grid (0 empty, >0 color id)
        # moving particles are stored separately in particles list
        self.particles: List[Particle] = []
        self.settle_threshold: int = 3
        # ticks of no -movement before considered settled (writes to pixel_grid)

        # active piece
        self.active_piece: Optional[Piece] = None
        self.active_piece_pos: Pos = (0, 0)  # top-left pos of piece bounding box
        self.next_bag: List[str] = []
        self.tick_count: int = 0
        self.gravity_ticks: int = 1  # how many frames before auto-drop (can be tuned)
        self.drop_accumulator: int = 0

        if rng_seed is not None:
            random.seed(rng_seed)

        self._init_bag()
        self.spawn_piece()

    def _init_bag(self) -> None:
        # simple 7-bag
        bag = list(PIECES.keys())
        random.shuffle(bag)
        self.next_bag = bag

    def _pop_bag(self) -> str:
        return random.choice(list(PIECES.keys()))
        if not self.next_bag:
            self._init_bag()
        return self.next_bag.pop()

    # -------------------------
    # Active piece handling
    # -------------------------
    def spawn_piece(self, piece_id: Optional[str] = None) -> None:
        pid = piece_id or self._pop_bag()
        # pid = "I"  # for testing always I --- IGNORE ---
        self.active_piece = Piece(pid)
        # place near top center (in cell coords)
        # compute bounding width
        shape = PIECES[pid]
        # shape = states[self.active_piece.rotation]
        minx = min(p[0] for p in shape)
        maxx = max(p[0] for p in shape)
        h = max(p[1] for p in shape) + 1
        width = maxx - minx + 1
        start_x = (self.pw - width * self.ppc) // 2
        self.active_piece_pos = (start_x, -h * self.ppc)  # start above visible area
        # if spawn collides with existing settled cells (converted pixel grid collapsed), then game over
        # if self._is_piece_collides(self.active_piece_pos, self.active_piece.rotation):
        #     self.game_over = True

    def _get_piece_pixels(
        self, piece: Optional[Piece] = None, pos: Optional[Pos] = None, rot: Optional[int] = None
    ) -> List[Particle]:
        "Get list of occupied pixels for the piece at given pos & rotation"
        piece = piece or self.active_piece
        if piece is None:
            return []
        pos = pos or self.active_piece_pos
        shape = piece.cells
        return [
            Particle(pos[0] + x * self.ppc + i, pos[1] + y * self.ppc + j, piece.color)
            for x, y in shape
            for i in range(self.ppc)
            for j in range(self.ppc)
        ]

    def _get_piece_bottom_pixels(
        self, piece: Optional[Piece] = None, pos: Optional[Pos] = None, rot: Optional[int] = None
    ) -> List[Pos]:
        "Get list of bottom-most occupied pixels for the piece at given pos & rotation"
        piece = piece or self.active_piece
        if piece is None:
            return []
        pos = pos or self.active_piece_pos
        shape = piece.cells
        px = []
        for x, ps in itertools.groupby(shape, key=lambda p: p[0]):
            my = max(p[1] for p in ps) + 1
            px.extend((pos[0] + x + i, pos[1] + my * self.ppc) for i in range(self.ppc))
        return px

    # def _cell_occupied_by_settled(self, cx: int, cy: int) -> bool:
    #     # check if any pixel in this cell is nonzero => considered occupied
    #     if cx < 0 or cx >= self.cols or cy < 0 or cy >= self.rows:
    #         return True  # out-of-bounds considered occupied
    #     ox, oy = self.cell_to_pixel_origin(cx, cy)
    #     for py in range(oy, oy + self.ppc):
    #         for px in range(ox, ox + self.ppc):
    #             if self.pixel_grid[py][px] != 0:
    #                 return True
    #     return False

    def _is_piece_collides(self, pos: Pos, rot: Optional[int] = None) -> bool:
        return any(
            x < 0 or x >= self.pw or y >= self.ph or self.pixel_grid[y][x]
            for x, y in self._get_piece_bottom_pixels(self.active_piece, pos, rot)
        )
        for x, y in self._get_piece_bottom_pixels(self.active_piece, pos, rot):
            # if x < 0 or x >= self.pw or y < 0 or y >= self.ph:
            #     return True
            if self.pixel_grid[y][x] != 0:
                return True
        return False

    def _apply_action_to_active(self, action: Union[int, str]) -> None:
        if self.active_piece is None:
            return  # nothing
        if action in (ACTION_LEFT, "left"):
            new_pos = (self.active_piece_pos[0] - 1, self.active_piece_pos[1])
            if not self._is_piece_collides(new_pos, self.active_piece.rotation):
                self.active_piece_pos = new_pos
        elif action in (ACTION_RIGHT, "right"):
            new_pos = (self.active_piece_pos[0] + 1, self.active_piece_pos[1])
            if not self._is_piece_collides(new_pos, self.active_piece.rotation):
                self.active_piece_pos = new_pos
        # elif action in (ACTION_ROTATE, "rotate"):
        #     new_rot = (self.active_piece.rotation + 1) % len(PIECES[self.active_piece])
        #     if not self._piece_collides(self.active_piece_pos, new_rot):
        #         self.active_piece.rotation = new_rot
        elif action in (ACTION_SOFT_DROP, "soft_drop"):
            new_pos = (self.active_piece_pos[0], self.active_piece_pos[1] + 1)
            if not self._is_piece_collides(new_pos, self.active_piece.rotation):
                self.active_piece_pos = new_pos
            else:
                # cannot move down -> convert to particles
                self._break_into_particle()
                print("break1")
        elif action in (ACTION_HARD_DROP, "hard_drop"):
            # move down until collision
            while True:
                new_pos = (self.active_piece_pos[0], self.active_piece_pos[1] + 1)
                if self._is_piece_collides(new_pos, self.active_piece.rotation):
                    self._break_into_particle()
                    print("break2")
                    break
                self.active_piece_pos = new_pos
        else:
            # none or unknown
            return

    # -------------------------
    # generate particles in pixel layer
    # -------------------------
    def _break_into_particle(self) -> None:
        if self.active_piece is None:
            return
        px = self._get_piece_pixels()
        print([(p.x, p.y) for p in px])
        self.particles.extend(px)
        # # for each cell, generate sub-particles inside the cell area
        # for x, y in self._get_piece_pixels():
        #     # if out of bounds ignore
        #     if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
        #         continue
        #     ox, oy = self.cell_to_pixel_origin(x, y)
        #     # generate micro-particles: fill all pixels in the cell with particles
        #     for py in range(oy, oy + self.ppc):
        #         for px in range(ox, ox + self.ppc):
        #             # if pixel already occupied in pixel_grid, skip and instead make it a moving particle above if possible
        #             if self.pixel_grid[py][px] == 0:
        #                 p = Particle(px, py, color)
        #                 self.particles.append(p)
        #             else:
        #                 # there is already settled pixel; try to create particle above cell area if possible
        #                 # find first empty pixel above within the board
        #                 ay = py - 1
        #                 while ay >= 0 and self.pixel_grid[ay][px] != 0:
        #                     ay -= 1
        #                 if ay >= 0:
        #                     self.particles.append(Particle(px, ay, color))
        #                 # else cannot place this micro-particle (stack full)
        # clear active piece
        self.active_piece = None
        # spawn next piece
        self.spawn_piece()

    # -------------------------
    # Particle simulation (pixel-level)
    # Basic sand rules:
    #   if below empty -> move down
    #   elif down-left empty -> move down-left
    #   elif down-right empty -> move down-right
    #   else -> stuck (increment stuck_ticks)
    # If stuck_ticks >= settle_threshold -> write to pixel_grid (settled)
    # -------------------------
    def simulate_particles(self) -> List[PixelWithColor]:
        if not self.particles:
            return []
        # # shuffle order to reduce bias
        # random.shuffle(self.particles)
        self.particles.sort(key=lambda p: p.y, reverse=True)  # process from bottom to top
        moved_particles = []  # list of particles that changed final position (for local BFS start)
        # occupancy map for intended moves in this tick to avoid collisions: (x,y) -> True
        intended = set()
        # settled occupancy is pixel_grid (nonzero) plus planned moves
        for p in self.particles:
            px, py = p.x, p.y
            moved = False
            # try down
            ny = py + 1
            if ny < self.ph and self._pixel_empty(px, ny) and (px, ny) not in intended:
                p.y = ny
                moved = True
            else:
                # try down-left
                nx = px - 1
                ny = py + 1
                if nx >= 0 and ny < self.ph and self._pixel_empty(nx, ny) and (nx, ny) not in intended:
                    p.x = nx
                    p.y = ny
                    moved = True
                else:
                    # try down-right
                    nx = px + 1
                    ny = py + 1
                    if (
                        nx < self.pw
                        and ny < self.ph
                        and self._pixel_empty(nx, ny)
                        and (nx, ny) not in intended
                    ):
                        p.x = nx
                        p.y = ny
                        moved = True
            if moved:
                p.stuck_ticks = 0
                intended.add((p.x, p.y))
                moved_particles.append(p)
            else:
                # cannot move
                p.stuck_ticks += 1
                intended.add((p.x, p.y))
        # settle particles whose stuck_ticks >= threshold
        newly_settled_positions: List[PixelWithColor] = []
        remaining_particles: List[Particle] = []
        for p in self.particles:
            if p.stuck_ticks >= self.settle_threshold:
                # write to pixel grid if empty
                if self.pixel_grid[p.y][p.x] == 0:
                    self.pixel_grid[p.y][p.x] = p.color
                    newly_settled_positions.append((p.x, p.y, p.color))
            else:
                remaining_particles.append(p)
        self.particles = remaining_particles
        return newly_settled_positions  # pixel coords where new settled pixels were written

    def _pixel_empty(self, px: int, py: int) -> bool:
        # empty if in bounds and pixel_grid==0 and there is no moving particle currently at that pos
        if px < 0 or px >= self.pw or py < 0 or py >= self.ph:
            return False
        if self.pixel_grid[py][px] != 0:
            return False
        # also check moving particles list to avoid overlap
        for p in self.particles:
            if p.x == px and p.y == py:
                return False
        return True

    # -------------------------
    # Local BFS for color-connected starting from a given pixel
    # Returns (cells_list, touches_left, touches_right)
    # -------------------------
    def local_color_bfs(self, start_x: int, start_y: int, color: int) -> Tuple[List[Pixel], bool, bool]:
        if not (0 <= start_x < self.pw and 0 <= start_y < self.ph):
            return [], False, False
        # if start pixel not matching color, bail
        if self.pixel_grid[start_y][start_x] != color:
            return [], False, False
        q: deque[Pixel] = deque()
        q.append((start_x, start_y))
        visited: Set[Pixel] = set()
        visited.add((start_x, start_y))
        touches_left = False
        touches_right = False
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            if x == 0:
                touches_left = True
            if x == self.pw - 1:
                touches_right = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.pw and 0 <= ny < self.ph and (nx, ny) not in visited:
                    if self.pixel_grid[ny][nx] == color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return list(visited), touches_left, touches_right

    # -------------------------
    # Clear a set of pixels (list of (x,y))
    # -------------------------
    def clear_pixels(self, pixel_list: Iterable[Pixel]) -> int:
        removed = 0
        for x, y in pixel_list:
            if 0 <= x < self.pw and 0 <= y < self.ph:
                if self.pixel_grid[y][x] != 0:
                    self.pixel_grid[y][x] = 0
                    removed += 1
        return removed

    # -------------------------
    # After clears, we can optionally 'unjam' the board by turning settled pixels above empties into moving particles,
    # so that gravity/flow continues as sand (preserve lateral sliding).
    # We'll scan columns and for any settled pixel that has empty below, convert it to a moving particle.
    # -------------------------
    def remobilize_floating_pixels(self) -> int:
        new_particles: List[Particle] = []
        for y in range(self.ph - 1, -1, -1):
            for x in range(self.pw):
                if self.pixel_grid[y][x] != 0:
                    # if below empty then this pixel should become mobile
                    if y + 1 < self.ph and self.pixel_grid[y + 1][x] == 0:
                        color = self.pixel_grid[y][x]
                        self.pixel_grid[y][x] = 0
                        new_particles.append(Particle(x, y, color))
        self.particles.extend(new_particles)
        return len(new_particles)

    # -------------------------
    # Collapsing helper (optional fallback)
    # If you want to instantly collapse columns (less natural), you can use this.
    # Not used by default.
    # -------------------------
    def collapse_columns_instant(self) -> None:
        for x in range(self.pw):
            stack = []
            for y in range(self.ph - 1, -1, -1):
                if self.pixel_grid[y][x] != 0:
                    stack.append(self.pixel_grid[y][x])
            # refill
            y = self.ph - 1
            for val in stack:
                self.pixel_grid[y][x] = val
                y -= 1
            for yy in range(y, -1, -1):
                self.pixel_grid[yy][x] = 0

    # -------------------------
    # Main tick update. Accepts an action for the active piece.
    # Returns: (pixel_grid copy, score_delta, game_over)
    # -------------------------
    def step(self, action: Union[int, str] = ACTION_NONE) -> Dict[str, object]:
        """
        Process one frame:
          - apply action to active piece (cell-level)
          - gravity tick may advance active piece down (simulate natural drop)
          - simulate pixel particles (several sub-steps for smoothness)
          - when new settled pixels are created, run local BFS from each to check left/right contact
          - if a connected region touches both left & right, clear it and update score
          - after clearing, remobilize floating pixels to continue physics
        """
        if self.game_over:
            return self.get_state()

        # 1. apply user/AI action to active piece
        self._apply_action_to_active(action)

        # 2. gravity auto-drop for active piece (basic)
        self.drop_accumulator += 1
        if self.active_piece is not None and self.drop_accumulator >= self.gravity_ticks:
            self.drop_accumulator = 0
            # try move down
            new_pos = (
                self.active_piece_pos[0],
                min(
                    self.active_piece_pos[1] + self.ppc // 2,
                    self.ph - (self.active_piece.height - 1) * self.ppc - 1,
                ),
            )
            if not self._is_piece_collides(new_pos, self.active_piece.rotation):
                print("gravity move down")
                self.active_piece_pos = new_pos
            else:
                # lock piece
                self._break_into_particle()
                print("break3")

        # 3. simulate particles for some steps per tick (tunable)
        # If we want smoother sand, run simulate_particles multiple times per step.
        sim_steps = 2
        total_removed = 0
        newly_settled = []
        for _ in range(sim_steps):
            newly_settled = self.simulate_particles()

        # 4. for each newly settled pixel, do local BFS (only once per connected cluster)
        # We will group BFS by color and avoid duplicate BFS on same pixel by visited set.
        visited_global = set()
        clusters_cleared = 0
        pixels_removed_total = 0

        for px, py, color in newly_settled:
            if (px, py) in visited_global:
                continue
            # BFS from this pixel
            cluster_pixels, touches_left, touches_right = self.local_color_bfs(px, py, color)
            for pxy in cluster_pixels:
                visited_global.add(pxy)
            if touches_left and touches_right:
                # clear cluster
                removed = self.clear_pixels(cluster_pixels)
                if removed > 0:
                    clusters_cleared += 1
                    pixels_removed_total += removed
                    # after removal, we will remobilize floating pixels later

        # 5. after all clears, if any cleared -> remobilize floating pixels so sand continues to flow
        if pixels_removed_total > 0:
            self.remobilize_floating_pixels()
            # score: simple mapping (1 point per removed pixel)
            self.score += pixels_removed_total

        # 6. check game_over: if any pixel occupies top rows above spawn area
        # We'll define game over if any settled pixel in top two cell rows exists
        top_rows_cutoff = self.ppc * 2  # top two cell rows in pixels
        for y in range(0, top_rows_cutoff):
            for x in range(self.pw):
                if self.pixel_grid[y][x] != 0:
                    self.game_over = True
                    print("Game over detected due to settled pixels at top.", (x, y))
                    break
            if self.game_over:
                break

        self.tick_count += 1

        return self.get_state()

    def get_state(self) -> Dict[str, object]:
        # return shallow copy safe representation (deep copy pixel grid may be large)
        grid_copy = [row[:] for row in self.pixel_grid]
        for p in self.particles:
            grid_copy[p.y][p.x] = p.color  # overlay moving particles for visualization
        for p in self._get_piece_pixels():
            if p.y >= 0:
                grid_copy[p.y][p.x] = p.color  # overlay active piece for visualization
        # include moving particles for rendering as list of tuples
        particles_copy = [(p.x, p.y, p.color) for p in self.particles]
        return {
            "pixel_grid": grid_copy,
            "particles": particles_copy,
            "active_piece": (
                None
                if self.active_piece is None
                else {
                    "type": self.active_piece,
                    "pos": self.active_piece_pos,
                    "rotation": self.active_piece.rotation,
                }
            ),
            "score": self.score,
            "game_over": self.game_over,
            "pw": self.pw,
            "ph": self.ph,
            "cols": self.cols,
            "rows": self.rows,
            "ppc": self.ppc,
        }


# Convenience wrapper functions for easy external use
_core_singleton: Optional[SandTetrisCore] = None


def init_game(
    cols: int = 10, rows: int = 20, pixels_per_cell: int = 6, seed: Optional[int] = None
) -> Dict[str, object]:
    global _core_singleton
    _core_singleton = SandTetrisCore(cols, rows, pixels_per_cell, rng_seed=seed)
    return _core_singleton.get_state()


def step(action: Union[int, str] = ACTION_NONE) -> Dict[str, object]:
    global _core_singleton
    if _core_singleton is None:
        raise RuntimeError("Game not initialized. Call init_game() first.")
    return _core_singleton.step(action)


def spawn_piece(piece_id: Optional[str] = None) -> Dict[str, object]:
    global _core_singleton
    if _core_singleton is None:
        raise RuntimeError("Game not initialized. Call init_game() first.")
    _core_singleton.spawn_piece(piece_id)
    return _core_singleton.get_state()


def get_state() -> Optional[Dict[str, object]]:
    global _core_singleton
    if _core_singleton is None:
        return None
    return _core_singleton.get_state()


# If run as main, demo loop (text only)
if __name__ == "__main__":
    s = SandTetrisCore(8, 10, 4)  # small board for demo
    import time

    print("Started demo: press Ctrl+C to quit.")
    # spawn a few pieces programmatically
    for i in range(1000):
        print("=" * 40, "Tick", i, "=" * 40)
        state = s.step(ACTION_NONE)
        # every 10 ticks, soft drop random
        # if i % 8 == 0:
        #     s.step(ACTION_SOFT_DROP)
        if s.game_over:
            print("Game over at tick", i, "score", s.score)
            break

        ipt = input("Press Enter to continue, or type 'q' to quit: ")
        if ipt.lower() == "q":
            break
        s.step(int(ipt) if ipt.isdigit() else ACTION_NONE)
        stat = s.get_state()
        print(stat["particles"], stat["active_piece"])
        print("Score:", stat["score"])
        print(
            "\x1b[0m\n".join("".join(f"\x1b[{40 + p}m{p}" for p in row) for row in stat["pixel_grid"]),  # type: ignore
            end="\x1b[0m\n",
        )
        print("=" * 40, "Tick", i, "END", "=" * 40)

        time.sleep(0.1)
    print("Demo finished. Score:", s.score)
