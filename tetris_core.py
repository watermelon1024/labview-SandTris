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
DEFAULT_PPC = 4  # Pixels Per Cell (每個方塊格由 4x4 像素組成)

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
    HARDNESS_EASY: 4,  # 5 colors
    HARDNESS_MEDIUM: 5,  # 6 colors
    HARDNESS_HARD: 6,  # 7 colors
}

# Colors
# 0 is empty
COLORS = [1, 2, 3, 4, 5, 6, 7]
COLOR_TO_RGB_MAPPING = {
    0: (217, 217, 217),  # Black (Background)
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
    def __init__(
        self,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        ppc: int = DEFAULT_PPC,
        hardness: int = HARDNESS_MEDIUM,
    ):
        self.cols = cols
        self.rows = rows
        self.ppc = ppc
        self.width_px = cols * ppc
        self.height_px = rows * ppc

        # 物理網格 (存放靜止的沙子)
        self.grid: Grid = [[0 for _ in range(self.width_px)] for _ in range(self.height_px)]

        self.score: int = 0
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
        self.hardness = hardness
        COLORS[:] = COLORS[: HARDNESS_COLOR_MAPPING.get(hardness, 5)]

    def generate_next_piece(self) -> None:
        self.next_shape = random.choice(list(SHAPES.keys()))
        self.next_piece_color = random.choice(COLORS)

    def spawn_piece(self) -> None:
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

        # --- Reset AI Plan ---
        self.ai_plan = None

    def get_projected_pixels(self, px_x: int, px_y: int, shape_cells: ShapeCells) -> List[Pixel]:
        """
        將「細胞座標 (Cell)」轉換為真實的「像素座標列表 (Pixel List)」
        這是實現 "碎裂" 的關鍵：把大塊變成小沙粒
        """
        pixels = []
        for cx, cy in shape_cells:
            # 每個 Cell 轉換為 ppc * ppc 個像素
            base_px = px_x + cx * self.ppc
            base_py = px_y + cy * self.ppc

            for i in range(self.ppc):
                for j in range(self.ppc):
                    pixels.append((base_px + i, base_py + j))
        return pixels

    def check_collision(self, px_x: int, px_y: int, shape_cells: ShapeCells) -> bool:
        """
        檢查剛體方塊是否撞到邊界或現有的沙子
        """
        # 優化：只檢查每個 Cell 的邊界像素，或者檢查所有轉換後的像素
        # 為了準確，我們檢查所有投影像素
        pixels = self.get_projected_pixels(px_x, px_y, shape_cells)

        for x, y in pixels:
            # Wall / Floor Collision
            if x < 0 or x >= self.width_px or y >= self.height_px:
                return True

            # Sand Collision (只有當 y >= 0 時才檢查網格，避免生成時報錯)
            if y >= 0 and self.grid[y][x] != 0:
                return True

        return False

    def rotate_piece(self) -> None:
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

        # Wall Kick (牆壁推擠修正)
        # 計算新形狀在當前位置的絕對像素邊界
        # 由於每個 cell 寬度是 ppc，我们需要算最左邊和最右邊的像素點
        # min_cell_x 肯定是 0 (因為上面正規化了)，所以最左邊是 self.piece_x_px
        # 我們只需要算最右邊會不會超出去
        max_cell_x = max(p[0] for p in new_shape)
        # 預測的最左與最右像素 X 座標
        current_min_px = self.piece_x_px
        current_max_px = self.piece_x_px + (max_cell_x * self.ppc) + (self.ppc - 1)

        offset_x = 0

        if current_min_px < 0:  # 檢查左牆 (Left Wall)
            # 如果小於 0，就往右推 (正值) 把它推回 0
            offset_x = -current_min_px
        elif current_max_px >= self.width_px:  # 檢查右牆 (Right Wall)
            # 如果超出寬度，就往左推 (負值)
            offset_x = (self.width_px - 1) - current_max_px

        # 計算修正後的目標 X 座標
        target_x = self.piece_x_px + offset_x

        if not self.check_collision(target_x, self.piece_y_px, new_shape):
            self.piece_x_px = target_x
            self.current_shape_cells = new_shape

    def shatter_and_lock(self) -> None:
        """
        關鍵機制：碎裂
        將剛體的像素位置寫入 grid，從此之後它們變成獨立的沙子。
        """
        pixels = self.get_projected_pixels(self.piece_x_px, self.piece_y_px, self.current_shape_cells)
        for x, y in pixels:
            # --- Game Over Check ---
            # 如果方塊撞擊鎖定時，有任何一個像素點還在 y < 0 (螢幕上方區域)
            # 代表堆疊過高，方塊無法完全進入
            if y < 0:
                self.game_over = True
                return  # 結束，不再寫入沙子，也不再生成新方塊

            if 0 <= x < self.width_px and y < self.height_px:
                self.grid[y][x] = self.piece_color

        self._just_shattered = True
        self.check_lines()
        self.spawn_piece()

    def update_sand_physics(self) -> bool:
        """
        沙子物理：只針對 grid 裡的像素運算
        """
        changes = False
        # 從下往上掃描 (Bottom-Up)
        for y in range(self.height_px - 2, -1, -1):
            # 隨機化 X 軸遍歷順序，讓擴散更自然
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
        BFS 消除檢測：
        遍歷網格，尋找所有「同色」且「同時接觸左牆與右牆」的連通區塊並消除。
        """
        visited = set()
        pixels_to_clear = set()
        cleared_groups = 0

        for y in range(self.height_px):
            for x in range(self.width_px):
                # 跳過空像素或已處理過的像素
                if self.grid[y][x] == 0 or (x, y) in visited:
                    continue

                # --- 開始一個新的顏色區塊 BFS ---
                target_color = self.grid[y][x]
                queue = deque([(x, y)])
                visited.add((x, y))

                current_cluster = []  # 記錄當前區塊的所有座標
                touches_left = False
                touches_right = False

                while queue:
                    cx, cy = queue.popleft()
                    current_cluster.append((cx, cy))

                    # 檢查是否觸碰邊界
                    if cx == 0:
                        touches_left = True
                    if cx == self.width_px - 1:
                        touches_right = True

                    # 檢查四個方向的鄰居 (上下左右)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy

                        # 邊界檢查
                        if 0 <= nx < self.width_px and 0 <= ny < self.height_px:
                            # 關鍵條件：未訪問過 且 顏色相同
                            if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

                # --- BFS 結束，判斷是否消除 ---
                if touches_left and touches_right:
                    cleared_groups += 1
                    for px, py in current_cluster:
                        pixels_to_clear.add((px, py))

        # 執行消除
        if pixels_to_clear:
            # 分數計算：消除像素數 * 基礎分 + 額外獎勵
            points = len(pixels_to_clear)
            self.score += points + (cleared_groups * 100)

            for px, py in pixels_to_clear:
                self.grid[py][px] = 0  # 設為空

            return True  # 代表有消除發生

        return False

    def step(self, action: int) -> None:
        if self.game_over:
            return

        # check for AI action
        if action == ACTION_AI:
            # 1. 如果還沒有計畫，呼叫外部函數計算 (只算一次)
            if self.ai_plan is None:
                self.ai_plan = compute_best_move(self)
            # 2. 執行計畫 (Override action)
            # 優先級：旋轉 -> 橫移 -> 下落
            if self.ai_plan["rotation_count"] != 0:
                action = ACTION_ROTATE
                self.ai_plan["rotation_count"] -= 1
            else:
                # 處理 X 軸移動
                # 容許一點誤差，因為浮點數或 ppc 對齊問題
                diff = self.ai_plan["target_x"] - self.piece_x_px

                if abs(diff) <= self.ppc:  # 已經對齊
                    # 到達目標，執行降落
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
        # 方塊下落速度 (Pixels per tick)
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
        # 讓背景的沙子持續流動
        sand_moved = False
        if self.update_sand_physics():
            sand_moved = True

        # 4. 全局消除檢測 (Global Line Check)
        # --- 關鍵修改 ---
        # 只有當「沙子移動了」或者「剛有方塊碎裂」時，才耗費效能去跑 BFS
        if sand_moved or self._just_shattered:
            self.check_lines()

        self._just_shattered = False

    def get_render_grid(self) -> Grid:
        # Copy static grid
        display = [row[:] for row in self.grid]

        # Overlay Active Piece (Rigid)
        pixels = self.get_projected_pixels(self.piece_x_px, self.piece_y_px, self.current_shape_cells)
        for x, y in pixels:
            if 0 <= x < self.width_px and 0 <= y < self.height_px:
                display[y][x] = self.piece_color

        return display

    def get_play_time(self) -> float:
        return time.time() - self.start_time

    def get_play_time_formatted(self) -> str:
        minutes, seconds = divmod(int(self.get_play_time()), 60)
        return f"{minutes:02}:{seconds:02}"


def _render_grid_to_24bit(grid: Grid, scalar: int = 1) -> List[List[int]]:
    # 如果 scalar <= 1，直接轉換顏色並回傳
    if scalar <= 1:
        return [[COLOR_TO_24BIT.get(cell, 0) for cell in row] for row in grid]

    # 如果 scalar > 1，進行放大處理
    # 使用 List Comprehension 提升效能，同時完成水平與垂直放大
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
            time.sleep(0.25)  # slow down for demo
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
