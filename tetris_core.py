from collections import deque
import random

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

# Colors (對應 LabVIEW Intensity Graph Z-Scale)
# 0 is empty
COLORS = [1, 2, 3, 4, 5, 6, 7]

# Tetromino Shapes (Defined in Cells)
SHAPES = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
}


class SandtrisCore:
    def __init__(self, cols=DEFAULT_COLS, rows=DEFAULT_ROWS, ppc=DEFAULT_PPC):
        self.cols = cols
        self.rows = rows
        self.ppc = ppc
        self.width_px = cols * ppc
        self.height_px = rows * ppc

        # 物理網格 (存放靜止的沙子)
        self.grid = [[0 for _ in range(self.width_px)] for _ in range(self.height_px)]

        self.score = 0
        self.game_over = False
        self._just_shattered = False

        # Active Piece (Rigid Body State)
        self.current_shape_cells = []  # List of (x, y) in CELL coordinates relative to piece center
        self.piece_x_px = 0  # Top-left X in PIXELS
        self.piece_y_px = 0  # Top-left Y in PIXELS
        self.piece_color = 1

        self.spawn_piece()

    def spawn_piece(self):
        shape_keys = list(SHAPES.keys())
        key = shape_keys[random.randint(0, len(shape_keys) - 1)]
        self.current_shape_cells = list(SHAPES[key])  # Copy
        self.piece_color = COLORS[random.randint(0, len(COLORS) - 1)]

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

    def get_projected_pixels(self, px_x, px_y, shape_cells):
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

    def check_collision(self, px_x, px_y, shape_cells):
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

    def rotate_piece(self):
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

        if not self.check_collision(self.piece_x_px, self.piece_y_px, new_shape):
            self.current_shape_cells = new_shape

    def shatter_and_lock(self):
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

    def update_sand_physics(self):
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

    def check_lines(self):
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

    def step(self, action):
        if self.game_over:
            return

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
        drop_speed = self.ppc if action == ACTION_DOWN else 1
        if action == ACTION_DROP:
            drop_speed = self.height_px  # Instant

        for _ in range(drop_speed):
            if not self.check_collision(self.piece_x_px, self.piece_y_px + 1, self.current_shape_cells):
                self.piece_y_px += 1
            else:
                # Collision detected -> SHATTER
                # Check collision at spawn position for game over
                if self.check_collision(self.piece_x_px, self.piece_y_px, self.current_shape_cells):
                    self.game_over = True
                    break
                self.shatter_and_lock()
                break

        # --- 3. Physics (Sand) ---
        # 讓背景的沙子持續流動
        sand_moved = False
        for _ in range(2):  # Run physics twice per frame for speed
            if self.update_sand_physics():
                sand_moved = True

        # 4. 全局消除檢測 (Global Line Check)
        # --- 關鍵修改 ---
        # 只有當「沙子移動了」或者「剛有方塊碎裂」時，才耗費效能去跑 BFS
        if sand_moved or self._just_shattered:
            self.check_lines()

        self._just_shattered = False

    def get_render_grid(self):
        # Copy static grid
        display = [row[:] for row in self.grid]

        # Overlay Active Piece (Rigid)
        if not self.game_over:
            pixels = self.get_projected_pixels(self.piece_x_px, self.piece_y_px, self.current_shape_cells)
            for x, y in pixels:
                if 0 <= x < self.width_px and 0 <= y < self.height_px:
                    display[y][x] = self.piece_color
        return display


# --- Interface for LabVIEW ---
game = None


def init(cols=10, rows=20, ppc=4):
    global game
    game = SandtrisCore(cols, rows, ppc)
    return "Ready"


def update(action):
    global game
    if game:
        game.step(action)
        return game.score, game.game_over
    return 0, True


def get_view():
    global game
    if game:
        return game.get_render_grid()
    return []


# If run as main, demo loop (text only)
if __name__ == "__main__":
    import itertools

    s = SandtrisCore(8, 10, 4)  # small board for demo

    print("Started demo: press Ctrl+C to quit.")
    # spawn a few pieces programmatically
    for i in itertools.count(0):
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
        print("Score:", s.score)
        print(
            "\x1b[0m\n".join("".join(f"\x1b[{40 + p}m{p}" for p in row) for row in s.get_render_grid()),
            end="\x1b[0m\n",
        )
        print("=" * 38, "Tick", i, "END", "=" * 38)

    print("Demo finished. Score:", s.score)
