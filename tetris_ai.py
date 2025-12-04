import copy
from typing import TYPE_CHECKING, List, Optional, TypedDict

if TYPE_CHECKING:
    from tetris_core import Pixel, SandtrisCore, ShapeCells


class AIPlan(TypedDict):
    rotation_count: int
    target_x: int


# --- 權重設定 (Weights) ---

# 1. 物理勢能策略 (當場上有同色時)
W_OFFSET_X = 2.5  # 水平擴散獎勵 (在範圍內，越遠越高)
W_POTENTIAL_H = 3.5  # 垂直位能獎勵 (越高越好)
W_OVER_RANGE = 5.0  # [新增] 超出範圍懲罰 (係數要比獎勵大，讓分數驟降)

# 參數設定
MAX_FLOW_RANGE = 4  # 最大搜尋半徑
MAX_FLOW_RANGE_PPC = MAX_FLOW_RANGE * 4

# 2. 填坑策略 (當場上無同色時)
W_VALLEY = 6.0  # 填坑獎勵

# 3. 基礎物理限制
W_HEIGHT = -1.0
W_HOLES = -4.0
W_BUMPINESS = -0.5


def compute_best_move(game: "SandtrisCore") -> AIPlan:
    """
    計算最佳落點。
    回傳: (best_rotation_count, best_target_x)
    """
    best_score = -float("inf")
    best_move = (0, game.piece_x_px)

    original_shape = copy.deepcopy(game.current_shape_cells)
    current_test_shape = original_shape

    # 預先掃描目標色
    target_pixels = []
    for x in range(game.width_px):
        for y in range(game.height_px):
            if game.grid[y][x] != 0:
                if game.grid[y][x] == game.piece_color:
                    target_pixels.append((x, y))
                break  # 每列只取最上面一個

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

    # change MAX_FLOW_RANGE by ppc
    global MAX_FLOW_RANGE_PPC
    MAX_FLOW_RANGE_PPC = MAX_FLOW_RANGE * game.ppc

    # 模擬
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

        # 準備下一次旋轉
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
    # 策略分支
    # ==========================================

    if has_target:
        # --- 策略 A: 勢能衝擊 (限制範圍版) ---

        # 1. 水平距離 (Horizontal Offset) - 山峰型計分
        dist_x = abs(current_piece_center_x - target_center_x)

        if dist_x <= MAX_FLOW_RANGE_PPC:
            # 範圍內：距離越遠，分數越高 (鼓勵擴散)
            score += dist_x * W_OFFSET_X
        else:
            # 範圍外：分數驟降
            # 基礎分是 MAX_FLOW_RANGE * W_OFFSET_X (最高點)
            # 扣分項是 (超出距離 * 懲罰係數)
            base_max_score = MAX_FLOW_RANGE_PPC * W_OFFSET_X
            excess_dist = dist_x - MAX_FLOW_RANGE_PPC

            # 這裡我們讓分數快速扣減，甚至可以扣到負分
            penalty = excess_dist * W_OVER_RANGE
            score += base_max_score - penalty

        # 2. 垂直位能 (Vertical Potential)
        diff_h = target_avg_y - current_piece_y
        if diff_h > 0:
            # 優化：如果距離太遠 (超出 MAX_FLOW_RANGE)，垂直位能的獎勵應該打折
            # 因為太遠了即使很高也流不過去
            if dist_x > MAX_FLOW_RANGE_PPC:
                score += diff_h * W_POTENTIAL_H * 0.2  # 遠距離時位能獎勵大幅降低
            else:
                score += diff_h * W_POTENTIAL_H

    else:
        # --- 策略 B: 填坑防禦 ---
        valley_score = 0
        for bx, by in bottom_pixels.items():
            left_solid = (bx == 0) or (game.grid[by][bx - 1] != 0)
            right_solid = (bx == game.width_px - 1) or (game.grid[by][bx + 1] != 0)
            if left_solid and right_solid:
                valley_score += W_VALLEY
        score += valley_score

    # --- 通用指標 ---
    avg_y = sum_y / len(pixels)
    score += avg_y * -(W_HEIGHT)

    bumpiness = _calculate_local_bumpiness(game, pixels)
    score += bumpiness * abs(W_BUMPINESS)

    return score


# --- Helper Functions ---


def _calculate_local_bumpiness(game: "SandtrisCore", pixels: List["Pixel"]) -> int:
    """
    計算局部的平整度懲罰 (負值)
    只檢查受影響的 X 範圍
    """
    xs = [p[0] for p in pixels]
    min_x, max_x = min(xs), max(xs)

    # 擴展檢查範圍多一格，以便比較邊緣
    check_start = max(0, min_x - 1)
    check_end = min(game.width_px - 1, max_x + 1)

    # 取得當前這些列的高度
    col_heights = {}

    # 1. 取得原始高度
    for cx in range(check_start, check_end + 1):
        h = 0
        for cy in range(game.height_px):
            if game.grid[cy][cx] != 0:
                h = game.height_px - cy
                break
        col_heights[cx] = h

    # 2. 模擬加入新方塊後的高度
    for px, py in pixels:
        new_h = game.height_px - py
        if new_h > col_heights[px]:
            col_heights[px] = new_h

    # 3. 計算相鄰差異
    total_bumpiness = 0
    for cx in range(check_start, check_end):  # 兩兩比較
        total_bumpiness += abs(col_heights[cx] - col_heights[cx + 1])

    # 回傳負的分數
    return -total_bumpiness


def _get_shape_pixels_relative(shape_cells: "ShapeCells", ppc: int) -> List["Pixel"]:
    """Cell 座標 -> 相對 Pixel 座標"""
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
    # 從當前實際高度 (或略高) 開始嘗試，以節省時間
    # 為了安全，從 -10 開始搜 (假設方塊不會高過 -10)
    # 或者如果 Core 已經生成在 -10 左右，直接從 game.piece_y_px 開始
    y = game.piece_y_px

    # 防呆：如果 current y 已經撞到了 (這不應該發生在 AI 預判，但防一下)
    if game.check_collision(px_x, y, shape_cells):
        return None

    while True:
        if game.check_collision(px_x, y + 1, shape_cells):
            return y
        y += 1
        if y >= game.height_px:
            return game.height_px
