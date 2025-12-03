import copy
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from tetris_core import SandtrisCore


class AIPlan(TypedDict):
    rotation_count: int
    target_x: int


# --- 權重設定 (Weights) ---
W_COLOR_MATCH = 15.0  # 同色獎勵 (大幅提高，這是 Sandtris 的核心)
W_WALL = 2.0  # 靠牆獎勵 (稍微提高，鼓勵填邊)
W_HEIGHT = -1.0  # 高度權重 (負值，配合下面邏輯：越高分越低)
W_HOLES = -4.0  # 空洞懲罰 (剛體下方的空隙)
W_BUMPINESS = -0.5  # 平整度權重 (保留項目，負值代表越不平越扣分)


def compute_best_move(game: "SandtrisCore") -> AIPlan:
    """
    計算最佳落點。
    回傳: (best_rotation_count, best_target_x)
    """
    best_score = -float("inf")
    best_move = (0, game.piece_x_px)

    original_shape = copy.deepcopy(game.current_shape_cells)
    current_test_shape = original_shape

    # 嘗試 4 種旋轉
    for rot_idx in range(4):
        # 1. 計算該旋轉狀態下的寬度範圍
        pixels_shape = _get_shape_pixels_relative(current_test_shape, game.ppc)
        min_x = min(p[0] for p in pixels_shape)
        max_x = max(p[0] for p in pixels_shape)

        # 2. 遍歷可能的 X 座標
        # 我們以 ppc 為單位進行步進，模擬玩家的移動
        start_x = -min_x
        end_x = game.width_px - 1 - max_x

        # 保持與當前位置相同的 ppc 模數對齊
        base_mod = game.piece_x_px % game.ppc

        # 建立搜尋範圍
        valid_x_range = range(start_x, end_x + 1, game.ppc)

        for tx in valid_x_range:
            if (tx % game.ppc) != base_mod:
                continue

            # 3. 模擬 Hard Drop (尋找著陸的 Y)
            land_y = _simulate_drop_y(game, tx, current_test_shape)

            if land_y is None:
                continue  # 無法放置

            # 4. 評估分數 (整合你的 evaluate_position)
            score = evaluate_position(game, tx, land_y, current_test_shape)

            if score > best_score:
                best_score = score
                best_move = (rot_idx, tx)

        # 準備下一次旋轉
        current_test_shape = _rotate_shape_simulate(current_test_shape)

    return {"rotation_count": best_move[0], "target_x": best_move[1]}


def evaluate_position(game: "SandtrisCore", x, y, shape_cells):
    """
    評分函數 (整合版)
    """
    score = 0
    # 取得投影後的絕對像素座標
    pixels = _get_projected_pixels(x, y, shape_cells, game.ppc)

    sum_y = 0
    piece_color = game.piece_color

    # 建立一個 set 用於快速查詢「自身像素」，避免把自己的其他部分當成鄰居
    my_pixels_set = set(pixels)

    for px, py in pixels:
        # Game Over 預判：如果落點有任何部分在螢幕外，給予極大懲罰
        if py < 0:
            return -999999

        sum_y += py

        # --- 1. Color Match & Wall Check ---
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            nx, ny = px + dx, py + dy

            # 邊界檢查
            if 0 <= nx < game.width_px and 0 <= ny < game.height_px:
                # 排除自己 (相鄰的像素如果是同一塊方塊的一部分，不計分)
                if (nx, ny) in my_pixels_set:
                    continue

                neighbor_color = game.grid[ny][nx]

                # 同色獎勵
                if neighbor_color == piece_color:
                    score += W_COLOR_MATCH
                # 異色懲罰 (可選，這裡稍微扣一點分避免雜亂堆疊)
                elif neighbor_color != 0:
                    score -= 0.5
            else:
                # 碰到牆壁或地板
                if dx != 0:  # 左右牆
                    score += W_WALL
                # 地板不特別獎勵，自然會因為高度低而加分

        # --- 2. Hole Check (下方懸空) ---
        # 檢查正下方是否有空位
        if py + 1 < game.height_px:
            # 如果下方是空的，且下方那個點不屬於自己這塊方塊
            if game.grid[py + 1][px] == 0 and (px, py + 1) not in my_pixels_set:
                score += W_HOLES

    # --- 3. Height Score ---
    # 平均 Y 越大 (越低) 越好
    avg_y = sum_y / len(pixels)
    # W_HEIGHT 是負數 (-1.0)，所以這裡要反過來操作或者是配合權重邏輯
    # 你的邏輯：score += avg_y * -(-1.0) => score += avg_y
    # 也就是越下面 (Y越大) 分數加越多
    score += avg_y * -(W_HEIGHT)

    # --- 4. Bumpiness (平整度) ---
    # 為了保持表面平整，計算落點造成的凹凸程度
    # 這裡做一個輕量級估算：計算這塊方塊落下後的最高點差異
    # (完整計算太耗時，這裡只針對受影響的列)
    bumpiness_penalty = _calculate_local_bumpiness(game, pixels)
    score += bumpiness_penalty * abs(W_BUMPINESS)  # 因為 W_BUMPINESS 是負的，這裡直接加負值

    return score


# --- Helper Functions ---


def _calculate_local_bumpiness(game: "SandtrisCore", pixels):
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


def _get_shape_pixels_relative(shape_cells, ppc):
    """Cell 座標 -> 相對 Pixel 座標"""
    pixels = []
    for cx, cy in shape_cells:
        for i in range(ppc):
            for j in range(ppc):
                pixels.append((cx * ppc + i, cy * ppc + j))
    return pixels


def _get_projected_pixels(px, py, shape_cells, ppc):
    """Cell 座標 + Offset -> 絕對 Pixel 座標"""
    pixels = []
    for cx, cy in shape_cells:
        base_x = px + cx * ppc
        base_y = py + cy * ppc
        for i in range(ppc):
            for j in range(ppc):
                pixels.append((base_x + i, base_y + j))
    return pixels


def _rotate_shape_simulate(shape_cells):
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


def _simulate_drop_y(game: "SandtrisCore", px_x, shape_cells):
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
