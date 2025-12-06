import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import tetris_core

JOIN_ERR_CANNOT_CONNECT = 1
JOIN_ERR_ROOM_FULL = 2
JOIN_ERR_NAME_TAKEN = 3
JOIN_ERR_INVALID_NAME = 4


class SandtrisRemote(tetris_core.SandtrisCore):
    def __init__(self, *args, server_url: str, player_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = server_url
        self.player_id = player_id
        self.opponent_data: Optional[dict] = None
        self.opponent_name: Optional[str] = None
        self.is_opponent_updated: bool = False

        self.network_interval = 0.5  # seconds
        self.get_lock = threading.Lock()
        self.post_lock = threading.Lock()
        self.running = False

    def _post(self, endpoint, data):
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                return json.loads(error_body)
            except Exception:
                return {"error": str(e), "body": error_body}
        except Exception as e:
            print(f"[Network Error] {e}")
            return {"error": str(e)}

    def _get(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                return json.loads(error_body)
            except Exception:
                return {"error": str(e), "body": error_body}
        except Exception as e:
            print(f"[Network Error] {e}")
            return {"error": str(e)}

    def join_game(self):
        print(f"[{self.player_id}] Connecting to {self.base_url} ...")
        try:
            resp = self._post("/join", {"player": self.player_id})
            if "error" in resp:
                error_msg = resp["error"].lower()
                if "full" in error_msg:
                    print(f"[{self.player_id}] 無法加入: 房間已滿")
                    return JOIN_ERR_ROOM_FULL
                elif "taken" in error_msg:
                    print(f"[{self.player_id}] 無法加入: 名稱已被使用")
                    return JOIN_ERR_NAME_TAKEN
                elif "invalid" in error_msg:
                    print(f"[{self.player_id}] 無法加入: 無效的名稱")
                    return JOIN_ERR_INVALID_NAME
                else:
                    print(f"[{self.player_id}] 無法加入: {error_msg}")
                    return JOIN_ERR_CANNOT_CONNECT
            print(f"[{self.player_id}] 加入成功: {resp}")
            self.set_hardness(resp.get("hardness", 2))  # type: ignore
            return 0
        except Exception as e:
            print(f"無法連線至伺服器 ({self.base_url}): {e}")
            return JOIN_ERR_CANNOT_CONNECT

    def check_start(self, ready: bool = True):
        print(f"[{self.player_id}] 準備就緒，等待對手...")
        try:
            resp = self._post("/ready_check", {"player": self.player_id, "ready": ready})
            self.opponent_name = resp.get("opponent", None)
            if resp.get("start"):
                print(f"[{self.player_id}] --- 遊戲開始！ ---")
                # 設定旗標並啟動網路執行緒
                self.running = True
                # 建立執行緒
                t_send = threading.Thread(target=self._send_loop, daemon=True)
                t_recv = threading.Thread(target=self._recv_loop, daemon=True)
                # 啟動執行緒
                t_send.start()
                t_recv.start()
                return True
        except Exception as e:
            print(f"連線錯誤: {e}")
        return False

    def step(self, action: int) -> None:
        with self.post_lock:
            super().step(action)

    # --- 執行緒 1: 發送資料迴圈 ---
    def _send_loop(self):
        while self.running:
            # 1. 安全地複製當前狀態 (避免讀到寫一半的資料)
            with self.post_lock:
                data = {
                    "player": self.player_id,
                    "score": self.score,
                    "grid": self.get_render_grid(),
                    "game_over": self.game_over,
                }

            # 2. 發送請求 (可能會卡住幾百毫秒，但不會影響主程式)
            self._post("/update", data)

            # 3. 休息 x 秒 (網路同步頻率)
            time.sleep(self.network_interval)

    # --- 執行緒 2: 接收資料迴圈 ---
    def _recv_loop(self):
        while self.running:
            # 1. 發送 GET 請求
            data = self._get("/get_opponent", {"player": self.player_id})

            # 2. 如果成功拿到資料，安全地更新到共享變數
            if data and "error" not in data:
                with self.get_lock:
                    self.opponent_data = data
                self.is_opponent_updated = True

            # 3. 休息 x 秒 (網路同步頻率)
            time.sleep(self.network_interval)


# --- Interface for LabVIEW ---
def init(cols: int, rows: int, ppc: int, server_url: str, player_id: str, **kwargs):
    return SandtrisRemote(cols, rows, ppc, server_url=server_url, player_id=player_id, **kwargs)


def join(game: SandtrisRemote):
    return game.join_game()


def ready(game: SandtrisRemote, ready: bool = True):
    return game.check_start(ready=ready)


def update(*args, **kwargs):
    return tetris_core.update(*args, **kwargs)


def get_hardness(game: SandtrisRemote):
    return game.hardness


def get_view(*args, **kwargs):
    return tetris_core.get_view(*args, **kwargs)


def get_statistics(*args, **kwargs):
    return tetris_core.get_statistics(*args, **kwargs)


def get_opponent_name(game: SandtrisRemote):
    return game.opponent_name if game.opponent_name else ""


def is_opponent_update(game: SandtrisRemote):
    return game.is_opponent_updated


def get_opponent_data(game: SandtrisRemote):
    if game.opponent_data is None:
        return (0, [[]], False)

    with game.get_lock:
        game.is_opponent_updated = False
        return (
            game.opponent_data["score"],
            tetris_core._render_grid_to_24bit(game.opponent_data["grid"]),
            game.opponent_data["game_over"],
        )
