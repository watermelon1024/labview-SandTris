import http.server
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

# --- 全域記憶體資料庫 ---
SERVER_STATE = {
    "players": {},  # 結構: {"p1": {"ready": False, "last_update": 0, "data": {...}}, "p2": ...}
    "max_players": 2,
    "game_started": False,
    "hardness": 2,
}
LOCK = threading.Lock()  # 確保多執行緒寫入安全


class GameRequestHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):
        try:
            # --- 1. 玩家加入 (Join) ---
            if self.path == "/join":
                data = self._parse_body()
                p_id = data.get("player")
                if not p_id:
                    self._send_json({"error": "Invalid player ID"}, 400)
                    return

                with LOCK:
                    # cleanup timed-out players
                    for p in list(SERVER_STATE["players"].keys()):
                        if SERVER_STATE["players"][p]["last_update"] + 30 < time.time():
                            print(f"[Server] Player timed out: {p}")
                            del SERVER_STATE["players"][p]
                            SERVER_STATE["game_started"] = False

                    if len(SERVER_STATE["players"]) >= SERVER_STATE["max_players"]:
                        self._send_json({"error": "Room is full"}, 403)
                        return
                    if p_id in SERVER_STATE["players"]:
                        self._send_json({"error": "Name taken"}, 403)
                        return

                    # 初始化玩家資料
                    if p_id not in SERVER_STATE["players"]:
                        SERVER_STATE["players"][p_id] = {
                            "ready": False,
                            "data": None,
                            "last_update": time.time(),
                        }
                        print(f"[Server] Player joined: {p_id}")

                self._send_json({"result": "joined", "player": p_id, "hardness": SERVER_STATE["hardness"]})

            # --- 2. 準備與開始檢查 (Ready Check) ---
            elif self.path == "/ready_check":
                # 客戶端傳來: {"player": "p1", "ready": True}
                data = self._parse_body()
                p_id = data.get("player")
                is_ready = data.get("ready", False)

                with LOCK:
                    if p_id in SERVER_STATE["players"]:
                        player = SERVER_STATE["players"][p_id]
                        player["ready"] = is_ready
                        player["last_update"] = time.time()

                    # 檢查是否所有玩家 (必須滿2人) 都 Ready
                    players = SERVER_STATE["players"]
                    all_ready = len(players) == 2 and all(p["ready"] for p in players.values())

                    if all_ready:
                        SERVER_STATE["game_started"] = True

                    opponent_name = ""
                    for pid in players:
                        if pid != p_id:
                            opponent_name = pid
                            break

                # 回傳是否開始
                self._send_json(
                    {
                        "start": SERVER_STATE["game_started"],
                        "waiting_for": 2 - len(SERVER_STATE["players"]),
                        "opponent": opponent_name,
                    }
                )

            # --- 3. 遊戲中更新狀態 (Update) ---
            elif self.path == "/update":
                # 客戶端傳來: {"player": "p1", "score": 100, "grid": [...]}
                data = self._parse_body()
                p_id = data.get("player")

                with LOCK:
                    if p_id in SERVER_STATE["players"]:
                        player = SERVER_STATE["players"][p_id]
                        # 更新該玩家的遊戲數據
                        player["data"] = data
                        player["last_update"] = time.time()

                self._send_json({"status": "updated"})

            else:
                self._send_json({"error": "Invalid Path"}, 404)

        except Exception as e:
            print(f"Error: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_GET(self):
        # --- 4. 獲取對手資料 (Get Opponent) ---
        parsed = urlparse(self.path)
        if parsed.path == "/get_opponent":
            query = parse_qs(parsed.query)
            # 解析請求者是誰 (例如 ?player=p1)
            requester_id = query.get("player", [None])[0]

            opponent_data = None

            with LOCK:
                # 遍歷所有玩家，找出 "不是請求者" 的那一位
                for pid, info in SERVER_STATE["players"].items():
                    if pid != requester_id:
                        opponent_data = info.get("data")
                        break

            if opponent_data:
                self._send_json(opponent_data)
            else:
                # 可能對手還沒送出第一筆資料，或是還沒有對手
                self._send_json({"error": "no data yet"})
        else:
            self._send_json({"error": "Path not found"}, 404)


if __name__ == "__main__":
    import argparse

    # --- 參數解析設定 ---
    parser = argparse.ArgumentParser(description="Simple Python Game Server")

    # 設定 -p 或 --port 參數，型別為 int，預設 8000
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    # 設定 --host 參數，型別為 str，預設 0.0.0.0 (允許外部連線)
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--hardness",
        type=int,
        default=2,
        help="Game hardness level (default: 2), 1: easy; 2: normal; 3: hard",
    )

    args = parser.parse_args()

    # 使用解析後的參數啟動伺服器
    server_address = (args.host, args.port)
    SERVER_STATE["hardness"] = args.hardness
    server = http.server.ThreadingHTTPServer(server_address, GameRequestHandler)

    print(f"Server started on {args.host}:{args.port} (Threading enabled)...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
