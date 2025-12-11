"""
Tetris Game Server Module
=========================

This module implements a simple HTTP server to manage multiplayer Sand Tetris games.
It handles player connections, game state synchronization, and matchmaking.

Endpoints:
    POST `/join`: Allows a player to join the lobby.
    POST `/ready_check`: Signals player readiness and checks if the game can start.
    POST `/update`: Receives game state updates from a player.
    GET `/get_opponent`: Retrieves the latest game state of the opponent.

Global State:
    SERVER_STATE: A dictionary storing all active players and game status.
"""

import http.server
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

# --- Global In-Memory Database ---
SERVER_STATE = {
    "players": {},  # Structure: {"p1": {"ready": False, "last_update": 0, "data": {...}}, "p2": ...}
    "max_players": 2,
    "game_started": False,
    "hardness": 2,
}
LOCK = threading.Lock()  # Ensure thread-safe writes


class GameRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP Request Handler for the Game Server.
    """

    def _send_json(self, data, status=200):
        """Helper method to send JSON responses."""
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _parse_body(self):
        """Helper method to parse JSON request bodies."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):
        """Handles POST requests (Join, Ready, Update)."""
        try:
            # --- 1. Player Join ---
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

                    # Initialize player data
                    if p_id not in SERVER_STATE["players"]:
                        SERVER_STATE["players"][p_id] = {
                            "ready": False,
                            "data": None,
                            "last_update": time.time(),
                        }
                        print(f"[Server] Player joined: {p_id}")

                self._send_json({"result": "joined", "player": p_id, "hardness": SERVER_STATE["hardness"]})

            # --- 2. Ready & Start Check ---
            elif self.path == "/ready_check":
                # Client sends: {"player": "p1", "ready": True}
                data = self._parse_body()
                p_id = data.get("player")
                is_ready = data.get("ready", False)

                with LOCK:
                    if p_id in SERVER_STATE["players"]:
                        player = SERVER_STATE["players"][p_id]
                        player["ready"] = is_ready
                        player["last_update"] = time.time()

                    # Check if all players (must be 2) are Ready
                    players = SERVER_STATE["players"]
                    all_ready = len(players) == 2 and all(p["ready"] for p in players.values())

                    if all_ready:
                        SERVER_STATE["game_started"] = True

                    opponent_name = ""
                    for pid in players:
                        if pid != p_id:
                            opponent_name = pid
                            break

                # Return whether to start
                self._send_json(
                    {
                        "start": SERVER_STATE["game_started"],
                        "waiting_for": 2 - len(SERVER_STATE["players"]),
                        "opponent": opponent_name,
                    }
                )

            # --- 3. In-Game Status Update ---
            elif self.path == "/update":
                # Client sends: {"player": "p1", "score": 100, "grid": [...]}
                data = self._parse_body()
                p_id = data.get("player")

                with LOCK:
                    if p_id in SERVER_STATE["players"]:
                        player = SERVER_STATE["players"][p_id]
                        # Update game data for this player
                        player["data"] = data
                        player["last_update"] = time.time()

                self._send_json({"status": "updated"})

            else:
                self._send_json({"error": "Invalid Path"}, 404)

        except Exception as e:
            print(f"Error: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_GET(self):
        """Handles GET requests (Get Opponent Data)."""
        # --- 4. Get Opponent Data ---
        parsed = urlparse(self.path)
        if parsed.path == "/get_opponent":
            query = parse_qs(parsed.query)
            # Parse who the requester is (e.g., ?player=p1)
            requester_id = query.get("player", [None])[0]

            opponent_data = None

            with LOCK:
                # Iterate through all players to find the one who is "not the requester"
                for pid, info in SERVER_STATE["players"].items():
                    if pid != requester_id:
                        opponent_data = info.get("data")
                        break

            if opponent_data:
                self._send_json(opponent_data)
            else:
                # Opponent might not have sent first data yet, or no opponent exists
                self._send_json({"error": "no data yet"})
        else:
            self._send_json({"error": "Path not found"}, 404)


if __name__ == "__main__":
    import argparse

    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Simple Python Game Server")

    # Set -p or --port argument, type int, default 8000
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    # Set --host argument, type str, default 0.0.0.0 (allow external connections)
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

    # Start server with parsed arguments
    server_address = (args.host, args.port)
    SERVER_STATE["hardness"] = args.hardness
    server = http.server.ThreadingHTTPServer(server_address, GameRequestHandler)

    print(f"Server started on {args.host}:{args.port} (Threading enabled)...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
