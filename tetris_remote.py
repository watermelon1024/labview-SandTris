"""
Tetris Remote Client Module
===========================

This module implements a network-enabled version of the Sand Tetris game client.
It extends the `SandtrisCore` to support multiplayer features by communicating
with a central game server.

Features:
- Connects to a game server via HTTP.
- Synchronizes game state (score, grid) with the server.
- Retrieves opponent's game state for display.
- Handles network communication in separate threads to avoid blocking the game loop.

Classes:
    SandtrisRemote: The network-enabled game client class.

Functions:
    init: Factory function to create a new remote game instance.
    join: Joins a game room on the server.
    ready: Signals readiness to start the game.
    update: Updates the game state and handles network sync.
    get_opponent_data: Retrieves the opponent's current game state.
"""

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
    """
    A subclass of SandtrisCore that adds networking capabilities.

    Attributes:
        base_url (str): The URL of the game server.
        player_id (str): The unique identifier for this player.
        opponent_data (dict): The latest data received from the opponent.
        opponent_name (str): The name of the opponent.
        is_opponent_updated (bool): Flag indicating if new opponent data is available.
        network_interval (float): Time interval (in seconds) between network requests.
    """

    def __init__(self, *args, server_url: str, player_id: str, **kwargs):
        """
        Initialize the remote game client.

        Args:
            server_url (str): The URL of the game server.
            player_id (str): The player's name/ID.
            *args, **kwargs: Arguments passed to the parent SandtrisCore class.
        """
        super().__init__(*args, **kwargs)
        self.base_url = server_url
        self.player_id = player_id
        self.opponent_data: Optional[dict] = None
        self.opponent_name: Optional[str] = None
        self.is_opponent_updated: bool = False

        self.network_interval = 0.5  # seconds
        self.get_lock = threading.Lock()
        self.post_lock = threading.Lock()
        self.t_send: Optional[threading.Thread] = None
        self.t_recv: Optional[threading.Thread] = None
        self.running = False

    def _post(self, endpoint, data):
        """Helper method to send HTTP POST requests."""
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
        """Helper method to send HTTP GET requests."""
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
        """
        Attempts to join the game server.

        Returns:
            int: 0 on success, or an error code (JOIN_ERR_*) on failure.
        """
        print(f"[{self.player_id}] Connecting to {self.base_url} ...")
        try:
            resp = self._post("/join", {"player": self.player_id})
            if "error" in resp:
                error_msg = resp["error"].lower()
                if "full" in error_msg:
                    print(f"[{self.player_id}] Cannot join: Room is full")
                    return JOIN_ERR_ROOM_FULL
                elif "taken" in error_msg:
                    print(f"[{self.player_id}] Cannot join: Name taken")
                    return JOIN_ERR_NAME_TAKEN
                elif "invalid" in error_msg:
                    print(f"[{self.player_id}] Cannot join: Invalid name")
                    return JOIN_ERR_INVALID_NAME
                else:
                    print(f"[{self.player_id}] Cannot join: {error_msg}")
                    return JOIN_ERR_CANNOT_CONNECT
            print(f"[{self.player_id}] Joined successfully: {resp}")
            self.set_hardness(resp.get("hardness", 2))  # type: ignore
            return 0
        except Exception as e:
            print(f"Cannot connect to server ({self.base_url}): {e}")
            return JOIN_ERR_CANNOT_CONNECT

    def check_start(self, ready: bool = True):
        """
        Checks if the game can start (i.e., if the opponent is ready).
        Also signals this player's readiness.

        Args:
            ready (bool): Whether this player is ready.

        Returns:
            bool: True if the game has started, False otherwise.
        """
        print(f"[{self.player_id}] Ready, waiting for opponent...")
        try:
            resp = self._post("/ready_check", {"player": self.player_id, "ready": ready})
            self.opponent_name = resp.get("opponent", None)
            if resp.get("start"):
                print(f"[{self.player_id}] --- Game Start! ---")
                # Set flag and start network threads
                self.running = True
                # Create threads
                self.t_send = threading.Thread(target=self._send_loop, daemon=True)
                self.t_recv = threading.Thread(target=self._recv_loop, daemon=True)
                # Start threads
                self.t_send.start()
                self.t_recv.start()
                return True
        except Exception as e:
            print(f"Connection error: {e}")
        return False

    def step(self, action: int) -> None:
        """
        Advances the game state and manages network threads.
        Overrides the parent `step` method to include thread safety.
        """
        with self.post_lock:
            super().step(action)
        if self.game_over:
            self.running = False
            if self.t_send is not None:
                self.t_send.join()
            if self.t_recv is not None:
                self.t_recv.join()

    def post_data(self):
        """Sends the current game state to the server."""
        # 1. Safely copy current state (avoid reading partially written data)
        with self.post_lock:
            data = {
                "player": self.player_id,
                "score": self.score,
                "grid": self.get_render_grid(),
                "game_over": self.game_over,
            }
            # 2. Send request (may block for a few hundred ms, but won't affect main program)
        self._post("/update", data)

    # --- Thread 1: Data Sending Loop ---
    def _send_loop(self):
        """Background thread loop for sending data to the server."""
        while self.running:
            self.post_data()

            # 3. Sleep for x seconds (Network sync interval)
            time.sleep(self.network_interval)

        # Send data one last time to ensure update
        self.post_data()

    # --- Thread 2: Data Receiving Loop ---
    def _recv_loop(self):
        """Background thread loop for receiving opponent data from the server."""
        while self.running:
            # 1. Send GET request
            data = self._get("/get_opponent", {"player": self.player_id})

            # 2. If data received successfully, safely update shared variables
            if data and "error" not in data:
                with self.get_lock:
                    self.opponent_data = data
                self.is_opponent_updated = True

            # 3. Sleep for x seconds (Network sync interval)
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


def get_opponent_data(game: SandtrisRemote, scaler: int = 1):
    if game.opponent_data is None:
        return (0, [[]], False)

    with game.get_lock:
        game.is_opponent_updated = False
        return (
            game.opponent_data["score"],
            tetris_core._render_grid_to_24bit(game.opponent_data["grid"], scaler),
            game.opponent_data["game_over"],
        )


if __name__ == "__main__":
    import argparse

    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Simple Python Game Client Demo")

    parser.add_argument(
        "-p", "--player", type=str, default="demo player", help="Player ID (default: demo player)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost:8080", help="Server host (default: localhost:8080)"
    )

    args = parser.parse_args()

    s = SandtrisRemote(8, 12, 4, server_url=f"http://{args.host}", player_id=args.player)
    err = s.join_game()
    if err == 0:
        while True:
            if s.check_start():
                break
            time.sleep(1)
        while not s.game_over:
            s.step(0)
            if s.opponent_data and s.opponent_data.get("game_over", False):
                print(f"[{s.player_id}] Opponent has finished the game!")
                break
            time.sleep(0.1)
