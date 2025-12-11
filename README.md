# Sand Tetris (LabVIEW + Python)

> [!NOTE]
> This project was created as a learning exercise while studying LabVIEW. It demonstrates how to integrate a LabVIEW front-end with a complex Python back-end.

## Overview

This project implements a **Sand Tetris** game. Unlike traditional Tetris, blocks "shatter" into individual sand pixels upon landing. These pixels obey simple physics (gravity and sliding), allowing them to fill gaps below.

The project uses a hybrid architecture:

- **Core Logic & AI:** Written in **Python** for performance and ease of algorithm implementation.
- **User Interface:** Built in **LabVIEW** for visualization and user control.
- **Networking:** A Python HTTP server enables multiplayer functionality.

## Features

- **Sand Physics:** Blocks turn into independent pixels that fall and slide to fill empty spaces.
- **Line Clearing:** Lines are cleared when a continuous color connects the left and right walls (using BFS).
- **AI Auto-Play:** Includes a smart AI agent (`tetris_ai.py`) that evaluates board states based on potential energy and surface smoothness to play automatically.
- **Multiplayer Support:** A dedicated server (`server.py`) allows two players to compete and view each other's progress in real-time.
- **Adjustable Difficulty:** Supports Easy, Medium, and Hard modes (affects the number of colors used).

## Project Structure

### Python Modules (Backend)

- [`tetris_core.py`](tetris_core.py): The main game engine. Handles grid management, collision detection, sand physics, and line clearing.
- [`tetris_ai.py`](tetris_ai.py): The AI logic. Simulates moves and calculates scores to determine the best placement.
- [`tetris_remote.py`](tetris_remote.py): Client-side networking logic. Handles background threads for sending game state and receiving opponent data.
- [`server.py`](server.py): A standalone HTTP server that manages game rooms and synchronizes state between players.

### LabVIEW Files (Frontend)

- [`SandTetris.lvproj`](SandTetris.lvproj): The main LabVIEW project file.
- [`sandtetris.vi`](sandtetris.vi): The main application VI (User Interface).
- `next_piece.ctl` & `sandtetris_type.ctl`: Type definitions and controls.

## Requirements

- **LabVIEW:** Community 2025 Q3 (or compatible version).
- **Python:** Version 3.6 - 3.14 (or even newer version if compatible).
  - *Note:* Ensure the Python version specified in the LabVIEW "Python Version" String Control matches your installed version.

## How to Run

### 1. Single Player / AI Mode

1. Open `SandTetris.lvproj` in LabVIEW.
2. Open `sandtetris.vi`.
3. Run the VI.
4. Click "START" button.
5. Use the controls to toggle AI mode or play manually.

### 2. Multiplayer Mode

To play with an opponent, you must first start the game server.

#### Step 1: Start the Server

Open a terminal in the project directory and run:

```bash
python server.py --host 0.0.0.0 --port 8000
```

#### Step 2: Join a Game in LabVIEW

1. Open `sandtetris.vi` in LabVIEW.
2. Run the VI to connect and play with the opponent.
3. Click "Join Multiplayer" button.
4. Enter the server's IP address and port (e.g., `127.0.0.1:8000`) and the display name you like.
5. Wait for both players to be ready, then the game will start.
6. Who last longer wins!

## Controls

### LabVIEW Front-End

- **Move Left:** Left Arrow
- **Move Right:** Right Arrow
- **Rotate:** Up Arrow
- **Soft Drop:** Down Arrow

### Python Back-End (AI and Server)

- No direct controls. Configure settings in the code or use command-line arguments for the server.

## Troubleshooting

- **Python Errors:** Ensure the correct (or compatible) Python version is installed.
- **LabVIEW Errors:** Check that all necessary files are downloaded and present and that the project is correctly configured.
- **Networking Issues:** Ensure the server is running and accessible at the specified IP address and port. Check firewall settings if necessary.

## Acknowledgments

- Inspired by the classic Tetris game and Sand simulation concepts.
- Leveraged resources from the LabVIEW and Python communities for learning and troubleshooting.

## License

This project is released under the [Apache-2.0 License](LICENSE).
