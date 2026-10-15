"""
dice_websocket_server.py
------------------------

Standalone WebSocket server for the dice-recognition app. It broadcasts the two
recognised die values to any connected client (Unity's dice screen, or a
generic test client) as JSON text messages.

This is deliberately a separate module, the same way the camera-counting
project keeps its WebSocket server in its own file pair: main.py's
detection/display loop only has to hand its detections to ``update()`` once per
frame -- it never needs to know about sockets, threads or JSON, and this file
never needs to import ultralytics/torch.

Wire protocol
    URL   : ws://127.0.0.1:8082/dice
    Frame : {"type": "dice_update", "values": [<left die>, <right die>]}
              * values[0] is the left-most die, values[1] the right-most
              * each value is an integer 1..6
              * a frame is sent only when exactly two dice were detected,
                both values are valid, AND the pair changed since the last
                frame that was sent (otherwise the previous values stand)

Requires the ``websockets`` package (not in requirements.txt yet):
    pip install websockets

Wiring it into main.py (4 lines, main.py otherwise unchanged)::

    import dice_websocket_server as dice_ws       # at the top

    dice_ws.start()                               # before the inference loop

    detected_dice = model(frame)[0].boxes
    dice_ws.update_from_yolo(detected_dice)       # inside the loop

    dice_ws.stop()                                # in the finally block (optional)

Or drive it directly with (x, value) pairs, keeping this file dependency-free::

    dice_ws.update([(x_left_px, value), (x_left_px, value)])

Smoke-test the server on its own with changing dummy values::

    python dice_websocket_server.py
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Iterable, Optional, Sequence

try:
    import websockets
except ImportError as exc:  # give a clearer hint than a bare ImportError
    raise SystemExit(
        "dice_websocket_server needs the 'websockets' package.\n"
        "    pip install websockets"
    ) from exc


HOST = "127.0.0.1"
PORT = 8082
PATH = "/dice"

DIE_MIN = 1
DIE_MAX = 6
DICE_PER_ROLL = 2


class DiceWebSocketServer:
    """A WebSocket server that pushes ``dice_update`` messages on value changes.

    Every public method is safe to call from the detection thread. The actual
    socket I/O runs on a private asyncio event loop in a background daemon
    thread, so it never blocks the OpenCV inference loop -- the same reason the
    camera-counting server runs its service loop on its own thread.
    """

    def __init__(self, host: str = HOST, port: int = PORT, path: str = PATH):
        self._host = host
        self._port = port
        self._path = path

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._clients: set = set()          # touched only on the loop thread
        self._started = threading.Event()

        # Last pair actually broadcast, e.g. [2, 5]. Used both to suppress
        # duplicate messages and to bring a freshly connected client up to
        # date straight away. Read/written from both threads -> lock.
        self._last_sent: Optional[list[int]] = None
        self._state_lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the server. Idempotent; returns once it is accepting clients."""
        if self._thread and self._thread.is_alive():
            return
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run, name="dice-websocket", daemon=True
        )
        self._thread.start()
        # Block until the loop + listener are really up, so the caller can
        # update() on the very next line without racing startup.
        if not self._started.wait(timeout=5):
            raise RuntimeError("dice WebSocket server failed to start in time")

    def stop(self) -> None:
        """Stop the server and join its thread. Idempotent."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._server = None
        self._clients.clear()
        self._started.clear()

    def __enter__(self) -> "DiceWebSocketServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ----------------------------------------------------------------- updating

    def update(self, dice: Iterable[Sequence[float]]) -> None:
        """Feed the current detections.

        ``dice`` is an iterable of ``(x, value)`` pairs, one per detected die,
        where ``x`` is any horizontal position (used only to order the dice
        left to right) and ``value`` is the pip count.

        A message is broadcast only if all of the following hold:
          * exactly two dice were detected,
          * both values are integers in 1..6, and
          * the left-to-right pair differs from the last pair sent.
        Otherwise nothing is sent and the last known values stand.
        """
        pairs = list(dice)
        if len(pairs) != DICE_PER_ROLL:
            return

        pairs.sort(key=lambda p: p[0])  # left to right

        try:
            values = [int(value) for _x, value in pairs]
        except (TypeError, ValueError):
            return

        if any(v < DIE_MIN or v > DIE_MAX for v in values):
            return

        with self._state_lock:
            if values == self._last_sent:
                return
            self._last_sent = values

        self._broadcast({"type": "dice_update", "values": values})

    def update_from_yolo(self, boxes) -> None:
        """Convenience wrapper for an ultralytics ``Boxes`` object.

        Matches the detection format in main.py -- ``model(frame)[0].boxes`` --
        mapping each box to ``(left_x, class_index + 1)`` and calling
        ``update()``. Kept separate so importing this module never pulls in
        ultralytics/torch.
        """
        dice: list[tuple[float, int]] = []
        for box in boxes:
            x1 = float(box.xyxy[0][0])
            value = int(box.cls[0]) + 1
            dice.append((x1, value))
        self.update(dice)

    # ------------------------------------------------------------------ interns

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
            self._started.set()
            self._loop.run_forever()
        finally:
            if self._server is not None:
                self._server.close()
                self._loop.run_until_complete(self._server.wait_closed())
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _serve(self) -> None:
        # websockets.serve() is the stable API across v10..v14 (kept as a
        # compatibility alias for the newer websockets.asyncio.server.serve).
        self._server = await websockets.serve(
            self._handle_client, self._host, self._port
        )
        print(f"[dice-ws] listening on ws://{self._host}:{self._port}{self._path}")

    async def _handle_client(self, websocket, *args) -> None:
        # Every connection on this port is served, whatever its path -- same as
        # the camera-counting WebSocket server. `self._path` is only the
        # advertised URL; enforcing it reliably across websockets versions
        # (handler gets (ws, path) on <=12, just (ws) on >=13) is not worth the
        # fragility for a localhost single-purpose server.
        self._clients.add(websocket)
        peer = getattr(websocket, "remote_address", None)
        print(f"[dice-ws] client connected {peer} ({len(self._clients)} total)")

        # Bring a new / reconnected client straight up to date.
        with self._state_lock:
            last = list(self._last_sent) if self._last_sent else None
        if last is not None:
            await self._send_one(websocket, {"type": "dice_update", "values": last})

        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            print(
                f"[dice-ws] client disconnected {peer} "
                f"({len(self._clients)} total)"
            )

    def _broadcast(self, message: dict) -> None:
        """Schedule ``message`` for every client. Callable from any thread."""
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), loop)

    async def _broadcast_async(self, message: dict) -> None:
        for websocket in list(self._clients):
            await self._send_one(websocket, message)

    @staticmethod
    async def _send_one(websocket, message: dict) -> None:
        try:
            await websocket.send(json.dumps(message, separators=(",", ":")))
        except Exception:
            # Client went away mid-send; _handle_client's finally cleans up.
            pass


# ------------------------------------------------------------------- module API
# Mirrors camera-counting's WebSocketCountServer free-function style: one
# implicit server instance for the common "just call it from the loop" case.

_default: Optional[DiceWebSocketServer] = None


def _instance() -> DiceWebSocketServer:
    global _default
    if _default is None:
        _default = DiceWebSocketServer()
    return _default


def start(host: str = HOST, port: int = PORT, path: str = PATH) -> DiceWebSocketServer:
    """Start (and return) the default server instance."""
    global _default
    if _default is None:
        _default = DiceWebSocketServer(host, port, path)
    _default.start()
    return _default


def stop() -> None:
    """Stop the default server instance, if one was started."""
    if _default is not None:
        _default.stop()


def update(dice: Iterable[Sequence[float]]) -> None:
    """``DiceWebSocketServer.update`` on the default instance."""
    _instance().update(dice)


def update_from_yolo(boxes) -> None:
    """``DiceWebSocketServer.update_from_yolo`` on the default instance."""
    _instance().update_from_yolo(boxes)


# --------------------------------------------------------------------- self-test

def _demo() -> None:
    """Start the server and emit changing dummy values every 2s so a generic
    WebSocket client can verify the wire format. Ctrl+C to stop."""
    import random
    import time

    server = DiceWebSocketServer()
    server.start()
    print("Connect a client to ws://127.0.0.1:8082/dice  --  Ctrl+C to quit.")
    try:
        while True:
            a, b = random.randint(1, 6), random.randint(1, 6)
            # feed as (x, value): give 'a' the smaller x so the order is a, b
            server.update([(0.0, a), (100.0, b)])
            print(f"fed dice -> [{a}, {b}]")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.stop()


if __name__ == "__main__":
    _demo()
