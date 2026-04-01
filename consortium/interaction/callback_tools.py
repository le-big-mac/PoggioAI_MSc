"""
Live-steering socket and interrupt tools for the LangGraph pipeline.

The socket listener is unchanged: a background thread accepts TCP connections
and forwards typed lines into a Queue.

The manager node calls check_for_interrupt() at each routing step.  If a user
interrupt has been received, the instruction is returned as a HumanMessage that
the manager injects into graph state before continuing.
"""

from __future__ import annotations

import socket
import threading
from queue import Empty, Queue
from typing import Callable, Iterable, Optional


from .user_inststep import UserInstructionStep


# ---------------------------------------------------------------------------
# TCP socket listener
# ---------------------------------------------------------------------------

def setup_user_input_socket(host: str = "127.0.0.1", port: int = 5001) -> Queue:
    """
    Start a tiny TCP server that accepts multiple client connections.
    Anything typed by any connected client is pushed line-by-line into the
    returned Queue[str].
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[Info] Awaiting user input on {host}:{port}")
    print(f"[Info] Connect from another terminal:  nc {host} {port}")

    input_queue: Queue = Queue()
    stop_flag = threading.Event()

    def socket_listener(sock: socket.socket, client_addr, queue: Queue) -> None:
        buf = ""
        try:
            print(f"[Info] Connected: {client_addr}")
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                buf += data.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    queue.put(line.rstrip("\r"))
        except Exception as exc:
            print(f"[Warn] Client {client_addr} listener error: {exc}")
        finally:
            try:
                sock.close()
            finally:
                print(f"[Info] Input connection closed: {client_addr}")

    def accept_loop() -> None:
        try:
            while not stop_flag.is_set():
                try:
                    client_socket, client_addr = server_socket.accept()
                except OSError:
                    break
                threading.Thread(
                    target=socket_listener,
                    args=(client_socket, client_addr, input_queue),
                    daemon=True,
                ).start()
        finally:
            try:
                server_socket.close()
            except OSError:
                pass
            print("[Info] Server socket closed")

    threading.Thread(target=accept_loop, daemon=True).start()
    return input_queue


# ---------------------------------------------------------------------------
# Interrupt checker (called by the manager node each routing step)
# ---------------------------------------------------------------------------

_INTERRUPT_SIGNALS = frozenset({"interrupt", "stop", "pause"})


def make_interrupt_checker(
    input_queue: Queue,
    interrupt_signals: Optional[Iterable[str]] = None,
) -> Callable[[], Optional[dict]]:
    """
    Return a zero-arg callable that the manager node should call once per step.

    If no interrupt has been received, returns None.
    If an interrupt signal arrives, blocks reading the user instruction and
    returns a HumanMessage containing that instruction (to be appended to
    graph state messages).
    """
    signals = frozenset(interrupt_signals or _INTERRUPT_SIGNALS)
    paused = False

    def _try_get_nowait(q: Queue) -> Optional[str]:
        try:
            return q.get_nowait()
        except Empty:
            return None

    def _read_until_double_enter(q: Queue, banner: str, timeout: float = 300.0) -> str:
        print(banner)
        print(">>> Type your instruction. Press Enter twice to finish.\n")
        print(f">>> (Auto-cancels after {int(timeout)}s of inactivity)\n")
        lines: list = []
        empty_streak = 0
        while empty_streak < 2:
            try:
                line = q.get(timeout=timeout)
            except Empty:
                print("[Warn] Instruction timeout — resuming without change.")
                return ""
            if line is None:
                line = ""
            if line.strip() == "":
                empty_streak += 1
            else:
                empty_streak = 0
            lines.append(line)
        while lines and lines[-1].strip() == "":
            lines.pop()
            if lines and lines[-1].strip() == "":
                lines.pop()
                break
        return "\n".join(lines).strip()

    def check() -> Optional[dict]:
        nonlocal paused

        if not paused:
            cmd = _try_get_nowait(input_queue)
            if cmd and cmd.strip().lower() in signals:
                paused = True
                print("\n🛑 Interrupt received — pausing for user instruction...")

        if not paused:
            return None

        print("\n" + "=" * 60)
        print("📝 WAITING FOR USER INSTRUCTION")
        print("=" * 60)

        instruction = _read_until_double_enter(
            input_queue, banner="--- PROVIDE INSTRUCTION (leave empty to cancel) ---"
        )

        if instruction:
            print("\nIs this a 'modification' to the current task or a 'new' task? (m/n)")
            choice = ""
            while choice not in ["m", "n"]:
                try:
                    choice = input_queue.get(timeout=60).strip().lower()
                except Empty:
                    print("[Warn] Choice timeout — treating as modification.")
                    choice = "m"
                    break
                if choice not in ["m", "n"]:
                    print("Invalid choice. Please enter 'm' for modification or 'n' for new task.")

            is_new_task = choice == "n"
            step = UserInstructionStep(user_instruction=instruction, is_new_task=is_new_task)
            msgs = step.to_messages()
            paused = False
            print("✅ Resuming...\n")
            return msgs[0] if msgs else None

        paused = False
        print("ℹ️ No instruction provided. Resuming without change.\n")
        return None

    return check


# ---------------------------------------------------------------------------
# Backwards-compatible shim (used by runner.py during transition)
# ---------------------------------------------------------------------------

def make_user_input_step_callback(input_queue: Queue, interrupt_signals=None):
    """
    Compatibility shim: returns an interrupt-checker callable.
    The manager node calls this each step to check for live-steering input.
    """
    return make_interrupt_checker(input_queue=input_queue, interrupt_signals=interrupt_signals)
