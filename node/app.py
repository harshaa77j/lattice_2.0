"""Lattice volunteer app: a small window to start, pause and stop volunteering.

    python -m node.app

Built on node/client.py: the same registration, heartbeats and training code,
with a window instead of a terminal. Closing the window stops Lattice completely;
the server notices within its heartbeat timeout and carries on without this machine.
"""
import contextlib
import io
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

import requests

from node import client as C

COLORS = {"idle": "#9AA0A6", "waiting": "#3B82F6", "training": "#16A34A",
          "paused": "#D97706", "offline": "#DC2626"}
BG, CARD, INK, MUTED = "#F2F2EF", "#FFFFFF", "#17181C", "#6B6F76"


class QueueHandler(logging.Handler):
    """Sends log lines to the window (Tk widgets may only be touched from the main thread)."""
    def __init__(self, events):
        super().__init__()
        self.events = events

    def emit(self, record):
        self.events.put(("log", self.format(record)))


class Worker:
    """Runs the volunteer loop in a background thread; the window controls it."""

    def __init__(self, config, events):
        self.config, self.events = config, events
        self.active = threading.Event()       # set = accept work
        self.started = False

    def volunteer(self):
        self.active.set()
        if not self.started:
            self.started = True
            threading.Thread(target=self._run, name="worker", daemon=True).start()
        else:
            self._set_available(True)

    def pause(self):
        self.active.clear()
        self._set_available(False)
        self.events.put(("state", "paused", None))

    def _set_available(self, flag):
        url = f"{self.config['server']}/nodes/{self.config['node_id']}/availability"
        try:
            C.make_session(self.config).post(url, json={"available": flag}, timeout=C.REQUEST_TIMEOUT)
        except requests.RequestException:
            pass                                # the next heartbeat or poll will show offline

    def _run(self):
        session = C.make_session(self.config)
        self.events.put(("state", "waiting", None))
        try:
            C.register(self.config, session)
        except C.Rejected as e:
            self.events.put(("error", str(e)))
            return
        threading.Thread(target=C.heartbeat_loop, args=(self.config,), name="heartbeat", daemon=True).start()
        url = f"{self.config['server']}/task/{self.config['node_id']}"
        while True:
            if not self.active.is_set():
                self.active.wait()               # paused: sleep until "Volunteer now"
                continue
            try:
                r = session.get(url, timeout=C.REQUEST_TIMEOUT)
                if r.status_code == 404 and C.is_not_registered(r):
                    time.sleep(C.POLL_INTERVAL)  # the heartbeat thread re-registers
                    continue
                r.raise_for_status()
                task = r.json()
                if task.get("status") == "train":
                    self.events.put(("state", "training", task))
                    C.run_task(self.config, session, task)
                    continue
                self.events.put(("state", "waiting", None))
            except requests.RequestException:
                self.events.put(("state", "offline", None))
            except Exception:
                logging.getLogger("lattice.node").exception("Task failed; waiting for the next one")
            time.sleep(C.POLL_INTERVAL)


class App:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.events = queue.Queue()
        self.worker = Worker(config, self.events)
        self.rounds = 0
        self.state = "idle"

        root.title("Lattice volunteer")
        root.configure(bg=BG)
        root.minsize(440, 520)
        root.protocol("WM_DELETE_WINDOW", self.stop)

        wrap = tk.Frame(root, bg=BG, padx=22, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="lattice", font=("Helvetica", 26, "bold"), bg=BG, fg=INK).pack(anchor="w")
        tk.Label(wrap, text="Lend your idle computer to train models.", font=("Helvetica", 12),
                 bg=BG, fg=MUTED).pack(anchor="w", pady=(0, 14))

        card = tk.Frame(wrap, bg=CARD, padx=18, pady=16, highlightthickness=0)
        card.pack(fill="x")
        row = tk.Frame(card, bg=CARD)
        row.pack(anchor="w")
        self.dot = tk.Canvas(row, width=16, height=16, bg=CARD, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.dot_id = self.dot.create_oval(2, 2, 14, 14, fill=COLORS["idle"], outline="")
        self.status = tk.Label(row, text="Not volunteering", font=("Helvetica", 18, "bold"), bg=CARD, fg=INK)
        self.status.pack(side="left")
        self.detail = tk.Label(card, text="Click Volunteer now to start.", font=("Helvetica", 12),
                               bg=CARD, fg=MUTED, justify="left")
        self.detail.pack(anchor="w", pady=(6, 10))
        host = config["server"].split("://", 1)[-1]
        for text in (f"This computer: {config['node_id']} \u00B7 {config['device']} \u00B7 {config['cpu_count']} cores",
                     f"Server: {host}"):
            tk.Label(card, text=text, font=("Helvetica", 11), bg=CARD, fg=MUTED).pack(anchor="w")
        self.count = tk.Label(card, text="Rounds contributed: 0", font=("Helvetica", 12, "bold"), bg=CARD, fg=INK)
        self.count.pack(anchor="w", pady=(8, 0))

        buttons = tk.Frame(wrap, bg=BG)
        buttons.pack(fill="x", pady=14)
        self.main_btn = tk.Button(buttons, text="Volunteer now", font=("Helvetica", 13, "bold"),
                                  command=self.toggle, padx=14, pady=8)
        self.main_btn.pack(side="left")
        tk.Button(buttons, text="Stop Lattice", font=("Helvetica", 13), fg="#B91C1C",
                  command=self.stop, padx=14, pady=8).pack(side="right")

        tk.Label(wrap, text="Activity", font=("Helvetica", 11, "bold"), bg=BG, fg=MUTED).pack(anchor="w")
        self.log = tk.Text(wrap, height=9, font=("Menlo", 10), bg=CARD, fg=INK, relief="flat",
                           wrap="word", state="disabled", padx=8, pady=6)
        self.log.pack(fill="both", expand=True)

        handler = QueueHandler(self.events)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        self.root.after(200, self.pump)

    # ---- controls
    def toggle(self):
        if self.state in ("idle", "paused"):
            self.worker.volunteer()
            self.main_btn.config(text="Pause")
        else:
            self.worker.pause()
            self.main_btn.config(text="Volunteer now")

    def stop(self):
        # Exiting at once stops heartbeats; the server marks this machine dropped and carries on.
        self.root.destroy()
        os._exit(0)

    # ---- updates from the worker (main thread only)
    def set_state(self, state, task=None):
        if self.state == "paused" and state != "paused" and not self.worker.active.is_set():
            return                               # ignore late messages after pausing
        self.state = state
        self.dot.itemconfig(self.dot_id, fill=COLORS[state])
        if state == "training" and task:
            self.status.config(text=f"Training round {task['round']}")
            self.detail.config(text=f"Slice {task['slice_id'] + 1} of {task['num_slices']} \u00B7 {task['model']}")
        else:
            self.status.config(text={"waiting": "Waiting for work", "paused": "Paused",
                                     "offline": "Can't reach the server", "idle": "Not volunteering"}[state])
            self.detail.config(text={"waiting": "Ready. Training starts when a job needs this computer.",
                                     "paused": "This computer won't be given new work until you resume.",
                                     "offline": "Retrying automatically. Check your internet connection.",
                                     "idle": "Click Volunteer now to start."}[state])

    def pump(self):
        try:
            while True:
                kind, *rest = self.events.get_nowait()
                if kind == "state":
                    self.set_state(*rest)
                elif kind == "error":
                    messagebox.showerror("Lattice", rest[0])
                elif kind == "log":
                    line = rest[0]
                    if "update accepted" in line:
                        self.rounds += 1
                        self.count.config(text=f"Rounds contributed: {self.rounds}")
                    self.log.config(state="normal")
                    self.log.insert("end", line + "\n")
                    self.log.see("end")
                    if int(self.log.index("end-1c").split(".")[0]) > 300:
                        self.log.delete("1.0", "50.0")
                    self.log.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(200, self.pump)


def main():
    root = tk.Tk()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            config = C.load_config()             # exits with a message if node/.env is wrong
    except SystemExit:
        root.withdraw()
        messagebox.showerror("Lattice", "Lattice isn't set up correctly:\n\n" + err.getvalue().strip()
                             + "\n\nRun the installer again, or download a fresh package.")
        return
    App(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()