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
import re
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import requests

C = None   # node.client (which loads PyTorch) is imported in main(), after the window appears

# ---------------------------------------------------------------- look and feel
BG, CARD, INK, MUTED, LINE, DARK = "#F2F2EF", "#FFFFFF", "#17181C", "#6B6F76", "#E3E3DE", "#17181C"
STATES = {   # state: (pill text, pill colour, headline, detail)
    "idle": ("Not volunteering", "#9AA0A6", "Ready when you are", "Click Volunteer now to lend this computer."),
    "waiting": ("Online", "#2563EB", "Waiting for work", "Training starts automatically when a job needs this computer."),
    "training": ("Training", "#16A34A", "Training", ""),
    "paused": ("Paused", "#D97706", "Paused", "No new work will be given to this computer until you resume."),
    "offline": ("Offline", "#DC2626", "Can't reach the server", "Retrying automatically. Check your internet connection."),
}
if sys.platform == "darwin":
    SANS, MONO = "Helvetica Neue", "Menlo"
elif sys.platform.startswith("win"):
    SANS, MONO = "Segoe UI", "Consolas"
else:
    SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"


def font(size, weight="normal"):
    return (SANS, size, weight)


class FlatButton(tk.Label):
    """A button drawn by the app, so it looks the same on Windows and macOS."""

    def __init__(self, parent, text, command, primary=True, danger=False):
        self.colors = (("#FFFFFF", DARK, "#2C2E35") if primary else
                       ("#B91C1C" if danger else INK, CARD, "#F5F5F2"))
        fg, bg, _ = self.colors
        super().__init__(parent, text=text, font=font(12, "bold"), fg=fg, bg=bg, padx=18, pady=10,
                         cursor="hand2", highlightthickness=0 if primary else 1,
                         highlightbackground=LINE)
        self.command = command
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", lambda e: self.config(bg=self.colors[2]))
        self.bind("<Leave>", lambda e: self.config(bg=self.colors[1]))


class QueueHandler(logging.Handler):
    """Sends log lines to the window (Tk widgets may only be touched from the main thread)."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    def emit(self, record):
        self.events.put(("log", self.format(record)))


# ---------------------------------------------------------------- volunteer loop
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
            pass

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
                self.active.wait()
                continue
            try:
                r = session.get(url, timeout=C.REQUEST_TIMEOUT)
                if r.status_code == 404 and C.is_not_registered(r):
                    time.sleep(C.POLL_INTERVAL)
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


# ---------------------------------------------------------------- the window
class App:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.events = queue.Queue()
        self.worker = Worker(config, events=self.events)
        self.state = "idle"
        self.rounds, self.images, self.pending_images = 0, 0, 0
        self.volunteer_seconds, self.last_tick = 0.0, None
        self.stripe_x = 0

        root.title("Lattice volunteer")
        root.configure(bg=BG)
        root.geometry("440x600")
        root.minsize(400, 520)
        root.protocol("WM_DELETE_WINDOW", self.stop)

        # Header
        head = tk.Frame(root, bg=DARK, padx=22, pady=16)
        head.pack(fill="x")
        tk.Label(head, text="lattice", font=font(22, "bold"), bg=DARK, fg="#FFFFFF").pack(side="left")
        tk.Label(head, text="volunteer", font=font(12), bg=DARK, fg="#A9ACB3").pack(side="left", padx=(8, 0), pady=(8, 0))

        body = tk.Frame(root, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)

        # Status card
        card = tk.Frame(body, bg=CARD, padx=18, pady=16, highlightthickness=1, highlightbackground=LINE)
        card.pack(fill="x")
        self.pill = tk.Label(card, font=font(10, "bold"), fg="#FFFFFF", padx=10, pady=3)
        self.pill.pack(anchor="w")
        self.headline = tk.Label(card, font=font(20, "bold"), bg=CARD, fg=INK, anchor="w")
        self.headline.pack(anchor="w", pady=(10, 2))
        self.detail = tk.Label(card, font=font(11), bg=CARD, fg=MUTED, anchor="w", justify="left", wraplength=360)
        self.detail.pack(anchor="w")
        self.stripe = tk.Canvas(card, height=4, bg=LINE, highlightthickness=0)
        self.stripe.pack(fill="x", pady=(14, 0))
        self.stripe_bar = self.stripe.create_rectangle(0, 0, 0, 4, fill=STATES["training"][1], outline="")

        # Stats
        stats = tk.Frame(body, bg=BG)
        stats.pack(fill="x", pady=12)
        self.stat_labels = {}
        for i, (key, title) in enumerate((("rounds", "Rounds"), ("images", "Images trained"), ("time", "Volunteering"))):
            box = tk.Frame(stats, bg=CARD, padx=12, pady=10, highlightthickness=1, highlightbackground=LINE)
            box.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            stats.columnconfigure(i, weight=1)
            value = tk.Label(box, text="0", font=font(17, "bold"), bg=CARD, fg=INK)
            value.pack(anchor="w")
            tk.Label(box, text=title, font=font(10), bg=CARD, fg=MUTED).pack(anchor="w")
            self.stat_labels[key] = value

        # Buttons
        buttons = tk.Frame(body, bg=BG)
        buttons.pack(fill="x", pady=(2, 12))
        self.main_btn = FlatButton(buttons, "Volunteer now", self.toggle, primary=True)
        self.main_btn.pack(side="left")
        FlatButton(buttons, "Stop Lattice", self.stop, primary=False, danger=True).pack(side="right")

        # Machine line and activity
        host = config["server"].split("://", 1)[-1]
        tk.Label(body, text=f"{config['node_id']} \u00B7 {config['device']} \u00B7 {config['cpu_count']} cores \u00B7 {host}",
                 font=font(10), bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self.toggle_log = tk.Label(body, text="Show activity \u25BE", font=font(10, "bold"), bg=BG, fg=INK, cursor="hand2")
        self.toggle_log.pack(anchor="w", pady=(10, 4))
        self.toggle_log.bind("<Button-1>", lambda e: self.flip_log())
        self.log = tk.Text(body, height=8, font=(MONO, 9), bg=CARD, fg=INK, relief="flat", wrap="word",
                           state="disabled", padx=8, pady=6, highlightthickness=1, highlightbackground=LINE)
        self.log_visible = False

        handler = QueueHandler(self.events)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.show_state("idle")
        self.root.after(200, self.pump)
        self.root.after(1000, self.tick)
        self.root.after(40, self.animate)

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

    def flip_log(self):
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log.pack(fill="both", expand=True)
            self.toggle_log.config(text="Hide activity \u25B4")
        else:
            self.log.pack_forget()
            self.toggle_log.config(text="Show activity \u25BE")

    # ---- display
    def show_state(self, state, task=None):
        if self.state == "paused" and state != "paused" and not self.worker.active.is_set():
            return                               # ignore late messages after pausing
        self.state = state
        pill, color, headline, detail = STATES[state]
        self.pill.config(text=pill.upper(), bg=color)
        if state == "training" and task:
            headline = f"Training round {task['round']}"
            detail = f"Slice {task['slice_id'] + 1} of {task['num_slices']} \u00B7 model {task['model']}"
        self.headline.config(text=headline)
        self.detail.config(text=detail)
        if state != "training":
            self.stripe.coords(self.stripe_bar, 0, 0, 0, 4)

    def animate(self):
        if self.state == "training":
            width = max(self.stripe.winfo_width(), 1)
            self.stripe_x = (self.stripe_x + 6) % (width + 120)
            self.stripe.coords(self.stripe_bar, self.stripe_x - 120, 0, self.stripe_x, 4)
        self.root.after(40, self.animate)

    def tick(self):
        now = time.monotonic()
        if self.state in ("waiting", "training") and self.last_tick is not None:
            self.volunteer_seconds += now - self.last_tick
        self.last_tick = now
        s = int(self.volunteer_seconds)
        self.stat_labels["time"].config(text=f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s")
        self.root.after(1000, self.tick)

    def pump(self):
        try:
            while True:
                kind, *rest = self.events.get_nowait()
                if kind == "state":
                    self.show_state(*rest)
                elif kind == "error":
                    messagebox.showerror("Lattice", rest[0])
                elif kind == "log":
                    self.on_log(rest[0])
        except queue.Empty:
            pass
        self.root.after(200, self.pump)

    def on_log(self, line):
        m = re.search(r"\((\d+) images", line)
        if m:
            self.pending_images = int(m.group(1))
        if "update accepted" in line:
            self.rounds += 1
            self.images += self.pending_images
            self.pending_images = 0
            self.stat_labels["rounds"].config(text=str(self.rounds))
            self.stat_labels["images"].config(text=f"{self.images:,}")
        self.log.config(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        if int(self.log.index("end-1c").split(".")[0]) > 300:
            self.log.delete("1.0", "50.0")
        self.log.config(state="disabled")


# ---------------------------------------------------------------- start-up
_instance_lock = None   # kept for the whole run: holds the "already running" lock


def main():
    global C, _instance_lock
    # One copy at a time: the first app holds this local port; a second one can't bind it.
    _instance_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _instance_lock.bind(("127.0.0.1", 47913))
    except OSError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Lattice", "Lattice is already running. Look for its window on your screen.")
        return

    root = tk.Tk()
    root.title("Lattice volunteer")
    root.configure(bg=BG)
    splash = tk.Label(root, text="Starting Lattice\u2026", font=font(16, "bold"), bg=BG, fg=INK, padx=60, pady=40)
    splash.pack()
    root.update()                                  # show the window before the slow import

    from node import client as client_module       # loads PyTorch: the slow part
    C = client_module

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            config = C.load_config()               # exits with a message if node/.env is wrong
    except SystemExit:
        root.withdraw()
        messagebox.showerror("Lattice", "Lattice isn't set up correctly:\n\n" + err.getvalue().strip()
                             + "\n\nRun the installer again, or download a fresh package.")
        return
    splash.destroy()
    App(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()