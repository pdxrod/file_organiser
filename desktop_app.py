#!/usr/bin/env python3
"""
File Organizer Desktop App
Cross-platform GUI for managing the file organizer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import time
import os
import signal
import yaml
import sys
from pathlib import Path
from collections import defaultdict
import psutil


def check_desktop_app_running():
    """Check if desktop_app.py is already running"""
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'desktop_app.py' in ' '.join(proc.info['cmdline']):
                    pid = proc.info['pid']
                    if pid != current_pid:
                        return True, pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, None
    except Exception:
        return False, None


class FileOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Organizer")
        self.root.geometry("1200x750")

        self.center_window()

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        # State
        self.organizer_running = False
        self.organizer_pid = None
        self.log_refresh_thread = None
        self.tree_refresh_thread = None
        self._category_data = {}  # semantic_category -> [files]

        self.script_dir = Path(__file__).parent
        self.manage_script = self.script_dir / "manage_organizer.sh"

        self.setup_ui()
        self.update_status()
        self.start_log_refresh()
        self.start_status_refresh()
        self.start_tree_refresh()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ── UI setup ────────────────────────────────────────────────────────

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self.setup_control_panel(main_frame)

        # Content area: notebook with three tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        self._setup_log_tab()
        self._setup_files_tab()
        self._setup_categories_tab()

    def setup_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Organizer Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        control_frame.columnconfigure(1, weight=1)

        # Left: buttons + status
        left = ttk.Frame(control_frame)
        left.grid(row=0, column=0, sticky="nw")

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=0, column=0, sticky="w")

        self.start_button = ttk.Button(btn_frame, text="Start", command=self.start_organizer)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self.stop_organizer)
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        self.restart_button = ttk.Button(btn_frame, text="Restart", command=self.restart_organizer)
        self.restart_button.grid(row=0, column=2, padx=(0, 5))
        ttk.Button(btn_frame, text="Refresh", command=self.update_status).grid(row=0, column=3, padx=(0, 5))

        status_frame = ttk.Frame(left)
        status_frame.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.status_box = tk.Frame(status_frame, bg="black", relief="raised", bd=2)
        self.status_box.grid(row=0, column=0)
        self.status_label = tk.Label(
            self.status_box, text="Checking...",
            font=("Arial", 12, "bold"), bg="black", fg="white", padx=8, pady=4,
        )
        self.status_label.pack()
        ttk.Button(status_frame, text="Exit", command=self.exit_app).grid(
            row=0, column=1, padx=(10, 0)
        )

        # Right: mode radio buttons
        mode_frame = ttk.LabelFrame(control_frame, text="Mode", padding="5")
        mode_frame.grid(row=0, column=1, sticky="ne", padx=(15, 0))
        self.mode_var = tk.StringVar(value="test")
        modes = [
            ("Test", "test"), ("Real Background Daemon", "daemon"),
            ("Real - Scan Once", "scan_once"), ("Remove duplicate links", "dedupe"),
            ("Sync Only", "sync"),
        ]
        for i, (label, val) in enumerate(modes):
            ttk.Radiobutton(mode_frame, text=label, variable=self.mode_var, value=val).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 12)
            )

    # ── Log tab ─────────────────────────────────────────────────────────

    def _setup_log_tab(self):
        log_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(log_frame, text="  Logs  ")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Menlo", 11))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        ctrl = ttk.Frame(log_frame)
        ctrl.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(ctrl, text="Clear", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(ctrl, text="Refresh", command=self.refresh_log).grid(row=0, column=1, padx=(0, 5))
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto-refresh", variable=self.auto_refresh_var).grid(row=0, column=2, padx=(8, 0))

    # ── Files tab ───────────────────────────────────────────────────────

    def _setup_files_tab(self):
        files_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(files_frame, text="  Files  ")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        container = ttk.Frame(files_frame)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(container, show="tree")
        sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        ctrl = ttk.Frame(files_frame)
        ctrl.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(ctrl, text="Refresh", command=self.refresh_tree).grid(row=0, column=0, padx=(0, 5))
        self.auto_tree_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Auto-refresh", variable=self.auto_tree_refresh_var).grid(row=0, column=1, padx=(8, 0))

    # ── Categories tab ──────────────────────────────────────────────────

    def _setup_categories_tab(self):
        cat_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(cat_frame, text="  Categories  ")
        cat_frame.columnconfigure(0, weight=1)
        cat_frame.rowconfigure(1, weight=1)

        # Search bar
        search_frame = ttk.Frame(cat_frame)
        search_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        search_frame.columnconfigure(1, weight=1)
        ttk.Label(search_frame, text="Filter:").grid(row=0, column=0, padx=(0, 5))
        self._cat_search_var = tk.StringVar()
        self._cat_search_var.trace_add("write", lambda *_: self._filter_categories())
        search_entry = ttk.Entry(search_frame, textvariable=self._cat_search_var)
        search_entry.grid(row=0, column=1, sticky="ew")

        # Category tree
        container = ttk.Frame(cat_frame)
        container.grid(row=1, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.cat_tree = ttk.Treeview(container, columns=("count",), show="tree headings")
        self.cat_tree.heading("#0", text="Category / File", anchor="w")
        self.cat_tree.heading("count", text="Files", anchor="e")
        self.cat_tree.column("#0", width=600, stretch=True)
        self.cat_tree.column("count", width=70, stretch=False, anchor="e")
        cat_sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.cat_tree.yview)
        self.cat_tree.configure(yscrollcommand=cat_sb.set)
        self.cat_tree.grid(row=0, column=0, sticky="nsew")
        cat_sb.grid(row=0, column=1, sticky="ns")

        # Double-click to reveal in Finder
        self.cat_tree.bind("<Double-1>", self._on_category_double_click)

        ctrl = ttk.Frame(cat_frame)
        ctrl.grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(ctrl, text="Refresh", command=self.refresh_categories).grid(row=0, column=0, padx=(0, 5))
        self._cat_summary = ttk.Label(ctrl, text="")
        self._cat_summary.grid(row=0, column=1, padx=(10, 0))

    def _on_category_double_click(self, event):
        """Reveal file in Finder on double-click."""
        sel = self.cat_tree.selection()
        if not sel:
            return
        item = sel[0]
        path_str = self.cat_tree.item(item, "values")
        if path_str and len(path_str) > 0:
            path_val = path_str[0] if len(path_str) == 1 else path_str[1] if len(path_str) > 1 else ""
            # values tuple has (count,) for parents, or we stash the path as tag
            pass
        tags = self.cat_tree.item(item, "tags")
        if tags:
            file_path = tags[0]
            if os.path.exists(file_path):
                subprocess.Popen(["open", "-R", file_path])

    def refresh_categories(self):
        """Scan ~/organized for semantic categories and populate the tree."""
        organized_path = self._get_organized_path()
        if not organized_path or not organized_path.exists():
            return

        # Build category data: scan top-level dirs for Title-Case names
        # (semantic categories use Title Case, keyword categories are lowercase)
        self._category_data = {}
        all_dirs = []
        try:
            all_dirs = sorted(organized_path.iterdir())
        except Exception:
            return

        for d in all_dirs:
            if not d.is_dir():
                continue
            name = d.name
            # Collect files (symlinks) in each category folder
            try:
                files = [f for f in d.iterdir() if f.is_symlink() or f.is_file()]
                if files:
                    self._category_data[name] = files
            except Exception:
                continue

        self._render_category_tree()

    def _render_category_tree(self, filter_text=""):
        """Render the category tree, optionally filtering by search text."""
        self.cat_tree.delete(*self.cat_tree.get_children())
        ft = filter_text.lower()

        # Separate semantic (Title Case) and keyword (lowercase) categories
        semantic = {}
        keyword = {}
        for name, files in sorted(self._category_data.items()):
            if ft and ft not in name.lower() and not any(ft in f.name.lower() for f in files):
                continue
            if any(c.isupper() for c in name) and " " in name:
                semantic[name] = files
            else:
                keyword[name] = files

        total_cats = 0
        total_files = 0

        # Semantic categories first
        if semantic:
            sem_node = self.cat_tree.insert("", "end", text="Semantic Categories", values=(str(len(semantic)),), open=True)
            for cat_name, files in sorted(semantic.items(), key=lambda x: len(x[1]), reverse=True):
                cat_node = self.cat_tree.insert(
                    sem_node, "end", text=cat_name, values=(str(len(files)),), open=False
                )
                for f in sorted(files, key=lambda x: x.name)[:200]:
                    resolved = ""
                    try:
                        resolved = str(f.resolve()) if f.is_symlink() else str(f)
                    except Exception:
                        resolved = str(f)
                    self.cat_tree.insert(cat_node, "end", text=f.name, values=("",), tags=(resolved,))
                total_cats += 1
                total_files += len(files)

        # Keyword categories
        if keyword:
            kw_node = self.cat_tree.insert("", "end", text="Keyword Categories", values=(str(len(keyword)),), open=False)
            for cat_name, files in sorted(keyword.items(), key=lambda x: len(x[1]), reverse=True):
                cat_node = self.cat_tree.insert(
                    kw_node, "end", text=cat_name, values=(str(len(files)),), open=False
                )
                for f in sorted(files, key=lambda x: x.name)[:200]:
                    resolved = ""
                    try:
                        resolved = str(f.resolve()) if f.is_symlink() else str(f)
                    except Exception:
                        resolved = str(f)
                    self.cat_tree.insert(cat_node, "end", text=f.name, values=("",), tags=(resolved,))
                total_cats += 1
                total_files += len(files)

        self._cat_summary.config(text=f"{total_cats} categories, {total_files} file links")

    def _filter_categories(self):
        ft = self._cat_search_var.get()
        self._render_category_tree(ft)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_organized_path(self):
        """Get the path to the organized folder based on mode."""
        if self.mode_var.get() == "test":
            p = self.script_dir / "test" / "organized"
            return p if p.exists() else None

        try:
            cfg = self.script_dir / "config.yaml"
            if cfg.exists():
                with open(cfg) as f:
                    config = yaml.safe_load(f)
                    ob = config.get('output_base', '') if config else ''
                    if ob:
                        p = Path(os.path.expanduser(ob))
                        if p.exists():
                            return p
        except Exception:
            pass

        for p in [Path.home() / "organized", self.script_dir / "test" / "organized"]:
            if p.exists():
                return p
        return None

    def run_command(self, command, wait=True):
        try:
            if wait:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                return result.returncode == 0, result.stdout, result.stderr
            else:
                subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return True, "", ""
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    # ── Status ───────────────────────────────────────────────────────────

    def update_status(self):
        has_running, processes = self.check_for_running_processes()
        if has_running:
            is_real = any('--REAL' in p.get('cmdline', '') or '-R' in p.get('cmdline', '') for p in processes)
            self.organizer_running = True
            self.status_label.config(text=f"Running ({'REAL' if is_real else 'TEST'})", fg="lime")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.restart_button.config(state="normal")
        else:
            self.organizer_running = False
            self.status_label.config(text="Stopped", fg="white")
            self.organizer_pid = None
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.restart_button.config(state="disabled")

    def start_organizer(self):
        parts = ["bash", str(self.manage_script)]
        mode_map = {"test": "test", "daemon": "start", "scan_once": "test-real", "dedupe": "dedupe", "sync": "sync"}
        mode = self.mode_var.get()
        if mode not in mode_map:
            messagebox.showwarning("Mode Selection", "Please select a mode")
            return
        parts.append(mode_map[mode])
        self.start_button.config(state="disabled")
        success, _, stderr = self.run_command(" ".join(parts), wait=False)
        if success:
            self.root.after(1000, self.update_status)
        else:
            self.start_button.config(state="normal")
            messagebox.showerror("Error", f"Failed to start organizer:\n{stderr}")

    def stop_organizer(self):
        cmd = f"bash '{self.manage_script.absolute()}' stop"
        success, _, stderr = self.run_command(cmd, wait=True)
        if success:
            self.update_status()
        elif hasattr(self, 'root') and self.root.winfo_exists():
            messagebox.showerror("Error", f"Failed to stop organizer:\n{stderr}")

    def restart_organizer(self):
        self.stop_organizer()
        time.sleep(1)
        self.start_organizer()

    # ── Logs ─────────────────────────────────────────────────────────────

    def refresh_log(self):
        self.log_path = Path.home() / ".file_organizer.log"
        if not self.log_path.exists():
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Log file not found")
            return
        try:
            with open(self.log_path) as f:
                lines = f.readlines()
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(tk.END, ''.join(lines[-80:]))
                self.log_text.see(tk.END)
        except Exception as e:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Error reading log: {e}")

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def start_log_refresh(self):
        def loop():
            while True:
                if self.auto_refresh_var.get():
                    self.root.after(0, self.refresh_log)
                time.sleep(5)
        threading.Thread(target=loop, daemon=True).start()

    def start_status_refresh(self):
        def loop():
            while True:
                self.root.after(0, self.update_status)
                time.sleep(2)
        threading.Thread(target=loop, daemon=True).start()

    # ── File tree ────────────────────────────────────────────────────────

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        organized_path = self._get_organized_path()
        if not organized_path:
            self.tree.insert("", "end", text="No organized folder found")
            return
        self._build_tree_node("", organized_path)

    def _build_tree_node(self, parent, path):
        try:
            if path.is_dir():
                node = self.tree.insert(parent, "end", text=path.name, values=(str(path),))
                children = sorted(path.iterdir())
                for child in children[:200]:
                    if child.is_dir():
                        self._build_tree_node(node, child)
                    else:
                        self.tree.insert(node, "end", text=child.name, values=(str(child),))
        except PermissionError:
            self.tree.insert(parent, "end", text=f"{path.name} (Permission Denied)")
        except Exception as e:
            self.tree.insert(parent, "end", text=f"{path.name} (Error: {e})")

    def start_tree_refresh(self):
        def loop():
            while True:
                if self.auto_tree_refresh_var.get():
                    self.root.after(0, self.refresh_tree)
                    self.root.after(0, self.refresh_categories)
                time.sleep(30)
        threading.Thread(target=loop, daemon=True).start()

    # ── Process management ───────────────────────────────────────────────

    def get_running_processes(self):
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'file_organizer.py' in ' '.join(proc.info['cmdline']):
                        cmdline = ' '.join(proc.info['cmdline'])
                        processes.append({
                            'pid': proc.info['pid'],
                            'is_daemon': '--internal-daemon' in cmdline,
                            'cmdline': cmdline,
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return processes

    def check_for_running_processes(self):
        procs = self.get_running_processes()
        return len(procs) > 0, procs

    def exit_app(self):
        self.on_closing()

    def on_closing(self):
        print("Closing File Organizer GUI...")
        print("Stopping file_organizer.py...")
        try:
            result = subprocess.run(
                ["bash", str(self.manage_script.absolute()), "stop"],
                cwd=str(self.script_dir.absolute()),
                capture_output=True, text=True, timeout=15,
            )
            if result.stdout:
                print(f"Stop output: {result.stdout}")
        except subprocess.TimeoutExpired:
            print("Warning: Stop command timed out")
        except Exception as e:
            print(f"Error running stop command: {e}")

        time.sleep(2)

        has_running, processes = self.check_for_running_processes()
        if has_running:
            for p in processes:
                try:
                    os.kill(p['pid'], signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(1)
            _, processes = self.check_for_running_processes()
            for p in processes:
                try:
                    os.kill(p['pid'], signal.SIGKILL)
                except Exception:
                    pass

        self.root.destroy()


def main():
    is_running, pid = check_desktop_app_running()
    if is_running:
        print(f"File Organizer Desktop App is already running (PID {pid})")
        print("Only one instance of the desktop app can run at a time.")
        sys.exit(1)

    root = tk.Tk()
    app = FileOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
