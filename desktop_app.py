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
import json
import sys
from pathlib import Path
import psutil


def check_desktop_app_running():
    """Check if desktop_app.py is already running"""
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'desktop_app.py' in ' '.join(proc.info['cmdline']):
                    pid = proc.info['pid']
                    if pid != current_pid:  # Don't count ourselves
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
        self.root.geometry("1000x700")
        
        # State variables
        self.organizer_running = False
        self.organizer_pid = None
        self.log_refresh_thread = None
        self.tree_refresh_thread = None
        
        # Get script directory for manage_organizer.sh
        self.script_dir = Path(__file__).parent
        self.manage_script = self.script_dir / "manage_organizer.sh"
        
        self.setup_ui()
        self.update_status()
        self.start_log_refresh()
        self.start_status_refresh()
        self.start_tree_refresh()
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Create the main UI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Control panel (top)
        self.setup_control_panel(main_frame)
        
        # Log viewer (left)
        self.setup_log_viewer(main_frame)
        
        # File tree (right)
        self.setup_file_tree(main_frame)
    
    def setup_control_panel(self, parent):
        """Create the control panel with buttons and checkboxes"""
        control_frame = ttk.LabelFrame(parent, text="Organizer Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configure grid weights for the control frame
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        
        # Left side: Status box and buttons
        left_frame = ttk.Frame(control_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # Status display - prominent black box (moved down)
        status_frame = ttk.Frame(left_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create a black background frame for status
        self.status_box = tk.Frame(status_frame, bg="black", relief="raised", bd=2)
        self.status_box.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = tk.Label(self.status_box, text="Checking...", 
                                   font=("Arial", 12, "bold"), 
                                   bg="black", fg="white", 
                                   padx=8, pady=4)
        self.status_label.pack()
        
        # Mode radio buttons (moved to the right)
        mode_frame = ttk.LabelFrame(control_frame, text="Mode Options", padding="5")
        mode_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), pady=(0, 10), padx=(10, 0))
        
        # Single variable for all radio buttons (exclusive selection)
        self.mode_var = tk.StringVar(value="test")
        
        # Radio buttons for exclusive mode selection
        ttk.Radiobutton(mode_frame, text="Test", variable=self.mode_var, value="test").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Real Background Daemon", variable=self.mode_var, value="daemon").grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Real - Scan Once", variable=self.mode_var, value="scan_once").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Deduplicate", variable=self.mode_var, value="dedupe").grid(row=1, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Sync Only", variable=self.mode_var, value="sync").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        
        # Action buttons (moved to left side, above status box)
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_organizer)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_organizer)
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        self.restart_button = ttk.Button(button_frame, text="Restart", command=self.restart_organizer)
        self.restart_button.grid(row=0, column=2, padx=(0, 5))
        
        self.refresh_button = ttk.Button(button_frame, text="Refresh Status", command=self.update_status)
        self.refresh_button.grid(row=0, column=3, padx=(0, 5))
        
        # Log file path
        log_frame = ttk.Frame(control_frame)
        log_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(log_frame, text="Log File:").grid(row=0, column=0, sticky=tk.W)
        self.log_path = Path.home() / ".file_organizer.log"
        self.log_path_label = ttk.Label(log_frame, text=str(self.log_path), foreground="blue")
        self.log_path_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
    
    def setup_log_viewer(self, parent):
        """Create the log viewer panel"""
        log_frame = ttk.LabelFrame(parent, text="Recent Logs", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=50, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(log_controls, text="Clear", command=self.clear_log).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(log_controls, text="Refresh", command=self.refresh_log).grid(row=0, column=1, padx=(0, 5))
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto-refresh", variable=self.auto_refresh_var).grid(row=0, column=2, padx=(5, 0))
    
    def setup_file_tree(self, parent):
        """Create the file tree panel"""
        tree_frame = ttk.LabelFrame(parent, text="Organized Files", padding="5")
        tree_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Treeview with scrollbar
        tree_container = ttk.Frame(tree_frame)
        tree_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(tree_container, show="tree")
        tree_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Tree controls
        tree_controls = ttk.Frame(tree_frame)
        tree_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(tree_controls, text="Refresh Tree", command=self.refresh_tree).grid(row=0, column=0, padx=(0, 5))
        
        self.auto_tree_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tree_controls, text="Auto-refresh", variable=self.auto_tree_refresh_var).grid(row=0, column=1, padx=(5, 0))
    
    def run_command(self, command, wait=True):
        """Run a shell command and return the result"""
        try:
            if wait:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                return result.returncode == 0, result.stdout, result.stderr
            else:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return True, "", ""
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def update_status(self):
        """Update the status display"""
        # Check for running processes first (most accurate)
        has_running, processes = self.check_for_running_processes()
        
        if has_running:
            # Something is running
            self.organizer_running = True
            self.status_label.config(text="Running", fg="lime")
            
            # Update button states
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.restart_button.config(state="normal")
        else:
            # Nothing running
            self.organizer_running = False
            self.status_label.config(text="Stopped", fg="white")
            self.organizer_pid = None
            
            # Update button states
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.restart_button.config(state="disabled")
    
    def start_organizer(self):
        """Start the organizer with selected options"""
        # Build command based on radio button selection
        command_parts = ["bash", str(self.manage_script)]
        
        mode = self.mode_var.get()
        
        if mode == "test":
            command_parts.append("test")
        elif mode == "daemon":
            command_parts.append("start")
        elif mode == "scan_once":
            command_parts.append("test-real")
        elif mode == "dedupe":
            command_parts.append("dedupe")
        elif mode == "sync":
            command_parts.append("sync")
        else:
            messagebox.showwarning("Mode Selection", "Please select a mode")
            return
        
        command = " ".join(command_parts)
        
        # Disable start button immediately to prevent multiple starts
        self.start_button.config(state="disabled")
        
        # Run the command
        success, stdout, stderr = self.run_command(command, wait=False)
        
        if success:
            # Update status after a short delay to allow process to start
            self.root.after(1000, self.update_status)
        else:
            # Re-enable start button if command failed
            self.start_button.config(state="normal")
            messagebox.showerror("Error", f"Failed to start organizer:\n{stderr}")
    
    def stop_organizer(self):
        """Stop the organizer"""
        success, stdout, stderr = self.run_command(f"bash {self.manage_script} stop")
        
        if success:
            self.update_status()
        else:
            messagebox.showerror("Error", f"Failed to stop organizer:\n{stderr}")
    
    def restart_organizer(self):
        """Restart the organizer"""
        self.stop_organizer()
        time.sleep(1)
        self.start_organizer()
    
    def refresh_log(self):
        """Refresh the log display"""
        if not self.log_path.exists():
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Log file not found")
            return
        
        try:
            # Read last 50 lines
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-50:] if len(lines) > 50 else lines
                log_content = ''.join(last_lines)
            
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, log_content)
            self.log_text.see(tk.END)  # Scroll to bottom
        except Exception as e:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Error reading log: {e}")
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
    
    def start_log_refresh(self):
        """Start the log auto-refresh thread"""
        def refresh_loop():
            while True:
                if self.auto_refresh_var.get():
                    self.root.after(0, self.refresh_log)
                time.sleep(5)  # Refresh every 5 seconds
        
        self.log_refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.log_refresh_thread.start()
    
    def start_status_refresh(self):
        """Start the status auto-refresh thread"""
        def status_refresh_loop():
            while True:
                self.root.after(0, self.update_status)
                time.sleep(2)  # Refresh every 2 seconds
        
        self.status_refresh_thread = threading.Thread(target=status_refresh_loop, daemon=True)
        self.status_refresh_thread.start()
    
    def refresh_tree(self):
        """Refresh the file tree"""
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Determine which organized folder to show based on mode
        organized_path = None
        
        # Check if we're in test mode
        if self.mode_var.get() == "test":
            # Test mode - show test/organized
            test_organized = self.script_dir / "test" / "organized"
            if test_organized.exists():
                organized_path = test_organized
            else:
                self.tree.insert("", "end", text="Test organized folder not found", values=("",))
                return
        else:
            # Production mode - find the actual organized folder
            try:
                # Try to read config to find output_base
                config_path = self.script_dir / "organizer_config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        output_base = config.get('output_base', '')
                        if output_base:
                            organized_path = Path(output_base)
            except:
                pass
            
            # Fallback to common locations
            if not organized_path or not organized_path.exists():
                possible_paths = [
                    Path.home() / "organized",
                    self.script_dir / "test" / "organized",
                    Path.home() / ".file_organizer" / "organized"
                ]
                for path in possible_paths:
                    if path.exists():
                        organized_path = path
                        break
        
        if not organized_path or not organized_path.exists():
            self.tree.insert("", "end", text="No organized folder found", values=("",))
            return
        
        # Build tree
        self.build_tree_node("", organized_path)
    
    def build_tree_node(self, parent, path):
        """Recursively build tree nodes"""
        try:
            if path.is_dir():
                # Add directory
                node = self.tree.insert(parent, "end", text=path.name, values=(str(path),))
                
                # Add children (limit to avoid performance issues)
                children = sorted(path.iterdir())
                for child in children[:100]:  # Limit to 100 items per directory
                    if child.is_dir():
                        self.build_tree_node(node, child)
                    else:
                        self.tree.insert(node, "end", text=child.name, values=(str(child),))
            else:
                # Add file
                self.tree.insert(parent, "end", text=path.name, values=(str(path),))
        except PermissionError:
            self.tree.insert(parent, "end", text=f"{path.name} (Permission Denied)", values=("",))
        except Exception as e:
            self.tree.insert(parent, "end", text=f"{path.name} (Error: {e})", values=("",))
    
    def start_tree_refresh(self):
        """Start the tree auto-refresh thread"""
        def refresh_loop():
            while True:
                if self.auto_tree_refresh_var.get():
                    self.root.after(0, self.refresh_tree)
                time.sleep(30)  # Refresh every 30 seconds
        
        self.tree_refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self.tree_refresh_thread.start()
    
    def get_running_processes(self):
        """Get all running file_organizer.py processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'file_organizer.py' in ' '.join(proc.info['cmdline']):
                        # Check if it's a daemon process (has --internal-daemon flag)
                        cmdline = ' '.join(proc.info['cmdline'])
                        is_daemon = '--internal-daemon' in cmdline
                        processes.append({
                            'pid': proc.info['pid'],
                            'is_daemon': is_daemon,
                            'cmdline': cmdline
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error getting processes: {e}")
        return processes
    
    def kill_non_daemon_processes(self):
        """Kill all non-daemon file_organizer.py processes"""
        processes = self.get_running_processes()
        killed_count = 0
        
        for proc_info in processes:
            if not proc_info['is_daemon']:
                try:
                    os.kill(proc_info['pid'], signal.SIGTERM)
                    killed_count += 1
                    print(f"Killed non-daemon process PID {proc_info['pid']}")
                except Exception as e:
                    print(f"Could not kill PID {proc_info['pid']}: {e}")
        
        if killed_count > 0:
            print(f"Killed {killed_count} non-daemon file organizer processes")
    
    def check_for_running_processes(self):
        """Check if any file_organizer.py processes are running"""
        processes = self.get_running_processes()
        return len(processes) > 0, processes
    
    def on_closing(self):
        """Handle window closing - kill non-daemon processes"""
        print("Closing File Organizer GUI...")
        
        # Kill any non-daemon processes
        self.kill_non_daemon_processes()
        
        # Close the window
        self.root.destroy()


def main():
    """Main entry point"""
    # Check if desktop app is already running
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
