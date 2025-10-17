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
import json
from pathlib import Path
import psutil


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
        self.start_tree_refresh()
    
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
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Checking...", font=("Arial", 10, "bold"))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.pid_label = ttk.Label(status_frame, text="PID: None")
        self.pid_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        # Mode checkboxes
        mode_frame = ttk.LabelFrame(control_frame, text="Mode Options", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.test_mode_var = tk.BooleanVar()
        self.real_mode_var = tk.BooleanVar()
        self.scan_once_var = tk.BooleanVar()
        self.daemon_var = tk.BooleanVar()
        self.sync_only_var = tk.BooleanVar()
        self.dedupe_only_var = tk.BooleanVar()
        
        ttk.Checkbutton(mode_frame, text="Test Mode", variable=self.test_mode_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(mode_frame, text="Real Mode (-R)", variable=self.real_mode_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(mode_frame, text="Scan Once", variable=self.scan_once_var).grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(mode_frame, text="Daemon Mode", variable=self.daemon_var).grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(mode_frame, text="Sync Only", variable=self.sync_only_var).grid(row=1, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(mode_frame, text="Dedupe Only", variable=self.dedupe_only_var).grid(row=1, column=2, sticky=tk.W, padx=(0, 10))
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
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
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
        ttk.Button(tree_controls, text="Open Folder", command=self.open_folder).grid(row=0, column=1, padx=(0, 5))
        
        self.auto_tree_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tree_controls, text="Auto-refresh", variable=self.auto_tree_refresh_var).grid(row=0, column=2, padx=(5, 0))
    
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
        # Check if organizer is running
        success, stdout, stderr = self.run_command(f"bash {self.manage_script} status")
        
        if success and "is running" in stdout.lower():
            self.organizer_running = True
            self.status_label.config(text="Status: Running", foreground="green")
            
            # Try to get PID
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if 'file_organizer.py' in ' '.join(proc.info['cmdline'] or []):
                        self.organizer_pid = proc.info['pid']
                        self.pid_label.config(text=f"PID: {self.organizer_pid}")
                        break
            except:
                self.pid_label.config(text="PID: Unknown")
        else:
            self.organizer_running = False
            self.status_label.config(text="Status: Stopped", foreground="red")
            self.pid_label.config(text="PID: None")
            self.organizer_pid = None
        
        # Update button states
        self.start_button.config(state="normal" if not self.organizer_running else "disabled")
        self.stop_button.config(state="normal" if self.organizer_running else "disabled")
        self.restart_button.config(state="normal")
    
    def start_organizer(self):
        """Start the organizer with selected options"""
        # Build command based on checkboxes
        command_parts = ["bash", str(self.manage_script)]
        
        if self.test_mode_var.get() and not self.real_mode_var.get():
            command_parts.append("test")
        elif self.real_mode_var.get():
            if self.scan_once_var.get():
                command_parts.append("test-real")
            elif self.sync_only_var.get():
                command_parts.append("sync")
            elif self.dedupe_only_var.get():
                command_parts.append("dedupe")
            elif self.daemon_var.get():
                command_parts.append("start")
            else:
                command_parts.append("test-real")
        else:
            messagebox.showwarning("Mode Selection", "Please select either Test Mode or Real Mode")
            return
        
        command = " ".join(command_parts)
        
        # Run the command
        success, stdout, stderr = self.run_command(command, wait=False)
        
        if success:
            self.update_status()
            messagebox.showinfo("Organizer Started", f"Command: {command}")
        else:
            messagebox.showerror("Error", f"Failed to start organizer:\n{stderr}")
    
    def stop_organizer(self):
        """Stop the organizer"""
        success, stdout, stderr = self.run_command(f"bash {self.manage_script} stop")
        
        if success:
            self.update_status()
            messagebox.showinfo("Organizer Stopped", "File organizer has been stopped")
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
    
    def refresh_tree(self):
        """Refresh the file tree"""
        # Clear existing tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Determine which organized folder to show based on mode
        organized_path = None
        
        # Check if we're in test mode (test mode checkbox is checked)
        if self.test_mode_var.get() and not self.real_mode_var.get():
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
    
    def open_folder(self):
        """Open the selected folder in file manager"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a folder in the tree")
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        if values and values[0]:
            path = Path(values[0])
            if path.exists():
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(str(path))
                    elif os.name == 'posix':  # macOS and Linux
                        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(path)])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open folder: {e}")
            else:
                messagebox.showwarning("Path Not Found", f"Path does not exist: {path}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = FileOrganizerApp(root)
    
    # Handle window closing
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
