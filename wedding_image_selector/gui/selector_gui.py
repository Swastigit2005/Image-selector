# selector_gui.py - Fully Synced GUI with main.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import sys

# --- PATH SETUP & IMPORTS ---

# Ensure main project directory is in sys.path for imports
# ...existing code...
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ...existing code...
from main import select_best_images
import config

# --- UTILITY FUNCTIONS ---

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def get_image_files(folder):
    try:
        return [f for f in os.listdir(folder) if is_image_file(f)]
    except Exception:
        return []

def show_error(title, message):
    messagebox.showerror(title, message)

def log(msg):
    log_box.configure(state="normal")
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    log_box.configure(state="disabled")

def update_progress(val):
    progress["value"] = val
    root.update_idletasks()

# --- BROWSE DIALOGS ---

def browse_folder(var, title):
    folder = filedialog.askdirectory(title=title)
    if folder:
        var.set(folder)

def browse_input():
    browse_folder(input_var, "Select Input Folder")

def browse_output():
    browse_folder(output_var, "Select Output Folder")

# --- MAIN PROCESS LOGIC ---

def validate_paths(input_path, output_path):
    if not input_path or not output_path:
        show_error("Missing Path", "Please select both input and output folders.")
        return False
    if not os.path.isdir(input_path):
        show_error("Invalid Input", "Input folder does not exist.")
        return False
    if not os.path.isdir(output_path):
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            show_error("Invalid Output", f"Could not create output folder: {e}")
            return False
    return True

def clear_log():
    log_box.configure(state="normal")
    log_box.delete(1.0, tk.END)
    log_box.configure(state="disabled")

def start_process():
    input_path = input_var.get()
    output_path = output_var.get()

    if not validate_paths(input_path, output_path):
        return

    images = get_image_files(input_path)
    if not images:
        show_error("No Images", "No image files found in input folder.")
        return

    progress["maximum"] = len(images)
    progress["value"] = 0
    clear_log()
    log("🚀 Processing started...\n")

    def worker():
        try:
            count = select_best_images(
                input_path,
                output_path,
                log_callback=log,
                progress_callback=update_progress,
                show_benchmarks=False
            )
            log(f"\n✅ Done. {count} images selected.")
            messagebox.showinfo("Completed", f"Image selection completed.\n{count} image(s) saved.")
        except Exception as e:
            log(f"[ERROR] {e}")
            show_error("Error", str(e))

    threading.Thread(target=worker, daemon=True).start()

# --- GUI SETUP ---

def setup_gui():
    root.title("Wedding Image Selector")
    root.geometry("800x500")
    root.resizable(False, False)

    # Input Folder
    tk.Label(root, text="Input Folder").pack(anchor="w", padx=10, pady=(10, 0))
    tk.Entry(root, textvariable=input_var, width=70).pack(padx=10, fill="x")
    tk.Button(root, text="Browse", command=browse_input).pack(padx=10, pady=5, anchor="w")

    # Output Folder
    tk.Label(root, text="Output Folder").pack(anchor="w", padx=10)
    tk.Entry(root, textvariable=output_var, width=70).pack(padx=10, fill="x")
    tk.Button(root, text="Browse", command=browse_output).pack(padx=10, pady=5, anchor="w")

    # Start Button
    tk.Button(root, text="Start", command=start_process, bg="#4CAF50", fg="white", height=2).pack(pady=10)

    # Progress Bar
    progress.pack(pady=5)

    # Log Box
    tk.Label(root, text="Process Log:").pack(anchor="w", padx=10)
    log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

# --- MAIN EXECUTION ---

root = tk.Tk()
input_var = tk.StringVar()
output_var = tk.StringVar()
progress = ttk.Progressbar(root, mode="determinate", length=400)
log_box = tk.Text(root, state="disabled", wrap="word", height=15)

setup_gui()
root.mainloop()