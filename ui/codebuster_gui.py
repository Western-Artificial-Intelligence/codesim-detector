import tkinter as tk
from tkinter import scrolledtext
import sys
import os

# Add parent directory to path to import similarity modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CodebusterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("codebuster")
        self.root.geometry("1200x800")

        # Color scheme - Clean white and blue
        self.bg_color = "#ffffff"
        self.secondary_bg = "#f8f9fa"
        self.accent_color = "#007bff"
        self.accent_hover = "#0056b3"
        self.text_color = "#212529"
        self.title_color = "#1a1a1a"
        self.success_color = "#28a745"
        self.error_color = "#dc3545"
        self.code_bg = "#f5f5f5"

        # State tracking
        self.is_analyzing = False

        # Configure root background
        self.root.configure(bg=self.bg_color)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Header Frame
        self.header_frame = tk.Frame(root, bg=self.bg_color, height=100)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        # Title Label
        self.title_label = tk.Label(
            self.header_frame,
            text="Codebuster",
            font=("Helvetica Neue", 86, "bold"),
            bg=self.bg_color,
            fg=self.accent_color,
            pady=20
        )
        self.title_label.pack()

        self.subtitle_label = tk.Label(
            self.header_frame,
            text="AI-Powered Code Similarity Detection",
            font=("Helvetica Neue", 13),
            bg=self.bg_color,
            fg=self.text_color
        )
        self.subtitle_label.pack(pady=(0, 15))

        # Result Frame
        self.result_frame = tk.Frame(root, bg=self.bg_color, height=60)
        self.result_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 5))

        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Helvetica Neue", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            pady=10
        )
        self.result_label.pack()

        # Button Frame
        self.button_frame = tk.Frame(root, bg=self.bg_color)
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=15)

        # Custom button using Frame and Label for better color support on macOS
        self.button_container = tk.Frame(
            self.button_frame,
            bg=self.accent_color,
            cursor="hand2"
        )
        self.button_container.pack()

        self.check_button = tk.Label(
            self.button_container,
            text="Analyze Code Similarity",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.accent_color,
            fg="white",
            padx=40,
            pady=15,
            cursor="hand2"
        )
        self.check_button.pack()

        # Bind click and hover effects
        self.check_button.bind("<Button-1>", lambda _e: self.check_similarity())
        self.check_button.bind("<Enter>", self.on_button_hover)
        self.check_button.bind("<Leave>", self.on_button_leave)
        self.button_container.bind("<Enter>", self.on_button_hover)
        self.button_container.bind("<Leave>", self.on_button_leave)

        # Left Text Box
        self.left_frame = tk.Frame(root, bg=self.bg_color)
        self.left_frame.grid(row=3, column=0, sticky="nsew", padx=(20, 10), pady=(10, 20))
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.left_label = tk.Label(
            self.left_frame,
            text="Code Snippet 1",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.left_label.grid(row=0, column=0, pady=(0, 8))

        self.left_text = scrolledtext.ScrolledText(
            self.left_frame,
            wrap=tk.NONE,
            width=40,
            height=20,
            font=("Consolas", 11),
            bg=self.code_bg,
            fg=self.text_color,
            insertbackground=self.accent_color,
            selectbackground=self.accent_color,
            selectforeground="white",
            bd=1,
            padx=15,
            pady=15,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#dee2e6",
            highlightcolor=self.accent_color
        )
        self.left_text.grid(row=1, column=0, sticky="nsew")

        # Right Text Box
        self.right_frame = tk.Frame(root, bg=self.bg_color)
        self.right_frame.grid(row=3, column=1, sticky="nsew", padx=(10, 20), pady=(10, 20))
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.right_label = tk.Label(
            self.right_frame,
            text="Code Snippet 2",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.bg_color,
            fg=self.accent_color
        )
        self.right_label.grid(row=0, column=0, pady=(0, 8))

        self.right_text = scrolledtext.ScrolledText(
            self.right_frame,
            wrap=tk.NONE,
            width=40,
            height=20,
            font=("Consolas", 11),
            bg=self.code_bg,
            fg=self.text_color,
            insertbackground=self.accent_color,
            selectbackground=self.accent_color,
            selectforeground="white",
            bd=1,
            padx=15,
            pady=15,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#dee2e6",
            highlightcolor=self.accent_color
        )
        self.right_text.grid(row=1, column=0, sticky="nsew")

    def on_button_hover(self, _event):
        """Button hover effect"""
        if not self.is_analyzing:
            self.check_button.config(bg=self.accent_hover)
            self.button_container.config(bg=self.accent_hover)

    def on_button_leave(self, _event):
        """Button leave effect"""
        if not self.is_analyzing:
            self.check_button.config(bg=self.accent_color)
            self.button_container.config(bg=self.accent_color)

    def check_similarity(self):
        # Placeholder, will be used to check similarity between both snippets
        return 0

    def show_error_message(self):
        """Show error message for empty input"""
        self.result_label.config(
            text="Please enter code in both text boxes",
            fg=self.error_color,
            font=("Helvetica Neue", 14, "bold")
        )

    def run_model(self, code1, code2):
        # Placeholder: return 0 for now - This will be replaced with actual model logic
        return 0


def main():
    root = tk.Tk()
    app = CodebusterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
