# import necessary libraries
import os
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

# add parent directory and src directory to path for imports
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projectRoot)
sys.path.append(os.path.join(projectRoot, "src"))


class CodebusterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("codebuster")
        self.root.geometry("1200x800")

        # color scheme - clean white and blue
        self.bgColor = "#ffffff"
        self.secondaryBg = "#f8f9fa"
        self.accentColor = "#007bff"
        self.accentHover = "#0056b3"
        self.textColor = "#212529"
        self.titleColor = "#1a1a1a"
        self.successColor = "#28a745"
        self.errorColor = "#dc3545"
        self.codeBg = "#f5f5f5"

        # state tracking
        self.isAnalyzing = False

        # configure root background
        self.root.configure(bg=self.bgColor)

        # configure grid weights for resizing
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # header frame
        self.headerFrame = tk.Frame(root, bg=self.bgColor, height=100)
        self.headerFrame.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0
        )

        # title label
        self.titleLabel = tk.Label(
            self.headerFrame,
            text="Codebuster",
            font=("Helvetica Neue", 86, "bold"),
            bg=self.bgColor,
            fg=self.accentColor,
            pady=20,
        )
        self.titleLabel.pack()

        self.subtitleLabel = tk.Label(
            self.headerFrame,
            text="AI-Powered Code Similarity Detection",
            font=("Helvetica Neue", 13),
            bg=self.bgColor,
            fg=self.textColor,
        )
        self.subtitleLabel.pack(pady=(0, 15))

        # result frame
        self.resultFrame = tk.Frame(root, bg=self.bgColor, height=60)
        self.resultFrame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(10, 5)
        )

        self.resultLabel = tk.Label(
            self.resultFrame,
            text="",
            font=("Helvetica Neue", 16, "bold"),
            bg=self.bgColor,
            fg=self.textColor,
            pady=10,
        )
        self.resultLabel.pack()

        self.detailLabel = tk.Label(
            self.resultFrame,
            text="",
            font=("Helvetica Neue", 11),
            bg=self.bgColor,
            fg="#6c757d",
        )
        self.detailLabel.pack()

        # button frame
        self.buttonFrame = tk.Frame(root, bg=self.bgColor)
        self.buttonFrame.grid(row=2, column=0, columnspan=2, pady=15)

        # custom button using Frame and Label for better color support on macOS
        self.buttonContainer = tk.Frame(
            self.buttonFrame, bg=self.accentColor, cursor="hand2"
        )
        self.buttonContainer.pack()

        self.checkButton = tk.Label(
            self.buttonContainer,
            text="Analyze Code Similarity",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.accentColor,
            fg="white",
            padx=40,
            pady=15,
            cursor="hand2",
        )
        self.checkButton.pack()

        # bind click and hover effects
        self.checkButton.bind("<Button-1>", lambda _e: self.checkSimilarity())
        self.checkButton.bind("<Enter>", self.onButtonHover)
        self.checkButton.bind("<Leave>", self.onButtonLeave)
        self.buttonContainer.bind("<Enter>", self.onButtonHover)
        self.buttonContainer.bind("<Leave>", self.onButtonLeave)

        # left text box
        self.leftFrame = tk.Frame(root, bg=self.bgColor)
        self.leftFrame.grid(
            row=3, column=0, sticky="nsew", padx=(20, 10), pady=(10, 20)
        )
        self.leftFrame.grid_rowconfigure(1, weight=1)
        self.leftFrame.grid_columnconfigure(0, weight=1)

        self.leftLabel = tk.Label(
            self.leftFrame,
            text="Code Snippet 1",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.bgColor,
            fg=self.accentColor,
        )
        self.leftLabel.grid(row=0, column=0, pady=(0, 8))

        self.leftText = scrolledtext.ScrolledText(
            self.leftFrame,
            wrap=tk.NONE,
            width=40,
            height=20,
            font=("Consolas", 11),
            bg=self.codeBg,
            fg=self.textColor,
            insertbackground=self.accentColor,
            selectbackground=self.accentColor,
            selectforeground="white",
            bd=1,
            padx=15,
            pady=15,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#dee2e6",
            highlightcolor=self.accentColor,
        )
        self.leftText.grid(row=1, column=0, sticky="nsew")

        # right text box
        self.rightFrame = tk.Frame(root, bg=self.bgColor)
        self.rightFrame.grid(
            row=3, column=1, sticky="nsew", padx=(10, 20), pady=(10, 20)
        )
        self.rightFrame.grid_rowconfigure(1, weight=1)
        self.rightFrame.grid_columnconfigure(0, weight=1)

        self.rightLabel = tk.Label(
            self.rightFrame,
            text="Code Snippet 2",
            font=("Helvetica Neue", 14, "bold"),
            bg=self.bgColor,
            fg=self.accentColor,
        )
        self.rightLabel.grid(row=0, column=0, pady=(0, 8))

        self.rightText = scrolledtext.ScrolledText(
            self.rightFrame,
            wrap=tk.NONE,
            width=40,
            height=20,
            font=("Consolas", 11),
            bg=self.codeBg,
            fg=self.textColor,
            insertbackground=self.accentColor,
            selectbackground=self.accentColor,
            selectforeground="white",
            bd=1,
            padx=15,
            pady=15,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#dee2e6",
            highlightcolor=self.accentColor,
        )
        self.rightText.grid(row=1, column=0, sticky="nsew")

    # button hover effect
    def onButtonHover(self, _event):
        if not self.isAnalyzing:
            self.checkButton.config(bg=self.accentHover)
            self.buttonContainer.config(bg=self.accentHover)

    # button leave effect
    def onButtonLeave(self, _event):
        if not self.isAnalyzing:
            self.checkButton.config(bg=self.accentColor)
            self.buttonContainer.config(bg=self.accentColor)

    # validate input and launch ensemble analysis in a background thread
    def checkSimilarity(self):
        code1 = self.leftText.get("1.0", tk.END).strip()
        code2 = self.rightText.get("1.0", tk.END).strip()

        if not code1 or not code2:
            self.showErrorMessage()
            return

        self.isAnalyzing = True
        self.checkButton.config(text="Analyzing...", bg="#6c757d")
        self.buttonContainer.config(bg="#6c757d")
        self.resultLabel.config(text="Running analysis...", fg=self.textColor)
        self.detailLabel.config(text="This may take a moment while models load")

        thread = threading.Thread(
            target=self.runAnalysis, args=(code1, code2), daemon=True
        )
        thread.start()

    # background thread: run ensemble and post results back to UI
    def runAnalysis(self, code1, code2):
        try:
            result = self.runModel(code1, code2)
            self.root.after(0, self.showResults, result)
        except Exception as e:
            self.root.after(0, self.showError, str(e))

    # display analysis results in the UI
    def showResults(self, result):
        prob = result["probability"]
        pred = result["prediction"]

        if pred:
            self.resultLabel.config(
                text=f"Similarity: {prob * 100:.1f}% — Likely Similar",
                fg=self.errorColor,
            )
        else:
            self.resultLabel.config(
                text=f"Similarity: {prob * 100:.1f}% — Likely Different",
                fg=self.successColor,
            )

        detail = (
            f"Token: {result['token_similarity']:.3f}  |  "
            f"Semantic: {result['semantic_similarity']:.3f}  |  "
            f"Output: {result['output_similarity']:.3f}"
        )
        if result.get("errors"):
            detail += f"  [warnings: {len(result['errors'])}]"
        self.detailLabel.config(text=detail)

        self.resetButton()

    # display an error from analysis
    def showError(self, errorMsg):
        self.resultLabel.config(text=f"Error: {errorMsg}", fg=self.errorColor)
        self.detailLabel.config(text="")
        self.resetButton()

    # restore the analyze button to its default state
    def resetButton(self):
        self.isAnalyzing = False
        self.checkButton.config(text="Analyze Code Similarity", bg=self.accentColor)
        self.buttonContainer.config(bg=self.accentColor)

    # show error message for empty input
    def showErrorMessage(self):
        self.resultLabel.config(
            text="Please enter code in both text boxes",
            fg=self.errorColor,
            font=("Helvetica Neue", 14, "bold"),
        )
        self.detailLabel.config(text="")

    # run the ensemble model on two code snippets
    def runModel(self, code1, code2):
        from ensemble import analyze_pair

        # Declare the trained model path
        modelPath = "models/ensemble_model.pth"

        # use the default model path from ensemble module
        return analyze_pair(code1, code2, modelPath=modelPath)


# main function
def main():
    root = tk.Tk()
    app = CodebusterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
