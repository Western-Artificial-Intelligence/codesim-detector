# import necessary libraries
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# add src directory to path for sibling imports
srcDir = os.path.dirname(os.path.abspath(__file__))
if srcDir not in sys.path:
    sys.path.insert(0, srcDir)

import code_similarity
import output_similarity
import semantic_similarity_function_chunking as semantic_similarity

# model path configuration
modelPathName = "ensemble_model.pth"

# default model path
defaultModelPath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    modelPathName,
)

# legacy path (older repo layout)
legacyModelPath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    modelPathName,
)


# MLP ensemble model
class EnsembleMLP(nn.Module):
    def __init__(self, inputSize=3, hiddenSize=16, outputSize=1):
        super(EnsembleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outputSize),
        )

    def forward(self, x):
        return self.network(x)

    def predictProba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def _resolve_model_path(modelPath=None):
    if modelPath is None:
        modelPath = defaultModelPath
    if not os.path.exists(modelPath) and os.path.exists(legacyModelPath):
        modelPath = legacyModelPath
    return modelPath


def load_trained_model(modelPath=None, device="cpu"):
    modelPath = _resolve_model_path(modelPath)
    if not os.path.exists(modelPath):
        raise FileNotFoundError(
            f"Model not found at {modelPath}. "
            "Train first with: python src/ensemble.py --train"
        )

    mlp = EnsembleMLP()
    try:
        stateDict = torch.load(modelPath, map_location="cpu", weights_only=True)
    except TypeError:
        stateDict = torch.load(modelPath, map_location="cpu")
    mlp.load_state_dict(stateDict)
    mlp.to(device)
    mlp.eval()
    return mlp, modelPath


# compute output similarity for a single pair of code snippets
def computeOutputSimilarityPair(code1, code2):
    with tempfile.TemporaryDirectory() as tmpdir:
        fName1 = os.path.join(tmpdir, "prog_1.cpp")
        fName2 = os.path.join(tmpdir, "prog_2.cpp")
        result = output_similarity.run(fName1, fName2, code1, code2, tmpdir)
        return result

def computeSemanticSimilarityPair(code1, code2):
    """
    Computes semantic similarity in [0,1] using src/semantic_similarity_function_chunking.py
    (GraphCodeBERT + function-aware chunking + softmax-weighted aggregation).

    Caches model/tokenizer/parser after the first call.
    """
    # lazy init + cache to avoid re-loading model every pair
    if not hasattr(computeSemanticSimilarityPair, "_ctx"):
        projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        modelPath = os.path.join(projectRoot, "model", "graphcodebert_mil_finetuned.pt")
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"Semantic model not found at {modelPath}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        parser = semantic_similarity.get_parser("cpp")
        tokenizer = semantic_similarity.RobertaTokenizer.from_pretrained(
            "microsoft/graphcodebert-base"
        )
        encoder = semantic_similarity.load_model(filepath=modelPath, device=device)

        computeSemanticSimilarityPair._ctx = {
            "device": device,
            "parser": parser,
            "tokenizer": tokenizer,
            "encoder": encoder,
        }

    ctx = computeSemanticSimilarityPair._ctx

    score = semantic_similarity.predict_similarity(
        code1,
        code2,
        encoder=ctx["encoder"],
        tokenizer=ctx["tokenizer"],
        parser=ctx["parser"],
        device=ctx["device"],
        agg="softmax",
        temperature=0.45,
    )
    return float(score)

# # compute semantic similarity for a single pair of code snippets
# def computeSemanticSimilarityPair(code1, code2):
#     emb1 = semantic_similarity.compute_embedding(
#         code1, layer_pooling=True, last_n_layers=4, max_length=64
#     )
#     emb2 = semantic_similarity.compute_embedding(
#         code2, layer_pooling=True, last_n_layers=4, max_length=64
#     )

#     # convert torch tensors to numpy if needed
#     if hasattr(emb1, "numpy"):
#         emb1 = emb1.numpy()
#     if hasattr(emb2, "numpy")
#         emb2 = emb2.numpy()

#     sim = semantic_similarity.compute_semantic_similarity(emb1, emb2)

#     # extract scalar from matrix if needed
#     if hasattr(sim, "shape") and sim.ndim > 0:
#         sim = float(np.asarray(sim).flat[0])

#     kw = semantic_similarity.keyword_overlap(code1, code2)
#     return float(sim * kw)


# run the full ensemble analysis on a pair of code snippets
def analyze_pair(code1, code2, modelPath=None):
    errors = []

    # token similarity
    try:
        tokenSim = float(code_similarity.compute_cosine_similarity(code1, code2))
    except Exception as e:
        tokenSim = 0.0
        errors.append(f"token: {e}")

    # semantic similarity
    try:
        semanticSim = computeSemanticSimilarityPair(code1, code2)
    except Exception as e:
        semanticSim = 0.0
        errors.append(f"semantic: {e}")

    # output similarity
    try:
        outputSim = float(computeOutputSimilarityPair(code1, code2))
    except Exception as e:
        outputSim = 0.0
        errors.append(f"output: {e}")

    result = {
        "token_similarity": tokenSim,
        "semantic_similarity": semanticSim,
        "output_similarity": outputSim,
    }

    if errors:
        result["errors"] = errors

    features = np.array([[tokenSim, semanticSim, outputSim]], dtype=np.float32)

    mlp, modelPathResolved = load_trained_model(modelPath=modelPath, device="cpu")
    X = torch.FloatTensor(features)
    with torch.no_grad():
        prob = mlp.predictProba(X).squeeze().item()
    result["probability"] = prob
    result["prediction"] = int(prob >= 0.5)
    result["model_path"] = modelPathResolved

    return result


# compute similarity features for a DataFrame with code1, code2 columns
def compute_features_for_dataframe(df):
    n = len(df)
    features = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        code1 = df["code1"].iloc[i]
        code2 = df["code2"].iloc[i]

        # token similarity
        try:
            tokenSim = float(code_similarity.compute_cosine_similarity(code1, code2))
        except Exception:
            tokenSim = 0.0

        # semantic similarity
        try:
            semanticSim = float(computeSemanticSimilarityPair(code1, code2))
        except Exception:
            semanticSim = 0.0

        # output similarity
        try:
            outputSim = float(computeOutputSimilarityPair(code1, code2))
        except Exception:
            outputSim = 0.0

        features[i] = [tokenSim, semanticSim, outputSim]

    y = df["similar"].astype(np.float32).to_numpy()
    return features, y


# train the ensemble MLP model
def train_model(
    X,
    y,
    savePath=None,
    epochs=50,
    patience=10,
    batchSize=4096,
    learningRate=1e-3,
    device=None,
):
    if savePath is None:
        savePath = defaultModelPath

    os.makedirs(os.path.dirname(savePath), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # split data
    xTrain, xVal, yTrain, yVal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xTrain = torch.FloatTensor(xTrain)
    yTrain = torch.FloatTensor(yTrain)
    xVal = torch.FloatTensor(xVal)
    yVal = torch.FloatTensor(yVal)

    model = EnsembleMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    model.to(device)

    trainLoader = DataLoader(
        TensorDataset(xTrain, yTrain), batch_size=batchSize, shuffle=True
    )
    valLoader = DataLoader(
        TensorDataset(xVal, yVal), batch_size=batchSize, shuffle=False
    )

    bestValLoss = float("inf")
    patienceCount = 0

    for epoch in range(epochs):
        # training
        model.train()
        trainLossSum = 0.0
        trainCount = 0
        for xb, yb in trainLoader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batchSizeActual = int(yb.shape[0])
            trainLossSum += loss.item() * batchSizeActual
            trainCount += batchSizeActual

        trainLoss = trainLossSum / max(trainCount, 1)

        # validation
        model.eval()
        with torch.no_grad():
            valLossSum = 0.0
            valCount = 0
            for xb, yb in valLoader:
                xb = xb.to(device)
                yb = yb.to(device)
                valLogits = model(xb).squeeze(1)
                valLossBatch = criterion(valLogits, yb)
                batchSizeActual = int(yb.shape[0])
                valLossSum += valLossBatch.item() * batchSizeActual
                valCount += batchSizeActual

            valLoss = valLossSum / max(valCount, 1)

        # early stopping
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            patienceCount = 0
            torch.save(model.state_dict(), savePath)
        else:
            patienceCount += 1

        if patienceCount >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}"
            )

    # load best model
    try:
        bestStateDict = torch.load(savePath, map_location="cpu", weights_only=True)
    except TypeError:
        bestStateDict = torch.load(savePath, map_location="cpu")
    model.load_state_dict(bestStateDict)
    print(f"Model saved to {savePath}")
    return model


def _load_training_dataframe(projectRoot, use="both", limit=None):
    csvDir = os.path.join(projectRoot, "csv_data")
    trainPath = os.path.join(csvDir, "train_combined_scores.csv")
    cvPath = os.path.join(csvDir, "cross_validation_combined_scores.csv")

    toLoad = []
    if use in ("train", "both"):
        toLoad.append(("train", trainPath))
    if use in ("cv", "both"):
        toLoad.append(("cv", cvPath))

    missing = [p for _, p in toLoad if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing CSV file(s): "
            + ", ".join(missing)
            + ". Expected under csv_data/."
        )

    dfs = []
    for label, path in toLoad:
        dfPart = pd.read_csv(path)
        if limit is not None:
            dfPart = dfPart.head(int(limit))
        print(f"Loaded {len(dfPart):,} rows from {label}: {os.path.relpath(path, projectRoot)}")
        dfs.append(dfPart)

    df = pd.concat(dfs, ignore_index=True)
    required = {"similar", "token_similarity", "semantic_similarity", "output_similarity"}
    missingCols = sorted(required - set(df.columns))
    if missingCols:
        raise ValueError(f"Training CSV(s) missing required column(s): {missingCols}")

    return df


def evaluate_precision_matrix(
    testCsvPath,
    modelPath=None,
    threshold=0.5,
    batchSize=8192,
    device="cpu",
    saveImage=True,
    outDir=None,
    outFileName="precision_matrix.png",
):
    if not os.path.exists(testCsvPath):
        raise FileNotFoundError(f"Test CSV not found at {testCsvPath}")

    df = pd.read_csv(testCsvPath)
    required = {"similar", "token_similarity", "semantic_similarity", "output_similarity"}
    missingCols = sorted(required - set(df.columns))
    if missingCols:
        raise ValueError(f"Test CSV missing required column(s): {missingCols}")

    X = (
        df[["token_similarity", "semantic_similarity", "output_similarity"]]
        .astype(np.float32)
        .to_numpy()
    )
    yTrue = df["similar"].astype(np.int64).to_numpy()

    mlp, modelPathResolved = load_trained_model(modelPath=modelPath, device=device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)), batch_size=batchSize, shuffle=False
    )

    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pb = mlp.predictProba(xb).squeeze(1).detach().cpu().numpy()
            probs.append(pb)

    yProb = np.concatenate(probs, axis=0)
    yPred = (yProb >= float(threshold)).astype(np.int64)

    cm = confusion_matrix(yTrue, yPred, labels=[0, 1])
    cmDf = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

    report = classification_report(yTrue, yPred, output_dict=True, zero_division=0)
    reportDf = pd.DataFrame(report).transpose()

    print(f"Model: {modelPathResolved}")
    print(f"Test CSV: {testCsvPath}")
    print(f"Threshold: {threshold}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cmDf)
    print("\nClassification report:")
    # Keep it compact; show main rows if present
    for key in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
        if key in reportDf.index:
            print(f"\n[{key}]")
            print(reportDf.loc[[key]])

    imagePath = None
    if saveImage:
        if outDir is None:
            outDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

        os.makedirs(outDir, exist_ok=True)
        imagePath = os.path.join(outDir, outFileName)

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required to save the precision/confusion matrix image. "
                "Install it with: pip install matplotlib"
            ) from e

        accuracy = float(report.get("accuracy", float("nan")))

        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(9.2, 4.2),
            gridspec_kw={"width_ratios": [1.0, 1.25]},
        )

        im = ax1.imshow(cm, cmap="Blues")
        ax1.set_title("Confusion matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["0", "1"])
        ax1.set_yticklabels(["0", "1"])

        for (i, j), v in np.ndenumerate(cm):
            ax1.text(j, i, str(int(v)), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2.axis("off")
        rows = ["0", "1", "macro avg", "weighted avg"]
        cols = ["precision", "recall", "f1-score", "support"]
        reportCompact = reportDf.loc[[r for r in rows if r in reportDf.index], cols].copy()

        def _fmt(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                # support is float in sklearn report; keep it integer-like
                if float(x).is_integer():
                    return str(int(x))
                return f"{float(x):.3f}"
            return str(x)

        cellText = [[_fmt(v) for v in row] for row in reportCompact.to_numpy()]
        rowLabels = list(reportCompact.index)
        colLabels = list(reportCompact.columns)

        table = ax2.table(
            cellText=cellText,
            rowLabels=rowLabels,
            colLabels=colLabels,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.3)

        ax2.set_title(f"Precision/Recall/F1 (acc={accuracy:.3f})\nthreshold={threshold}")

        fig.tight_layout()
        fig.savefig(imagePath, dpi=200)
        plt.close(fig)
        print(f"\nSaved matrix image to: {imagePath}")

    return {
        "model_path": modelPathResolved,
        "test_csv": testCsvPath,
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
        "image_path": imagePath,
    }



# main function
def main():
    parser = argparse.ArgumentParser(description="Train the ensemble model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--precision-matrix",
        action="store_true",
        help="Evaluate on csv_data/test_combined_scores.csv and print confusion/precision metrics",
    )
    parser.add_argument(
        "--use",
        choices=["train", "cv", "both"],
        default="both",
        help="Which CSV(s) to use from csv_data/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit applied per CSV before concatenation",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device (auto picks cuda if available)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for test evaluation",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Optional path to test_combined_scores.csv (defaults to csv_data/test_combined_scores.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional model path override (defaults to model/ensemble_model.pth)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8192,
        help="Batch size for test evaluation",
    )
    parser.add_argument(
        "--no-save-matrix",
        action="store_true",
        help="Do not save the confusion/precision matrix PNG (only print metrics)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for saved matrix image (default: <projectRoot>/results)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="precision_matrix.png",
        help="Output image filename (default: precision_matrix.png)",
    )
    args = parser.parse_args()

    if not args.train and not args.precision_matrix:
        parser.print_help()
        return

    projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.train:
        df = _load_training_dataframe(projectRoot, use=args.use, limit=args.limit)

        # IMPORTANT: feature order must match analyze_pair(): [token, semantic, output]
        X = (
            df[["token_similarity", "semantic_similarity", "output_similarity"]]
            .astype(np.float32)
            .to_numpy()
        )
        y = df["similar"].astype(np.float32).to_numpy()

        outputPath = defaultModelPath
        trainDevice = None if args.device == "auto" else args.device

        train_model(
            X,
            y,
            savePath=outputPath,
            epochs=args.epochs,
            patience=args.patience,
            batchSize=args.batch_size,
            learningRate=args.lr,
            device=trainDevice,
        )

    if args.precision_matrix:
        testPath = (
            args.test_path
            if args.test_path is not None
            else os.path.join(projectRoot, "csv_data", "test_combined_scores.csv")
        )
        evalDevice = "cpu" if args.device == "auto" else args.device
        evaluate_precision_matrix(
            testCsvPath=testPath,
            modelPath=args.model_path,
            threshold=args.threshold,
            batchSize=args.eval_batch_size,
            device=evalDevice,
            saveImage=(not args.no_save_matrix),
            outDir=args.out_dir,
            outFileName=args.out_file,
        )


if __name__ == "__main__":
    main()
