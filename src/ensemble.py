# import necessary libraries
import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

# add src directory to path for sibling imports
srcDir = os.path.dirname(os.path.abspath(__file__))
if srcDir not in sys.path:
    sys.path.insert(0, srcDir)

import code_similarity
import output_similarity
import semantic_similarity

# model path configuration
# modelPathName = "best_model.pth"
modelPathName = "practice_model.pth"

# default model path
defaultModelPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), modelPathName)


# MLP ensemble model
class EnsembleMLP(nn.Module):
    def __init__(self, inputSize=3, hiddenSize=16, outputSize=1):
        super(EnsembleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outputSize)
        )

    def forward(self, x):
        return self.network(x)

    def predictProba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# compute output similarity for a single pair of code snippets
def computeOutputSimilarityPair(code1, code2):
    with tempfile.TemporaryDirectory() as tmpdir:
        fName1 = os.path.join(tmpdir, "prog_1.cpp")
        fName2 = os.path.join(tmpdir, "prog_2.cpp")
        job = (fName1, fName2, code1, code2, 0)
        result = output_similarity.run(job, tmpdir)
        return result["output"]


# compute semantic similarity for a single pair of code snippets
def computeSemanticSimilarityPair(code1, code2):
    emb1 = semantic_similarity.compute_embedding(
        code1, layer_pooling=True, last_n_layers=4, max_length=64
    )
    emb2 = semantic_similarity.compute_embedding(
        code2, layer_pooling=True, last_n_layers=4, max_length=64
    )

    # convert torch tensors to numpy if needed
    if hasattr(emb1, 'numpy'):
        emb1 = emb1.numpy()
    if hasattr(emb2, 'numpy'):
        emb2 = emb2.numpy()

    sim = semantic_similarity.compute_semantic_similarity(emb1, emb2)

    # extract scalar from matrix if needed
    if hasattr(sim, 'shape') and sim.ndim > 0:
        sim = float(np.asarray(sim).flat[0])

    kw = semantic_similarity.keyword_overlap(code1, code2)
    return float(sim * kw)


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

    # use default model path if none provided
    if modelPath is None:
        modelPath = defaultModelPath

    if not os.path.exists(modelPath):
        raise FileNotFoundError(
            f"Model not found at {modelPath}. "
            "Please train the model first by running: python src/ensemble.py --train"
        )

    mlp = EnsembleMLP()
    mlp.load_state_dict(torch.load(modelPath, map_location='cpu', weights_only=True))
    mlp.eval()
    X = torch.FloatTensor(features)
    with torch.no_grad():
        prob = mlp.predictProba(X).squeeze().item()
    result["probability"] = prob
    result["prediction"] = int(prob >= 0.5)

    return result


# compute similarity features for a DataFrame with code1, code2 columns
def compute_features_for_dataframe(df):
    n = len(df)
    features = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        code1 = df['code1'].iloc[i]
        code2 = df['code2'].iloc[i]

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

    y = df['similar'].astype(np.float32).to_numpy()
    return features, y


# train the ensemble MLP model
def train_model(X, y, savePath=None, epochs=600, patience=10):
    if savePath is None:
        savePath = defaultModelPath

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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    bestValLoss = float('inf')
    patienceCount = 0

    for epoch in range(epochs):
        # training
        model.train()
        optimizer.zero_grad()
        logits = model(xTrain)
        loss = criterion(logits.squeeze(), yTrain)
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            valLogits = model(xVal)
            valLoss = criterion(valLogits.squeeze(), yVal)

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
            print(f'Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {valLoss.item():.4f}')

    # load best model
    model.load_state_dict(torch.load(savePath, map_location='cpu', weights_only=True))
    print(f"Model saved to {savePath}")
    return model


# main function
def main():
    parser = argparse.ArgumentParser(description="Train the ensemble model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    # find data file
    projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataPath = os.path.join(projectRoot, "csv_data", "sample_train.csv")

    if not os.path.exists(dataPath):
        print(f"Error: Data file not found at {dataPath}")
        return

    df = pd.read_csv(dataPath)

    argumentLimit = 200
    df = df.head(argumentLimit)
    print(f"Using first {argumentLimit} samples")

    df = df[['code1', 'code2', 'similar']].copy()

    X, y = compute_features_for_dataframe(df)

    outputPath = args.output if args.output else defaultModelPath
    train_model(X, y, savePath=outputPath)


if __name__ == "__main__":
    main()
