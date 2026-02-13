import os
from operator import index
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import RobertaModel, RobertaTokenizer
from tree_sitter_language_pack import get_language, get_parser


def extract_functions(code, parser):
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    functions = []

    def traverse(node):
        if node.type == "function_definition":
            start = node.start_byte
            end = node.end_byte
            func_code = code[start:end]
            functions.append(func_code)

        for child in node.children:
            traverse(child)

    traverse(root)
    return functions


def smart_chunk_code_with_lengths(code, tokenizer, parser, min_tokens=20):
    functions = extract_functions(code, parser)
    final_chunks, lengths = [], []
    for func in functions:
        tokens = tokenizer.tokenize(func)
        if len(tokens) < min_tokens:
            continue
        # handle long functions as before
        if len(tokens) <= 400:
            final_chunks.append(func)
            lengths.append(len(tokens))
        else:
            for i in range(0, len(tokens), 200):
                chunk = tokens[i : i + 400]
                if len(chunk) < min_tokens:
                    continue
                final_chunks.append(tokenizer.convert_tokens_to_string(chunk))
                lengths.append(len(chunk))
    return final_chunks, lengths


def encode_chunks(chunks, tokenizer, encoder, device="cuda"):
    if len(chunks) == 0:
        return None
    encoded = tokenizer(
        chunks, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    outputs = encoder(**encoded)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings


def similarity_matrix(A, B):
    # A: (n, d)
    # B: (m, d)
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    return torch.mm(A, B.T)  # (n, m)


def aggregate_topk(sim_matrix, k=3):
    flat = sim_matrix.view(-1)
    topk = torch.topk(flat, min(k, len(flat))).values
    return (torch.mean(topk) + 1) / 2.0


def softmax_weighted_aggregate(sim_matrix, temperature=0.45):
    """
    sim_matrix: (num_chunks1 x num_chunks2) cosine similarity
    temperature: smaller -> more weight on highest similarity
    """
    # Flatten all pair similarities
    flat = sim_matrix.view(-1)

    # Apply softmax weighting
    weights = F.softmax(flat / temperature, dim=0)
    score = torch.sum(weights * flat)

    # Optional: normalize to [0,1] if cosine similarities are in [-1,1]
    score = (score + 1.0) / 2.0
    score = torch.clamp(score, 0.0, 1.0)

    return score


def train_step(
    code1, code2, label, optimizer, encoder, tokenizer, parser, device="cuda"
):
    encoder.train()

    chunks1, _ = smart_chunk_code_with_lengths(code1, tokenizer, parser)
    chunks2, _ = smart_chunk_code_with_lengths(code2, tokenizer, parser)

    if len(chunks1) == 0 or len(chunks2) == 0:
        return None

    emb1 = encode_chunks(chunks1, tokenizer, encoder, device=device)
    emb2 = encode_chunks(chunks2, tokenizer, encoder, device=device)

    sim_matrix = similarity_matrix(emb1, emb2)
    pred_score = softmax_weighted_aggregate(sim_matrix, temperature=0.4)

    target = torch.tensor(label, dtype=torch.float, device=device)
    loss = F.mse_loss(pred_score, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(encoder, val_df, tokenizer, parser, device="cuda", temperature=0.45):
    encoder.eval()
    preds, labels = [], []

    with torch.no_grad():
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            code1, code2, label = row["code1"], row["code2"], row["similar"]

            chunks1, _ = smart_chunk_code_with_lengths(code1, tokenizer, parser)
            chunks2, _ = smart_chunk_code_with_lengths(code2, tokenizer, parser)
            if len(chunks1) == 0 or len(chunks2) == 0:
                continue

            emb1 = encode_chunks(chunks1, tokenizer, encoder, device=device)
            emb2 = encode_chunks(chunks2, tokenizer, encoder, device=device)

            sim_matrix = similarity_matrix(emb1, emb2)
            score = softmax_weighted_aggregate(sim_matrix, temperature=temperature)

            preds.append(score.item())
            labels.append(label)

    return preds, labels


def train_model(
    train_df, val_df, tokenizer, parser, epochs=3, lr=2e-5, device=None, temperature=0.4
):
    """
    Train the GraphCodeBERT encoder on code similarity task.

    Args:
        train_df: Training dataframe with columns 'code1', 'code2', 'similar'
        val_df: Validation dataframe with same columns
        epochs: Number of training epochs (default: 3)
        lr: Learning rate (default: 2e-5)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)

    Returns:
        encoder: Trained model
        history: Dictionary with training loss and validation AUC per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = RobertaModel.from_pretrained("microsoft/graphcodebert-base").to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=lr)

    history = {"train_loss": [], "val_auc": []}

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for i in range(len(train_df)):
            row = train_df.iloc[i]
            loss = train_step(
                row["code1"],
                row["code2"],
                row["similar"],
                optimizer,
                encoder,
                tokenizer,
                parser,
                device,
            )
            if loss is not None:
                total_loss += loss
                count += 1

        avg_loss = total_loss / count
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # --- Validation ---
        preds, labels = validate(
            encoder, val_df, tokenizer, parser, device=device, temperature=temperature
        )
        auc = roc_auc_score(labels, preds)
        history["val_auc"].append(auc)
        print(f"Epoch {epoch+1}: Validation ROC-AUC = {auc:.4f}")

    return encoder, history


def save_model(encoder, filepath="model/graphcodebert_mil_finetuned.pt"):
    torch.save(encoder.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath="model/graphcodebert_mil_finetuned.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
    encoder.load_state_dict(torch.load(filepath, map_location=device))
    encoder.to(device)
    encoder.eval()
    print(f"Model loaded from {filepath} on {device}")

    return encoder


def predict_similarity(
    code1,
    code2,
    encoder,
    tokenizer,
    parser,
    device,
    agg="softmax",
    temperature=0.45,
    topk=3,
):
    encoder.eval()

    with torch.no_grad():
        chunks1, _ = smart_chunk_code_with_lengths(code1, tokenizer, parser)
        chunks2, _ = smart_chunk_code_with_lengths(code2, tokenizer, parser)

        if len(chunks1) == 0 or len(chunks2) == 0:
            return 0.0

        emb1 = encode_chunks(chunks1, tokenizer, encoder, device=device)
        emb2 = encode_chunks(chunks2, tokenizer, encoder, device=device)

        sim_matrix = similarity_matrix(emb1, emb2)

        if agg == "topk":
            score = aggregate_topk(sim_matrix, k=topk)
        elif agg == "softmax":
            score = softmax_weighted_aggregate(sim_matrix, temperature=temperature)
        return score.item()


def get_evaluation_scores(
    encoder,
    val_df,
    tokenizer,
    parser,
    device=None,
    threshold=0.6,
    temperature=0.45,
    print_report=True,
):
    """Compute evaluation metrics on a validation dataframe.

    Expects val_df to have columns: 'code1', 'code2', 'similar'.

    Returns a dict with: auc, accuracy, precision, recall, f1, n, threshold.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    preds, labels = validate(
        encoder,
        val_df,
        tokenizer=tokenizer,
        parser=parser,
        device=device,
        temperature=temperature,
    )

    if len(preds) == 0:
        raise ValueError("No valid rows to evaluate (all rows produced empty chunks).")

    true_labels = [int(l) for l in labels]
    binary_preds = [1 if p >= threshold else 0 for p in preds]

    metrics = {
        "auc": roc_auc_score(true_labels, preds),
        "accuracy": accuracy_score(true_labels, binary_preds),
        "precision": precision_score(true_labels, binary_preds, zero_division=0),
        "recall": recall_score(true_labels, binary_preds, zero_division=0),
        "f1": f1_score(true_labels, binary_preds, zero_division=0),
        "n": len(preds),
        "threshold": threshold,
    }

    if print_report:
        print(f"Evaluation Metrics (n={metrics['n']}, threshold={threshold}):")
        print(f"ROC-AUC:   {metrics['auc']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")

    return metrics


def add_semantic_similarity_score_and_save(
    df,
    encoder,
    tokenizer,
    parser,
    output_csv_path,
    device=None,
    score_col="semantic_similarity",
    agg="softmax",
    temperature=0.45,
    topk=3,
    print_every=500,
):
    """Add a semantic similarity score column and save to CSV.

    Expects df to have columns: 'code1', 'code2'.
    Writes a CSV to output_csv_path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    encoder.eval()

    scores = []
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            score = predict_similarity(
                row["code1"],
                row["code2"],
                encoder=encoder,
                tokenizer=tokenizer,
                parser=parser,
                device=device,
                agg=agg,
                temperature=temperature,
                topk=topk,
            )
            scores.append(score)

            if print_every and (i + 1) % print_every == 0:
                print(f"Scored {i+1}/{len(df)} rows...")

    df_out = df.copy()
    df_out["semantic_similarity"] = scores

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Check if output file already exitsts
    if not Path(output_csv_path).exists():
        df_out.to_csv(output_csv_path, index=False)
    # If does exist add to an existing col
    else:
        new_df = pd.read_csv(output_csv_path)
        new_df["semantic_similarity"] = df_out["semantic_similarity"]
        new_df.to_csv(output_csv_path, index=False)

    print(f"Saved scored dataframe to {output_csv_path}")

    return df_out


if __name__ == "__main__":

    train_df = pd.read_parquet("data/train.parquet")
    val_df = pd.read_parquet("data/cross_validation.parquet")

    language = get_language("cpp")
    parser = get_parser("cpp")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # true only if you want to train the model
    want_to_train = True

    if want_to_train:  # Train the model

        model_train_df = train_df.sample(n=1000, random_state=42)
        model_val_df = val_df.sample(n=200, random_state=42)

        trained_encoder, training_history = train_model(
            model_train_df,
            model_val_df,
            tokenizer=tokenizer,
            parser=parser,
            epochs=3,
            lr=2e-5,
        )
        save_model(trained_encoder, filepath="model/graphcodebert_mil_finetuned.pt")
    else:  # Load pre-trained model
        trained_encoder = load_model(
            filepath="model/graphcodebert_mil_finetuned.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # Add semantic similarity score column to train_df and save
    add_semantic_similarity_score_and_save(
        train_df,
        encoder=trained_encoder,
        tokenizer=tokenizer,
        parser=parser,
        output_csv_path="csv_data/combined_scores.csv",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    get_evaluation_scores(
        trained_encoder,
        val_df,
        tokenizer=tokenizer,
        parser=parser,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.6,
    )
