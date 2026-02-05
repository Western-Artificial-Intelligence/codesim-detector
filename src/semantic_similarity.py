# import necessary libraries
from transformers import RobertaModel, RobertaTokenizer
from tqdm.auto import tqdm
import torch
import numpy as np
import re
import pandas as pd
import time
from typing import Iterable, Optional

# set device and configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_fp16 = torch.cuda.is_available() and device.type == 'cuda'
last_n_layers_default = 4
kw_default = 0.3
default_layer_pooling = False
combine_method_default = 'prod'

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
# load model with hidden states enabled for optional layer pooling
model = RobertaModel.from_pretrained('microsoft/graphcodebert-base', output_hidden_states=True)
model.to(device)
model.eval()

# function to compute keyword overlap
def keyword_overlap(code1, code2):
    keywords = [
        "for", "while", "if", "else", "return", "int", "float", "double",
        "string", "bool", "class", "def", "import", "include", "namespace",
        "using", "public", "private", "protected", "void", "static", "try",
        "catch", "switch", "case", "break", "continue"
    ]    

    k1 = set([k for k in keywords if k in code1])   # Extract keywords from code1
    k2 = set([k for k in keywords if k in code2])   # Extract keywords from code2

    # Avoid division by zero
    if not k1 or not k2:
        return kw_default
    
    # Compute Jaccard similarity
    return len(k1.intersection(k2)) / max(len(k1), len(k2))

# function to normalize code snippets
def perform_normalization(code: str) -> str:
    # Remove comments and collapse whitespace
    code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)

    # remove common preprocessor/import lines
    lines = code.splitlines()
    kept = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include'):
            continue
        if stripped.startswith('using namespace'):
            continue
        if stripped.startswith('typedef'):
            continue
        if stripped.startswith('#define'):
            continue

        if not stripped:
            continue
        kept.append(stripped)

    code = "\n".join(kept)

    # compress whitespace
    code = re.sub(r'\s+', ' ', code).strip()

    return code


# function for mean pooling
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# function for layer-averaged pooling
def layer_average_pooling(hidden_states, attention_mask, last_n_layers):
    layers = hidden_states[-last_n_layers:]
    pooled_layers = []
    for layer in layers:
        pooled_layer = mean_pooling(layer, attention_mask)
        pooled_layers.append(pooled_layer)
    return torch.stack(pooled_layers, dim=0).mean(dim=0)

# function to compute code embeddings
def compute_embedding(codes: Iterable[str], batch_size: int = 32, max_length: int = 64, normalize: bool = True,
                      layer_pooling: Optional[bool] = None, last_n_layers: int = 4):
    """Compute embeddings for `codes` with optional layer-averaged pooling.

    Args:
        codes: iterable of strings (or a single string will be supported by caller).
        batch_size: tokenization / model batch size.
        max_length: tokenizer max length.
        layer_pooling: if True, average the last `last_n_layers` hidden states (mean pooling per layer then average).
        last_n_layers: number of last layers to average when `layer_pooling` is True.
    Returns:
        Torch tensor of shape (N, hidden_size) on CPU (L2-normalized rows).
    """
    # allow single string too
    single = False
    single_input = isinstance(codes, str)
    if single_input:
        single = True
        codes = [codes]

    all_embs = []

    if layer_pooling is None:
        layer_pooling = default_layer_pooling

    for i in range(0, len(codes), batch_size):
        batch_texts = [perform_normalization(c) for c in codes[i:i + batch_size]]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    out = model(**inputs)
            else:
                out = model(**inputs)

        
        if layer_pooling:
            hidden_states = out.hidden_states
            embeddings = layer_average_pooling(hidden_states, inputs['attention_mask'], last_n_layers)
        else:
            embeddings = mean_pooling(out.last_hidden_state, inputs['attention_mask'])


        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embs.append(embeddings.cpu())
                

    if all_embs:
        result = torch.cat(all_embs, dim=0).numpy()
    else:
        result = np.zeros((0, model.config.hidden_size), dtype=np.float32)

    if single:
        return embeddings[0]
    return result

# function to compute semantic similarity between two code embeddings
def compute_semantic_similarity(vec1, vec2):

    # for 1d scalars
    if vec1.ndim == 1 and vec2.ndim == 1:
        v1 = vec1.astype(np.float32)
        v2 = vec2.astype(np.float32)

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator == 0:
            return 0.0
        return float(np.dot(v1, v2) / denominator)
    

    # for 2d arrays
    if vec1.ndim == 1:
        vec1 = vec1[None, :]
    if vec2.ndim == 1:
        vec2 = vec2[None, :]


    v1 = vec1.astype(np.float32)
    v2 = vec2.astype(np.float32)

    v1_norms = np.linalg.norm(v1, axis=1, keepdims=True).clip(min=1e-9)
    v2_norms = np.linalg.norm(v2, axis=1, keepdims=True).clip(min=1e-9)

    v1 = v1 / v1_norms
    v2 = v2 / v2_norms

    # general case: (N, d) x (M, d) -> (N, M) matrix
    return np.matmul(v1, v2.T)

# function to process training data and compute similarity for each pair
def process_training_data(df, pair_batch_size: int = 128, embed_batch_size: int = 32, combine_method: Optional[str] = None, alpha: float = 0.85, rescale: bool = False):
    """Compute semantic similarity for each row in `df` by processing pairs in chunks.

    Args:
        df: DataFrame with columns `code1` and `code2`.
        pair_batch_size: number of pairs to process in each chunk (controls memory footprint).
        embed_batch_size: batch size passed to `compute_embedding` for tokenization/model batching.

    Returns:
        Copy of `df` with a new column `semantic_similarity`.
    """

    df = df.copy()
    n = len(df)
    if n == 0:
        df['semantic_similarity'] = []
        return df

    results = np.zeros(n, dtype=float)

    effective_combine = combine_method if combine_method is not None else combine_method_default

    # iterate over pairs in chunks
    for start in tqdm(range(0, n, pair_batch_size), desc="pair-chunks"):
        end = min(start + pair_batch_size, n)

        # normalize code snippets in the chunk
        codes1_chunk = df['code1'].iloc[start:end].astype(str).tolist()
        codes2_chunk = df['code2'].iloc[start:end].astype(str).tolist()

        # embed each chunk (embedding function itself can batch internally)
        # enable layer pooling and match optimized defaults for parity
        emb1 = compute_embedding(codes1_chunk, batch_size=embed_batch_size, layer_pooling=True, last_n_layers=4, max_length=64)
        emb2 = compute_embedding(codes2_chunk, batch_size=embed_batch_size, layer_pooling=True, last_n_layers=4, max_length=64)

        # compute semantic similarities for the chunk
        sims = compute_semantic_similarity(emb1, emb2)

        if sims.shape[0] == sims.shape[1]:
            sims = np.diagonal(sims)
        else:
            # If pair counts match, prefer diagonal; else use row-wise max
            if sims.shape[0] == (end - start) and sims.shape[1] == (end - start):
                sims = np.diag(sims)
            else:
                sims = np.max(sims, axis=1)

        kw = np.array([keyword_overlap(a, b) for a, b in zip(codes1_chunk, codes2_chunk)], dtype=float)

        print(f"Semantic sims (first 10): {sims[:10]}")
        print(f"Keyword overlaps (first 10): {kw[:10]}")
        if effective_combine == 'prod':
            combined = sims * kw
        elif effective_combine == 'avg':
            combined = 0.5 * (sims + kw)
        elif effective_combine == 'weighted':
            combined = alpha * sims + (1 - alpha) * kw
        else:
            raise ValueError(f'Unknown combine method: {effective_combine}')
        
        if rescale:
            combined = np.clip(combined, -1.0, 1.0)
            combined = (combined + 1.0) / 2.0

        results[start:end] = combined

    df['semantic_similarity'] = results
    return df

# Main execution
if __name__ == "__main__":
    # Load training data from Parquet instead of CSV
    # Expected path: ../data/sample_train.parquet
    df = pd.read_parquet("../data/train.parquet")
    s_time = time.time()
    df_processed = process_training_data(df.head(100), pair_batch_size=128, embed_batch_size=32)
    print(df_processed.head(20))
    print(f"Processing time: {time.time() - s_time} seconds")
