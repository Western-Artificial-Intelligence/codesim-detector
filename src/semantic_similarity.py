# import necessary libraries
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np
import re
import pandas as pd
import time

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
model = RobertaModel.from_pretrained('microsoft/graphcodebert-base')
model.eval()

# function to compute keyword overlap
def keyword_overlap(code1, code2):
    keywords = ["for", "while", "if", "else", "return", "int", "float", "double", "string", "bool", "class", "def", "import", "include", "namespace", "using", "public", "private", "protected", "void", "static", "cin", "cout", "endl", "try", "catch", "switch", "case", "break", "continue"]
    
    k1 = set([k for k in keywords if k in code1])   # Extract keywords from code1
    k2 = set([k for k in keywords if k in code2])   # Extract keywords from code2

    # Avoid division by zero
    if not k1 or not k2:
        return 0.3
    
    # Compute Jaccard similarity
    return len(k1.intersection(k2)) / max(len(k1), len(k2))

# function to normalize code snippets
def perform_normalization(code):
    # Remove comments
    code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.DOTALL | re.MULTILINE)

    # Remove whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    return code

# function to compute code embeddings
def compute_embedding(codes, batch_size: int = 8):
    """Compute mean-pooled, L2-normalized embeddings for `codes`.

    Args:
        codes: a single string or an iterable/list of strings.
        batch_size: batch size to use for tokenization/model inference.
    """

    # handle single input case
    single_input = isinstance(codes, str)
    if single_input:
        codes = [codes]

    all_embs = []   # list to store embeddings
    
    # determine model device
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device('cpu')

    # process in batches
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True,
        )

        # move inputs to model device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # forward pass
        with torch.no_grad():
            out = model(**inputs)

        # mean pooling
        embeddings = out.last_hidden_state  # (batch, seq_len, hidden)
        attention_mask = inputs['attention_mask']

        # apply attention mask
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = (embeddings * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)

        # final mean-pooled embedding
        mean_pooled = summed / counted
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        # append to all_embs
        all_embs.append(mean_pooled.cpu())

    # concatenate all embeddings
    if all_embs:
        result = torch.cat(all_embs, dim=0)
    else:
        result = torch.empty((0, model.config.hidden_size))

    # return single embedding if single input
    if single_input:
        return result.squeeze(0)
    return result

# function to compute semantic similarity between two code embeddings
def compute_semantic_similarity(vec1, vec2):

    v1 = vec1.detach()
    v2 = vec2.detach()

    # normalize along last dimension
    v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)
    v2 = torch.nn.functional.normalize(v2, p=2, dim=-1)

    # single vector vs single vector -> scalar
    if v1.dim() == 1 and v2.dim() == 1:
        return float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item())

    # per-row similarity when both are (N, d)
    if v1.dim() == 2 and v2.dim() == 2 and v1.size(0) == v2.size(0):
        return torch.nn.functional.cosine_similarity(v1, v2, dim=1)

    # general case: (N, d) x (M, d) -> (N, M) matrix
    return torch.matmul(v1, v2.t())

# function to process training data and compute similarity for each pair
def process_training_data(df, pair_batch_size: int = 64, embed_batch_size: int = 8):
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

    # iterate over pairs in chunks
    for start in range(0, n, pair_batch_size):
        end = min(start + pair_batch_size, n)

        # normalize code snippets in the chunk
        codes1_chunk = df['code1'].iloc[start:end].astype(str).map(perform_normalization).tolist()
        codes2_chunk = df['code2'].iloc[start:end].astype(str).map(perform_normalization).tolist()

        # embed each chunk (embedding function itself can batch internally)
        emb1 = compute_embedding(codes1_chunk, batch_size=embed_batch_size)
        emb2 = compute_embedding(codes2_chunk, batch_size=embed_batch_size)

        # compute semantic similarities for the chunk
        sims = compute_semantic_similarity(emb1, emb2)

        # Normalize sims to 1D numpy array for the chunk
        if isinstance(sims, float) or np.isscalar(sims):
            sims_arr = np.full(end - start, float(sims), dtype=float)
        elif isinstance(sims, torch.Tensor):
            sims_cpu = sims.cpu()
            if sims_cpu.dim() == 1:
                sims_arr = sims_cpu.numpy()
            elif sims_cpu.dim() == 2 and sims_cpu.size(0) == (end - start) and sims_cpu.size(1) == (end - start):
                # per-row pairs represented as square matrix; take diagonal
                sims_arr = sims_cpu.diag().numpy()
            else:
                # if we got an (N, M) matrix where M != N, we cannot reduce to per-row similarity
                raise ValueError(f'Cannot reduce similarity matrix of shape {tuple(sims_cpu.shape)} to per-row similarities for chunk {start}:{end}')
        else:
            sims_arr = np.array(sims, dtype=float)

        # compute keyword overlap for this chunk
        kw_chunk = np.array([
            keyword_overlap(a, b)
            for a, b in zip(df['code1'].iloc[start:end], df['code2'].iloc[start:end])
        ], dtype=float)

        results[start:end] = sims_arr * kw_chunk

    df['semantic_similarity'] = results
    return df

# Main execution
if __name__ == "__main__":
    df = pd.read_csv("csv_data/sample_train.csv")
    s_time = time.time()
    df_processed = process_training_data(df[['code1', 'code2', 'similar']][:100])
    print(df_processed.head(10))
    print(f"Processing time: {time.time() - s_time} seconds")