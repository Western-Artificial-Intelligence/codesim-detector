# import necessary libraries
from transformers import RobertaModel, RobertaTokenizer
import torch
import re
import pandas as pd

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
def compute_embedding(code: str):

    '''
    Compute the code embedding using GraphCodeBERT.
    Returns a normalized mean pooled embedding.
    '''

    # Tokenize input code
    inputs = tokenizer(
        code,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )

    # Get model outputs without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings and attention mask
    embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
    attention_mask = inputs['attention_mask']

    # Mean pooling
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
    masked_embeddings = embeddings * mask
    summed = masked_embeddings.sum(dim=1)
    counted = mask.sum(dim=1)

    # calculate mean pooled embeddings
    mean_pooled = summed / counted
    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
    return mean_pooled

# function to compute semantic similarity between two code embeddings
def compute_semantic_similarity(vec1, vec2) -> float:
    return torch.nn.functional.cosine_similarity(vec1, vec2).item()

# function to process training data and compute similarity for each pair
def process_training_data(df):
    df = df.copy()
    df['semantic_similarity'] = df.apply(lambda row: compute_semantic_similarity(
        compute_embedding(perform_normalization(row['code1'])),
        compute_embedding(perform_normalization(row['code2']))
    ) * keyword_overlap(row['code1'], row['code2']), axis=1)
    return df

# Main execution
if __name__ == "__main__":
    df = pd.read_csv("csv_data/sample_train.csv")
    df_processed = process_training_data(df[['code1', 'code2', 'similar']][:10])
    print(df_processed.head(10))