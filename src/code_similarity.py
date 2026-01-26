# import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# function to compute cosine similarity between two code snippets
def compute_cosine_similarity(code1, code2) -> float:

    # Remove comments
    code1 = re.sub(r'//.*?$|/\*.*?\*/', '', code1, flags=re.DOTALL | re.MULTILINE)
    code2 = re.sub(r'//.*?$|/\*.*?\*/', '', code2, flags=re.DOTALL | re.MULTILINE)

    # Remove whitespace
    code1 = re.sub(r'\s+', ' ', code1).strip()
    code2 = re.sub(r'\s+', ' ', code2).strip()

    # Define custom token pattern for code
    tokenPattern = ""
    tokenPattern += r"[A-Za-z_][A-Za-z0-9_]*"  # Identifiers
    tokenPattern += r"|\d+"                     # Numbers
    tokenPattern += r"|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\|"  # Multi-char operators
    tokenPattern += r"|\".*?\"|\'.*?\'"  # String literals
    tokenPattern += r"|[{}()\[\];=+\-*/<>!&|]"  # Single-char operators and punctuation
    
    # Define stop words common in code
    stopWords = ["include", "namespace", "using", "std", "return", "cin", "cout", "int", "float", "double", "string", "bool", "endl"]
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(token_pattern=tokenPattern, ngram_range=(1, 2), stop_words=stopWords, norm='l2', sublinear_tf=True)
    tfidfMatrix = vectorizer.fit_transform([code1, code2])

    # Compute cosine similarity
    similarityMatrix = cosine_similarity(tfidfMatrix[0:1], tfidfMatrix[1:2])
    similarityValue = similarityMatrix[0][0]

    # Return similarity value
    return similarityValue

# function to process training data and compute similarity for each pair
def process_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['token_similarity'] = df.apply(lambda row: compute_cosine_similarity(row['code1'], row['code2']), axis=1)
    return df

def save_token_similarity(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    df = pd.read_parquet("data/train.parquet")
    df_processed = process_training_data(df)
    print(df_processed.head(10))
    save_token_similarity(df_processed, "csv_data/combined_scores.csv")