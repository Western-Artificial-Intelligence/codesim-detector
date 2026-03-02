from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# load the sample training data from csv file
csvDataPath = "csv_data/sample_train.csv"
if not Path(csvDataPath).exists():
    raise FileNotFoundError(f"CSV data file not found: {csvDataPath}")
df = pd.read_csv(csvDataPath)
print('Loaded df shape:', df.shape)
print(df.head(5))

# load the sample generated training data from csv file
gen_csvDataPath = "csv_data/generated_pairs_train.csv"
if not Path(gen_csvDataPath).exists():
    raise FileNotFoundError(f"CSV data file not found: {gen_csvDataPath}")
gen_df = pd.read_csv(gen_csvDataPath)
print('Loaded generated df shape:', gen_df.shape)
print(gen_df.head(5))


# select only the relevant columns from the training csv
selected_cols = ['code1', 'code2', 'similar']
df = df[selected_cols]
print(df.head(5))

# added labels to generated training data which allows it to be merged with sample pairs
gen_df.columns = selected_cols
print(gen_df.head(5))

# Merge the generated pairs and sample pairs
df = pd.concat([df, gen_df], axis=0)
print('Final training df shape:', df.shape)
print(df.head(5))

# save the sample data to parquet
Path("data").mkdir(parents=True, exist_ok=True)
parquetFilePath = "data/sample_data.parquet"
df.to_parquet(parquetFilePath, engine='pyarrow', index=False)
print(f"Saved sample parquet to: {parquetFilePath}")


# reload the parquet to verify
df = pd.read_parquet(parquetFilePath)
print('\nReloaded parquet, shape:', df.shape)
print(df.head(5))


# split the merged labeled data into train/cv/test: 70% / 15% / 15%
# (do this in two stages so CV and test are equal-sized)
labels = df["similar"]

trainData, tempData = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=labels,
)
crossValidationData, testData = train_test_split(
    tempData,
    test_size=0.50,
    random_state=42,
    stratify=tempData["similar"],
)

print(
    "\nSplit sizes:",
    "train=", len(trainData),
    "cv=", len(crossValidationData),
    "test=", len(testData),
)

# save splits to parquet
trainData.to_parquet("data/train.parquet", engine="pyarrow", index=False)
crossValidationData.to_parquet(
    "data/cross_validation.parquet", engine="pyarrow", index=False
)
print("Saved train/cross-validation parquet files to data/.")


# --- Build labeled evaluation test set ---
# Start with the 15% held-out labeled split, then append generated labeled test pairs.
test_labeled_parts = [testData]

gen_csvTestDataPath = "csv_data/generated_pairs_test.csv"
if Path(gen_csvTestDataPath).exists():
    gen_test_df = pd.read_csv(gen_csvTestDataPath)
    test_labeled_parts.append(gen_test_df[selected_cols])
else:
    raise FileNotFoundError(f"CSV generated test data file not found: {gen_csvTestDataPath}")

testLabeledDf = pd.concat(test_labeled_parts, axis=0, ignore_index=True)
print("\nFinal labeled test data shape:", testLabeledDf.shape)

testLabeledParquetPath = "data/test.parquet"
testLabeledDf.to_parquet(testLabeledParquetPath, engine="pyarrow", index=False)
print(f"Saved labeled test parquet to: {testLabeledParquetPath}")


# --- Optional: Build inference-style test set (code1, code2 only) ---
# If you have an unlabeled csv_data/test.csv (e.g., for submission), save it separately
# so it doesn't conflict with the labeled evaluation test set above.
selected_test_cols = ["code1", "code2"]

csvTestDataPath = "csv_data/test.csv"
if Path(csvTestDataPath).exists():
    test_df = pd.read_csv(csvTestDataPath)
    test_infer_df = test_df[selected_test_cols].copy()
    testInferParquetFilePath = "data/test_infer.parquet"
    test_infer_df.to_parquet(testInferParquetFilePath, engine="pyarrow", index=False)
    print(f"Saved inference test parquet to: {testInferParquetFilePath}")

