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


# select only the relevant columns
selected_cols = ['code1', 'code2', 'similar']
df = df[selected_cols]
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


# split the data into train and cross validation sets with 15% for cross validation
trainData, crossValidationData = train_test_split(df, test_size=0.15, random_state=42)
print('\nTrain size:', len(trainData), 'CV size:', len(crossValidationData))


# save train and cross validation to parquet
trainData.to_parquet("data/train.parquet", engine='pyarrow', index=False)
crossValidationData.to_parquet("data/cross_validation.parquet", engine='pyarrow', index=False)
print("Saved train and cross-validation parquet files to data/.")


# get the test data from csv file
csvTestDataPath = "csv_data/test.csv"
if not Path(csvTestDataPath).exists():
    raise FileNotFoundError(f"CSV test data file not found: {csvTestDataPath}")
test_df = pd.read_csv(csvTestDataPath)


# select columns if present
selected_test_cols = ['code1', 'code2']
test_df = test_df[selected_test_cols]
print('\nTest data shape:', test_df.shape)
print(test_df.head(5))


# save test parquet
testParquetFilePath = "data/test.parquet"
test_df.to_parquet(testParquetFilePath, engine='pyarrow', index=False)
print(f"Saved test parquet to: {testParquetFilePath}")

