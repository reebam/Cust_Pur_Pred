import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# 1. Setup Constants
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

# Change this to your ACTUAL tourism dataset path on Hugging Face or local
DATASET_PATH = "tourism_project/data/tourism (1).csv" 
REPO_ID = "reebamarium/Customer-Purchase-Prediction-App"

# 2. Load Dataset
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    print("✅ Local dataset loaded successfully.")
else:
    # Fallback: If not local, try to load from your HF dataset repo
    DATASET_URL = f"hf://datasets/{REPO_ID}/tourism(1).csv"
    df = pd.read_csv(DATASET_URL)
    print("✅ HF dataset loaded successfully.")

# 3. Data Cleaning (Updated for Tourism Context)
# Dropping ID columns or columns with too many unique strings
cols_to_drop = ['CustomerID', 'Name'] 
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 4. Encoding Categorical Columns
# We use a loop to handle multiple categories safely
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# 5. Define Target and Split
# Assuming 'ProdTaken' or 'Purchase' is your target column
target_col = 'ProdTaken' if 'ProdTaken' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]



# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

