import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. Load Data
DATASET_PATH = "tourism_project/data/tourism (1).csv" 
df = pd.read_csv(DATASET_PATH)

# 2. Basic Cleaning
cols_to_drop = ['CustomerID', 'Name'] 
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 3. Encoding - SAVE THE ENCODER IMMEDIATELY
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Save encoder directly to the root folder
joblib.dump(le, './encoder.joblib')
print("✅ encoder.joblib saved to root.")

# 4. Split
target_col = 'ProdTaken' if 'ProdTaken' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipeline Setup
numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), []) # Placeholders if needed
)

# 6. Train with XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain.values.ravel())
    
    # 7. SAVE THE BEST MODEL DIRECTLY TO ROOT
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, './tourism_model.pkl')
    
    # Metrics
    ypred = best_model.predict(Xtest)
    print(f"🚀 Training Complete! Accuracy: {accuracy_score(ytest, ypred):.4f}")
    print("✅ tourism_model.pkl saved to root.")
