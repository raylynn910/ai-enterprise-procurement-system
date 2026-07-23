import sqlite3
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'procurement.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def train_model():
    print("Loading data from database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query data
    query = """
    SELECT 
        PO_Status,
        Invoice_Status,
        Category,
        Supplier_Risk,
        Contract_Type,
        Quantity,
        Maverick_Spend,
        Single_Source_Flag
    FROM procurement_data
    """
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} rows.")

    # Create target variable: 1 if Disputed, 0 otherwise
    df['Is_Disputed'] = ((df['PO_Status'] == 'Disputed') | (df['Invoice_Status'] == 'Disputed')).astype(int)
    print(f"Found {df['Is_Disputed'].sum()} disputed records.")

    # Clean and encode features
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(1)
    df['Maverick_Spend'] = df['Maverick_Spend'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
    df['Single_Source_Flag'] = df['Single_Source_Flag'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

    # Encode categorical features
    cat_le = LabelEncoder()
    df['Category_Encoded'] = cat_le.fit_transform(df['Category'].fillna('Unknown'))
    
    risk_le = LabelEncoder()
    df['Supplier_Risk_Encoded'] = risk_le.fit_transform(df['Supplier_Risk'].fillna('Unknown'))
    
    contract_le = LabelEncoder()
    df['Contract_Type_Encoded'] = contract_le.fit_transform(df['Contract_Type'].fillna('Unknown'))

    # Prepare training data
    features = ['Category_Encoded', 'Supplier_Risk_Encoded', 'Contract_Type_Encoded', 
                'Quantity', 'Maverick_Spend', 'Single_Source_Flag']
    
    X = df[features]
    y = df['Is_Disputed']

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    # Save model and encoders
    print("Saving model and encoders...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(os.path.join(MODELS_DIR, "dispute_model.pkl"), "wb") as f:
        pickle.dump(model, f)
        
    with open(os.path.join(MODELS_DIR, "dispute_encoders.pkl"), "wb") as f:
        pickle.dump({
            'Category': cat_le,
            'Supplier_Risk': risk_le,
            'Contract_Type': contract_le
        }, f)

    print("Training complete! Model saved to models/dispute_model.pkl")

if __name__ == "__main__":
    train_model()
