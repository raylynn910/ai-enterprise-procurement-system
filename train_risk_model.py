import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def train_supplier_risk_model():
    print("Loading data...")
    df = pd.read_csv('dataset/data.csv')
    
    # Selected features based on the notebook
    features = ['Supplier ESG Score', 'On Time Delivery', 'Days Late', 'PO Status']
    target = 'Supplier Risk'
    
    X = df[features]
    y = df[target]
    
    print("Preprocessing data...")
    # Define categorical and numerical features
    categorical_features = ['On Time Delivery', 'PO Status']
    numerical_features = ['Supplier ESG Score', 'Days Late']
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42
        ))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model pipeline
    os.makedirs('backend/models', exist_ok=True)
    model_path = 'backend/models/supplier_risk_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_supplier_risk_model()
