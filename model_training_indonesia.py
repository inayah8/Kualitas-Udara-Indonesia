import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Import functions from data preparation
from data_preparation_indonesia import create_synthetic_dataset_indonesia, prepare_data

# Label mapping
LABEL_MAP = {'Aman': 0, 'Waspada': 1, 'Tidak Aman bagi Rentan': 2, 'Tidak Aman': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def train_model_indonesia(n_samples=5000):
    """Train Random Forest model with synthetic data"""
    print("🔄 Training model dengan data sintetis...")
    df = create_synthetic_dataset_indonesia(n_samples=n_samples)
    X_train, X_test, y_train_str, y_test_str, preprocessor_info = prepare_data(df)

    # Ensure all labels are valid
    valid_labels = set(LABEL_MAP.keys())
    y_train_valid = [l if l in valid_labels else 'Aman' for l in y_train_str]
    y_test_valid = [l if l in valid_labels else 'Aman' for l in y_test_str]

    # Encode labels
    y_train = np.array([LABEL_MAP[l] for l in y_train_valid], dtype=int)
    y_test = np.array([LABEL_MAP[l] for l in y_test_valid], dtype=int)

    # Train model
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred_num = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_num)
    print(f"✅ Accuracy: {acc:.3f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred_num, target_names=list(LABEL_MAP.keys())))

    # Save model and preprocessing info
    joblib.dump(rf_model, 'rf_model_indonesia.pkl')
    joblib.dump(preprocessor_info, 'preprocessor_info.pkl')
    joblib.dump(LABEL_MAP, 'label_map.pkl')
    print("✅ Model saved: rf_model_indonesia.pkl, preprocessor_info.pkl, label_map.pkl")

    return rf_model, preprocessor_info

def predict_quality_indonesia(features, model, preprocessor_info):
    """
    Predict air quality based on features
    """
    try:
        # Load label map
        label_map = joblib.load('label_map.pkl')
        inv_label_map = {v: k for k, v in label_map.items()}
        
        numeric_features = preprocessor_info['numeric_features']
        prov_cols = preprocessor_info['provinsi_columns']
        scaler = preprocessor_info['scaler']

        # Build numeric row vector
        X_num = np.array([[float(features.get(col, 0.0)) for col in numeric_features]])
        X_num_scaled = scaler.transform(X_num)

        # Build province one-hot encoding
        prov = str(features.get('Provinsi', 'Unknown'))
        prov_df = pd.get_dummies(pd.Series([prov]), prefix='Prov')
        
        # Reindex to match training columns
        prov_row = pd.DataFrame(0.0, columns=prov_cols, index=[0])
        for c in prov_cols:
            if c in prov_df.columns:
                prov_row.loc[0, c] = 1.0

        X_pre = np.hstack([X_num_scaled, prov_row.values])
        
        # Predict
        pred_num = model.predict(X_pre)[0]
        probs = model.predict_proba(X_pre)[0]
        maxp = float(np.max(probs))
        label_str = inv_label_map.get(int(pred_num), "Unknown")
        return label_str, maxp
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to manual calculation based on PM2.5
        pm25 = features.get('PM2.5', 0)
        from data_preparation_indonesia import _label_from_pm25
        manual_label = _label_from_pm25(pm25)
        return manual_label, 0.8  # Default confidence

if __name__ == "__main__":
    # Train model
    model, preproc = train_model_indonesia(n_samples=3000)
    
    # Test prediction
    sample = {
        'PM2.5': 12.0, 
        'PM10': 40.0, 
        'CO': 8.0, 
        'Suhu': 30.0, 
        'Kelembaban': 70.0, 
        'Provinsi': 'Jawa Timur'
    }
    label, prob = predict_quality_indonesia(sample, model, preproc)
    print(f"🧪 Test prediction: {label} (confidence: {prob:.3f})")