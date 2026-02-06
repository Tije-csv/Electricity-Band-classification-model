import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def generate_raw_data(n_feeders=1000):
    """
    Simulates the 'Business Logic' of the Nigerian Power Grid.
    """
    np.random.seed(42)
    data = []
    
    for f_id in range(n_feeders):
        # 1. Static Traits (Features)
        zone = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.4, 0.4, 0.2])
        disco = np.random.choice(['IKEDC', 'AEDC', 'EKEDC', 'KEDCO', 'IBEDC'])
        feeder_age = np.random.gamma(2, 5) # Realistic aging distribution
        has_broken_transformer = np.random.random() < 0.15 # 15% failure rate
        
        # 2. Supply Logic 
        # Urban starts high (~20), Rural starts low (~8)
        base_hours = {'Urban': 20, 'Suburban': 14, 'Rural': 8}[zone]
        
       
        daily_hours = base_hours + np.random.normal(0, 2)
        if has_broken_transformer: daily_hours -= 5
        if feeder_age > 15: daily_hours -= 3
        
        # Clamp between 0 and 24
        final_hours = max(0, min(24, daily_hours))
        
        data.append({
            'feeder_id': f_id,
            'disco': disco,
            'zone': zone,
            'feeder_age': round(feeder_age, 1),
            'transformer_issue': has_broken_transformer,
            'avg_supply_hours': final_hours # The 'Secret' value
        })
    
    return pd.DataFrame(data)


def prepare_dataset(df):
    """
    Cleans data and prevents 'Data Leakage'.
    """
    # Create the Target (The Label) based on NERC standards
    def get_band(h):
        if h >= 20: return 'A'
        if h >= 16: return 'B'
        if h >= 12: return 'C'
        if h >= 8:  return 'D'
        return 'E'
    
    df['target_band'] = df['avg_supply_hours'].apply(get_band)
    
    #Remove 'avg_supply_hours' so the model has to guess the band using ONLY the features.
    X = df.drop(columns=['feeder_id', 'target_band', 'avg_supply_hours'])
    y = df['target_band']
    
    # Encode Categorical Words to Numbers
    X = pd.get_dummies(X, columns=['disco', 'zone'], drop_first=True)
    
    # Encode Target Labels (A-E -> 0-4)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42), le



def train_and_evaluate(splits, encoder):
    X_train, X_test, y_train, y_test = splits
    
 
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=5)
    
  
    clf.fit(X_train, y_train)
    

    y_pred = clf.predict(X_test)
    
  
    print("--- Model Performance Report ---")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    return clf


if __name__ == "__main__":
  
    raw_df = generate_raw_data(2000)
    
    data_splits, label_encoder = prepare_dataset(raw_df)
    
    model = train_and_evaluate(data_splits, label_encoder)