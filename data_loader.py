import pandas as pd
import json

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(config):
    df = pd.read_csv(config['dataset_path'])
    
    # Load class mappings
    class_df = pd.read_csv("class.csv")
    class_mapping = dict(zip(class_df['Class_Number'], class_df['Class_Type']))
    
    features = config['features']
    target = config['target']
    
    feature_names = features
    X_df = df[features].copy()
    
    # Normalize inputs (e.g. 'legs' in zoo.csv goes from 0 to 8, we want it 0-1 for visualization)
    for col in X_df.columns:
        col_min = X_df[col].min()
        col_max = X_df[col].max()
        if col_max > col_min:
            X_df[col] = (X_df[col] - col_min) / (col_max - col_min)
            
    X = X_df.values
    y = df[target].values
    
    # Map classes to 0-indexed if they don't start from 0
    target_class_names = []
    if y.min() > 0:
        y_min = y.min()
        y = y - y_min
        for i in range(len(class_mapping)):
            if (i + y_min) in class_mapping:
                target_class_names.append(class_mapping[i + y_min])
            else:
                target_class_names.append(f"Class {i+y_min}")
    else:
        for i in range(len(class_mapping)):
            if i in class_mapping:
                target_class_names.append(class_mapping[i])
            else:
                target_class_names.append(f"Class {i}")
        
    names = df['animal_name'].values if 'animal_name' in df.columns else [f"Item {i}" for i in range(len(df))]
    
    return X, y, names, feature_names, target_class_names
