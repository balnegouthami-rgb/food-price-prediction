import pandas as pd

def preprocess(df):
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop unnecessary columns
    cols_to_drop = [
        'name', 'stations', 'description', 'icon',
        'sunrise', 'sunset', 'severerisk'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Sort by date
    df = df.sort_values('date')
    
    # Handle numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].interpolate(method='linear')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # Feature engineering from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df