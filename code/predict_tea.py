import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import joblib

def compare_excel_columns(file1, file2):
    # Read the Excel files
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    # Extract column names
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    
    # Compare column names
    only_in_file1 = columns1 - columns2
    only_in_file2 = columns2 - columns1
    
    # Print the differences
    if only_in_file1:
        print(f"Columns only in {file1}: {only_in_file1}")
    else:
        print(f"No unique columns in {file1}")
    
    if only_in_file2:
        print(f"Columns only in {file2}: {only_in_file2}")
    else:
        print(f"No unique columns in {file2}")
    
    # Keep the column order the same as in file2, excluding the unique ones in file2
    common_columns = [col for col in df2.columns if col in df1.columns]
    common_columns_sorted = sorted(common_columns)  # Sort columns alphabetically
    df1_reordered = df1[common_columns_sorted]
    
    return df1_reordered

def preprocess_data(df):
    encoder = OrdinalEncoder()
    cols = df.select_dtypes(include=['object', 'category']).columns
    df_enc = df.copy()
    df_enc[cols] = df_enc[cols].astype(str)
    df_enc[cols] = encoder.fit_transform(df_enc[cols])

    scaler = MinMaxScaler()
    df_scale = pd.DataFrame(scaler.fit_transform(df_enc), columns=df_enc.columns)

    return df_scale

def load_model_and_predict(df, model_path):
    model = joblib.load(model_path)
    predictions = model.predict(df)
    return predictions

# Example usage
file1 = '../data/TL_TEA_v2.xlsx'
file2 = '../data/2973_v2.xlsx'
df1_reordered = compare_excel_columns(file1, file2)
df_tea = preprocess_data(df1_reordered)
print(df_tea.head())

for target in ['growth_rate', 'product_titer', 'production_rate', 'OD']:
    model_path = f'../models/{target}_model.pkl'
    predictions = load_model_and_predict(df_tea, model_path)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=[f'{target}_pred'])
    predictions_df.to_csv(f'../data/predictions/{target}_pred.csv', index=False)
    
    print(f'Saved predictions for {target} to ../data/predictions/{target}_pred.csv')
