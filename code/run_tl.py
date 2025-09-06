from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import copy
import itertools
from tqdm import tqdm
import argparse
import os
import pandas as pd
import numpy as np

from rf_transfer import *
from utils import *
import pickle

data_path = "../data/"
figures_path = "../figures/"
models_path = "../models/"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process source and plot arguments.")
parser.add_argument('-split', type=str, required=True, choices=['percent', 'paper'], help="Train test data split method: 'percent' or 'paper'.")
parser.add_argument('-source', type=str, required=True, choices=['7942', '6803', 'both'], help="Source species: '7942', '6803', or 'both'.")
parser.add_argument('-plot', type=str, required=True, choices=['yes', 'no'], help="Whether to plot: 'yes' or 'no'.")
parser.add_argument('-refit', type=str, required=True, choices=['r2', 'maape', 'mse'], help="The metric to optimize in refitting: 'r2', 'maape', or 'mse'.")
args = parser.parse_args()

# Load the source data file based on the species
if args.source == '7942':
    df_source = pd.read_csv(os.path.join(data_path, f'{args.source}.csv'), usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 1'], na_values=['na', '', '#REF!'])
    figures_path = f"../figures/{args.source}to2973_splitby{args.split}_{args.refit}/"
elif args.source == '6803':
    df_source = pd.read_csv(os.path.join(data_path, f'{args.source}.csv'), usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 1'], na_values=['na', '', '#REF!'])
    figures_path = f"../figures/{args.source}to2973_splitby{args.split}_{args.refit}/"
elif args.source == 'both':
    df_source1 = pd.read_csv(os.path.join(data_path, '6803.csv'), usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 1'], na_values=['na', '', '#REF!'])
    df_source2 = pd.read_csv(os.path.join(data_path, '7942.csv'), usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 1'], na_values=['na', '', '#REF!'])
    df_source = pd.concat([df_source1, df_source2], ignore_index=True)
    figures_path = f"../figures/{args.source}to2973_splitby{args.split}_{args.refit}/"

if not os.path.exists(figures_path):
    os.makedirs(figures_path)
df_target = pd.read_csv(os.path.join(data_path, '2973.csv'), usecols=lambda x: x not in ['Unnamed: 0', 'Unnamed: 1'], na_values=['na', '', '#REF!'])

### Apply imputation rules to both source and target data
imputation_rules = pd.read_excel('../data/impute.xlsx', header=None)
apply_imputation(df_source, imputation_rules)
apply_imputation(df_target, imputation_rules)

### Divide data into x and y
output_features = ['OD', 'growth_rate', 'product_titer', 'production_rate']
input_features = df_source.columns.difference(output_features + ['paper'])
#numerical_cols = df_target.drop(["paper"] + output_features, axis=1).select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = df_target.drop(["paper"] + output_features, axis=1).columns.tolist()
print("Numerical columns in target data:", numerical_cols)
print(f"Source data shape: {df_source.shape}")
print(f"Target data shape: {df_target.shape}")
print(f"Source data columns: {sorted(df_source.columns)}")
print(f"Target data columns: {sorted(df_target.columns)}")

### Divide target data into train and test sets
if args.split == 'paper':   # without information breach
    df_source['paper'] = df_source['paper'].astype(int)
    df_target['paper'] = df_target['paper'].astype(int)

    num_test_paper = 3
    paper_index = df_target["paper"].unique()
    combinations = list(itertools.combinations(paper_index, num_test_paper))
    #combinations = list(itertools.combinations([1,2,3,4,5], num_test_paper))
elif args.split == 'percent':   # with information breach
    X_train, X_test, y_train, y_test = train_test_split(df_target[input_features], df_target[output_features], test_size=0.33, random_state=42, stratify=df_target['paper'])
    X_train_processed = preprocess_data(X_train)
    X_test_processed = preprocess_data(X_test)

X_source_processed = preprocess_data(df_source[input_features])

# Start training and testing for each target feature
results_data = []
errors_data = []
for target in output_features:
    print("\n\nTarget feature:", target)
    if args.split == 'paper':
        record_score = None
        final_combo = None
        best_rf_before_final = None
        best_rf_final = None
        if args.refit == 'mse' or args.refit == 'maape':
            record_score = np.inf
        elif args.refit == 'r2':
            record_score = -np.inf
        
        for test_paper_list in tqdm(combinations, desc=f"Processing combinations for {target}"):
            df_test = df_target[df_target['paper'].isin(test_paper_list)]
            df_train = df_target[~df_target['paper'].isin(test_paper_list)]
            X_train_processed = preprocess_data(df_train[input_features])
            X_test_processed = preprocess_data(df_test[input_features])

            best_rf_before, train_mse, train_std_mse, train_maape, train_std_maape, train_r2, test_mse_before, test_std_mse_before, test_maape_before, test_std_maape_before, test_r2_before = train_random_forest(X_train_processed, df_train[target], X_test_processed, df_test[target], args.refit)
            model_copy = copy.deepcopy(best_rf_before)
            best_n, best_r2, best_mse, best_mse_std, best_maape, best_maape_std, best_y_pred, best_rf_after = transfer_learning(best_rf_before, X_source_processed, df_source[target], X_train_processed, df_train[target], X_test_processed, df_test[target], args.refit)
            if args.refit == 'mse':
                if best_mse < record_score:
                    final_combo = test_paper_list
                    record_score = test_mse_before
                    best_rf_final = copy.deepcopy(best_rf_after)
                    best_rf_before_final = model_copy
                    best_n_final = best_n
                    best_r2_final = best_r2
                    best_maape_final = best_maape
                    best_maape_std_final = best_maape_std
                    best_mse_final = best_mse
                    best_mse_std_final = best_mse_std
                    best_y_pred_final = best_y_pred
            elif args.refit == 'maape':
                if best_maape < record_score:
                    final_combo = test_paper_list
                    record_score = test_maape_before
                    best_rf_final = copy.deepcopy(best_rf_after)
                    best_rf_before_final = model_copy
                    best_n_final = best_n
                    best_r2_final = best_r2
                    best_maape_final = best_maape
                    best_maape_std_final = best_maape_std
                    best_mse_final = best_mse
                    best_mse_std_final = best_mse_std
                    best_y_pred_final = best_y_pred
            elif args.refit == 'r2':
                if best_r2 > record_score:
                    final_combo = test_paper_list
                    record_score = test_r2_before
                    best_rf_final = copy.deepcopy(best_rf_after)
                    best_rf_before_final = model_copy
                    best_n_final = best_n
                    best_r2_final = best_r2
                    best_maape_final = best_maape
                    best_maape_std_final = best_maape_std
                    best_mse_final = best_mse
                    best_mse_std_final = best_mse_std
                    best_y_pred_final = best_y_pred
            result_list = [test_paper_list, target, train_mse, train_maape, train_r2, test_mse_before, test_maape_before, test_r2_before, best_mse, best_maape, best_r2, best_n]
            results_data.append(result_list)
        
        errors_data.append([target, train_mse, train_std_mse, train_maape, train_std_maape, test_mse_before, test_std_mse_before, test_maape_before, test_std_maape_before, best_mse_final, best_mse_std_final, best_maape_final, best_maape_std_final])
        with open(os.path.join(models_path, f'{target}_model_{args.source}to2973_splitby{args.split}_{args.refit}.plk'), 'wb') as f:
            pickle.dump(best_rf_final, f)
        
        df_test_final = df_target[df_target['paper'].isin(final_combo)]
        X_test_processed_final = preprocess_data(df_test_final[input_features])
        if args.plot == 'yes':
            plot_top10_shap_values(best_rf_final, best_rf_before_final, numerical_cols, X_test_processed_final, target, output_file=os.path.join(figures_path, f'{target}_shap.svg'))
            plot_predictions(df_test_final[target], best_y_pred_final, best_rf_before_final.predict(X_test_processed_final), df_test_final[input_features], target, output_file=os.path.join(figures_path, f'{target}_scatter_plot.svg'))
            
    else:
        best_rf_before, train_mse, train_std_mse, train_maape, train_std_maape, train_r2, test_mse_before, test_std_mse_before, test_maape_before, test_std_maape_before, test_r2_before = train_random_forest(X_train_processed, y_train[target], X_test_processed, y_test[target], args.refit)
        model_copy = copy.deepcopy(best_rf_before)
        best_n, best_r2, best_mse, best_mse_std, best_maape, best_maape_std, best_y_pred, best_rf_after = transfer_learning(best_rf_before, X_source_processed, df_source[target], X_train_processed, y_train[target], X_test_processed, y_test[target], args.refit)
        with open(os.path.join(models_path, f'{target}_model_{args.source}to2973_splitby{args.split}_{args.refit}.plk'), 'wb') as f:
            pickle.dump(best_rf_after, f)
        results_data.append([target, train_mse, train_maape, train_r2, 
                                test_mse_before, test_maape_before, test_r2_before, 
                                best_mse, best_maape, best_r2, best_n])
        errors_data.append([target, train_mse, train_std_mse, train_maape, train_std_maape, test_mse_before, test_std_mse_before, test_maape_before, test_std_maape_before, best_mse, best_mse_std, best_maape, best_maape_std])
        if args.plot == 'yes':
            plot_top10_shap_values(best_rf_after, model_copy, numerical_cols, X_test_processed, target, output_file=os.path.join(figures_path, f'{target}_shap.svg'))
            plot_predictions(y_test[target], best_y_pred, model_copy.predict(X_test_processed), X_test, target, output_file=os.path.join(figures_path, f'{target}_scatter_plot.svg'))

### Plot error bar plots
errors_df = pd.DataFrame(errors_data, columns=["target",
    "train_mse", "train_std_mse", "train_maape", "train_std_maape",
    "test_mse_before", "test_std_mse_before", "test_maape_before", "test_std_maape_before",
    "best_mse", "best_mse_std", "best_maape", "best_maape_std"])
""" if args.plot == 'yes':
    plot_error_bars(errors_df, output_file=os.path.join(figures_path, 'error_bar_maape.svg'), error_type="maape")
    plot_error_bars(errors_df, output_file=os.path.join(figures_path, 'error_bar_MSE.svg'), error_type="MSE") """

### Save results to CSV
if args.split == 'paper':
    results_df = pd.DataFrame(results_data, columns=["Test Paper", "Target Feature", "Train MSE", "Train maape", "Train R2", 
                                                    "Test MSE Before TL", "Test maape Before TL", "Test R2 Before TL", 
                                                    "Test MSE After TL", "Test maape After TL", "Test R2 After TL", "n_tree"])
elif args.split == 'percent':
    results_df = pd.DataFrame(results_data, columns=["Target Feature", "Train MSE", "Train maape", "Train R2", 
                                                        "Test MSE Before TL", "Test maape Before TL", "Test R2 Before TL", 
                                                        "Test MSE After TL", "Test maape After TL", "Test R2 After TL", "n_tree"])

results_df.to_csv(f'../data/results_{args.source}to2973_splitby{args.split}_{args.refit}.csv', index=False)