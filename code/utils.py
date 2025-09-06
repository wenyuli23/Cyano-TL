import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shap
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

def plot_predictions(y_test, y_pred, y_pred_before, x_test, target, output_file):
    """
    Plots predicted vs actual values with different markers for "after TL" and "before TL"
    and color-coded by unique products.

    Parameters:
    - y_test: pd.Series or np.ndarray, actual target values
    - y_pred: pd.Series or np.ndarray, predicted values from the model after transfer learning
    - y_pred_before: pd.Series or np.ndarray, predicted values from the model before transfer learning
    - x_test: pd.DataFrame, feature set including the 'product_old' column
    - target: str, target variable name used in title and output
    - output_file: str, file name to save the plot (default: 'scatter_plot.png')
    """

    unique_products = x_test['product'].str.replace(' ', '').unique()
    cmap = plt.colormaps['tab10']
    colors = [cmap(i / len(unique_products)) for i in range(len(unique_products))]
    color_dict = {product: colors[i] for i, product in enumerate(unique_products)}

    plt.figure(figsize=(8, 6))

    for product in unique_products:
        mask = x_test['product'] == product
        plt.scatter(y_test[mask], y_pred[mask], c=[color_dict[product]], label=f'{product}', alpha=0.6, marker='o', s=100)

    for product in unique_products:
        mask = x_test['product'] == product
        plt.scatter(y_test[mask], y_pred_before[mask], c=[color_dict[product]], alpha=0.6, marker='x', s=100)
    
    # Combine all values
    all_vals = np.concatenate([y_test, y_pred, y_pred_before])
    all_vals = all_vals[all_vals > 0]
    plt.plot([min(all_vals), max(all_vals)], [min(all_vals), max(all_vals)], 'k--', linewidth=1.5, label='y=x')

    circle_patch = Line2D([], [], marker='o', color='w', markerfacecolor='gray', markersize=8, label='after TL')
    cross_patch = Line2D([], [], marker='x', color='w', markeredgecolor='gray', markersize=8, label='before TL')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    handles.extend([circle_patch, cross_patch])
    labels.extend(['after TL', 'before TL'])
    plt.rcParams.update({
        'font.size': 14,           # base font size
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Thicken tick marks
    plt.tick_params(axis='both', which='both', width=1.5)

    plt.legend(handles, labels, loc='best', bbox_to_anchor=(1, 1))

    plt.xscale('log')
    plt.yscale('log')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Log10 of Actual Values')
    plt.ylabel('Log10 of Predicted Values')
    plt.title(f'{target} predictions for all products', pad=50)

    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')

def apply_imputation(df, rules):
    for _, row in rules.iterrows():
        feature = row.iloc[0]
        method = row.iloc[1]
        #print(f"Imputing {feature} with {method} method.")
        if feature in df.columns:
            if method == 'average':
                df[feature] = df[feature].astype('float64')
                mean_value = df[feature].mean()
                df.fillna({feature: mean_value}, inplace=True)
            elif method == 'mode':
                mode_value = df[feature].mode()[0]
                df.fillna({feature: mode_value}, inplace=True)
            elif method == 0:
                df.fillna({feature: 0}, inplace=True)
            elif method == 'na as a category':
                df[feature] = df[feature].astype('category')
                if 'missing' not in df[feature].cat.categories:
                    df[feature] = df[feature].cat.add_categories('missing')
                df.fillna({feature: 'missing'}, inplace=True)

def preprocess_data(df):
    encoder = OrdinalEncoder()
    cols = df.select_dtypes(include=['object', 'category']).columns
    df_enc = df.copy()
    df_enc[cols] = df_enc[cols].astype(str)
    df_enc[cols] = encoder.fit_transform(df_enc[cols])

    scaler = MinMaxScaler()
    df_scale = pd.DataFrame(scaler.fit_transform(df_enc), columns=df_enc.columns)

    return df_scale

def plot_top10_shap_values(model, model_before, features_to_include, x_test, target, output_file):
    """
    Plots the top 10 features with the highest SHAP values before and after transfer learning (TL).
    
    This function uses SHAP (SHapley Additive exPlanations) to interpret the contributions of features
    in two models. It highlights how the importance of the selected features changes between the models.
    
    Args:
        model: Trained model after transfer learning.
        model_before: Trained model before transfer learning.
        features_to_include: List of feature names to consider for SHAP value computation.
        x_test: DataFrame containing the test data used for SHAP value computation.
        target: Name of the target variable for the model (used in plot titles and file naming).
    
    Steps:
        1. Compute SHAP values for both models using the provided test data.
        2. Filter SHAP values and test data to include only the specified features.
        3. Calculate the mean absolute SHAP values for both models.
        4. Identify the top 10 features with the highest combined SHAP values.
        5. Create a scatter plot showing SHAP values for each model:
            - Use circles for the model after TL.
            - Use crosses for the model before TL.
        6. Use color mapping to represent the actual feature values.
        7. Save the plot as a PNG file in the `../figures/` directory.

    Output:
        - A scatter plot visualizing the top 10 feature contributions before and after transfer learning,
          with a color bar indicating the actual feature values.

    """

    explainer = shap.TreeExplainer(model)
    explainer2 = shap.TreeExplainer(model_before)
    shap_values = explainer.shap_values(x_test)
    shap_values2 = explainer2.shap_values(x_test)

    features_to_include_indices = [x_test.columns.get_loc(feature) for feature in features_to_include]

    filtered_shap_values = shap_values[:, features_to_include_indices]
    filtered_shap_values2 = shap_values2[:, features_to_include_indices]

    filtered_features = x_test[features_to_include]

    mean_abs_shap_values1 = np.mean(np.abs(filtered_shap_values), axis=0)
    mean_abs_shap_values2 = np.mean(np.abs(filtered_shap_values2), axis=0)

    # Get the top 10 features by average SHAP value for both before TL and after TL
    top_10_indices = np.argsort(mean_abs_shap_values1 + mean_abs_shap_values2)[-10:]
    top_10_features = np.array(features_to_include)[top_10_indices]

    # Filter SHAP values and feature data for top 10 features
    shap_values1_top10 = filtered_shap_values[:, top_10_indices]
    shap_values2_top10 = filtered_shap_values2[:, top_10_indices]
    X_test1_top10 = filtered_features.iloc[:, top_10_indices]
    X_test2_top10 = filtered_features.iloc[:, top_10_indices]

    fig, ax = plt.subplots()
    
    for i, feature in enumerate(top_10_features):
        # Normalize feature values for color mapping
        norm = plt.Normalize(vmin=X_test1_top10[feature].min(), vmax=X_test1_top10[feature].max())
        colors1 = plt.cm.coolwarm(norm(X_test1_top10[feature]))
        colors2 = plt.cm.coolwarm(norm(X_test2_top10[feature]))
        
        # Plot SHAP values for after TL with circles
        ax.scatter(shap_values1_top10[:, i], [i + 0.2] * len(shap_values1_top10), alpha=0.6, label='after TL' if i == 0 else "", marker='o', c=colors1)
        # Plot SHAP values for before TL with crosses
        ax.scatter(shap_values2_top10[:, i], [i - 0.2] * len(shap_values2_top10), alpha=0.6, label='before TL' if i == 0 else "", marker='x', c=colors2)

        ax.hlines(y=i, xmin=min(shap_values1_top10[:, i].min(), shap_values2_top10[:, i].min()), xmax=max(shap_values1_top10[:, i].max(), shap_values2_top10[:, i].max()), colors='grey', linestyles='dashed', alpha=0.3)
        
    ax.axvline(x=0, color='grey', linestyle='--')
    ax.set_yticks(np.arange(len(top_10_features)))
    ax.set_yticklabels(top_10_features)
    ax.set_xlabel("SHAP value (impact on model output)")
    #ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='after TL'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='grey', markersize=10, label='before TL')
    ]
    #ax.legend(handles=legend_handles, loc='best')
    
    color_bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax, label='Actual feature value', aspect=30, pad=0.01, shrink=0.6)
    color_bar.outline.set_visible(False)  # Remove color bar frame
    plt.title(f'Feature importance for {target} prediction')
    plt.tight_layout()
    plt.savefig(output_file, format='svg')

import numpy as np
import matplotlib.pyplot as plt

def plot_error_bars(df, output_file, error_type):
    """
    Plot grouped bar charts with error bars (± 1 STD) for either MSE or maape.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain these columns:
          - target
          - train_mse, train_std_mse
          - test_mse_before, test_std_mse_before
          - best_mse, best_mse_std
          - train_maape, train_std_maape
          - test_maape_before, test_std_maape_before
          - best_maape, best_maape_std
    output_file : str
        Path to save the figure (SVG).
    error_type : str
        "mse" or "maape" (case‑insensitive).
    """
    error_type = error_type.lower()
    metrics = df["target"][1:].tolist()

    if error_type == "maape":
        means = np.vstack([
            #df["train_maape"],
            df["test_maape_before"],
            df["best_maape"]
        ]).T
        stds = np.vstack([
            #df["train_std_maape"],
            df["test_std_maape_before"],
            df["best_maape_std"]
        ]).T
    else:  # default to MSE
        means = np.vstack([
            #df["train_mse"],
            df["test_mse_before"],
            df["best_mse"]
        ]).T
        stds = np.vstack([
            #df["train_std_mse"],
            df["test_std_mse_before"],
            df["best_mse_std"]
        ]).T

    phases = ["Test before TL", "Test after TL"]
    x      = np.arange(len(phases))
    width  = 0.8 / len(metrics)
    colors = plt.cm.tab10.colors[:len(metrics)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, mean_vals, std_vals, color) in enumerate(zip(metrics, means, stds, colors)):
        ax.bar(
            x + i*width,
            mean_vals,
            width,
            yerr=std_vals,
            capsize=6,
            label=metric,
            color=color,
            edgecolor="black",
            linewidth=1.5
        )
        ax.plot(
            x + i*width,
            mean_vals,
            marker='o',
            linestyle='-',
            color=color,
            linewidth=2.5,
            markeredgewidth=1.5
        )

    # Thicken axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_ylabel(f"{error_type.upper()} (±1 STD)", labelpad=10)
    ax.set_title(f"Model Performance ({error_type.upper()} ± STD)", pad=15)
    ax.set_xticks(x + width*(len(metrics)-1)/2)
    ax.set_xticklabels(phases)

    # Legend styling
    legend = ax.legend(title="Target feature", frameon=True, edgecolor="black")
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close(fig)
