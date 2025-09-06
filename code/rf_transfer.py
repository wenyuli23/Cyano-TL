from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np
import warnings
import copy

warnings.filterwarnings("ignore", message="invalid value encountered in subtract")

def aape(y_true, y_pred):
    """
    AAPE = arctan( |(y_true - y_pred) / y_true| )
    Bounded in [0, π/2).
    """
    # avoid division by zero; replace zeros with epsilon
    eps = np.finfo(float).eps
    y = np.where(y_true == 0, eps, y_true)
    ape = np.abs((y_true - y_pred) / y)
    return np.arctan(ape)

def mean_arctangent_absolute_percentage_error(y_true, y_pred):
    """
    MAAPE = mean( arctan( |(y_true - y_pred) / y_true| ) )
    Bounded in [0, π/2).
    """
    return np.mean(aape(y_true, y_pred))

def train_random_forest(X_train, y_train, X_test, y_test, refit_metric):
    """Train RandomForestRegressor with GridSearchCV and return best model and train metrics."""
    rf_regressor = RandomForestRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None]
    }
    maape_scorer = make_scorer(
        lambda y_true, y_pred: -mean_arctangent_absolute_percentage_error(y_true, y_pred),
        greater_is_better=True
    )
    scoring = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'maape': maape_scorer
    }
    grid_search = GridSearchCV(
        estimator=rf_regressor,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    val_mse = -grid_search.cv_results_['mean_test_mse'][grid_search.best_index_]
    val_std_mse = grid_search.cv_results_['std_test_mse'][grid_search.best_index_]
    val_maape = -grid_search.cv_results_['mean_test_maape'][grid_search.best_index_]
    val_std_maape = grid_search.cv_results_['std_test_maape'][grid_search.best_index_]
    val_r2 = grid_search.cv_results_['mean_test_r2'][grid_search.best_index_]
    print("Train mean MSE in 5-fold CV:", val_mse)
    print("Train mean maape in 5-fold CV:", val_maape)
    print("Train mean R² in 5-fold CV:", val_r2)

    y_pred_before = best_rf.predict(X_test)
    test_mse_before = mean_squared_error(y_test, y_pred_before)
    test_std_mse_before = np.std((y_test - y_pred_before) ** 2)
    test_maape_before = mean_arctangent_absolute_percentage_error(y_test, y_pred_before)
    test_std_maape_before = np.std(aape(y_test, y_pred_before))
    test_r2_before = r2_score(y_test, y_pred_before)
    print("Test mean MSE before TL:", test_mse_before)
    print("Test maape before TL:", test_maape_before)
    print("Test R² score before TL:", test_r2_before)
    return best_rf, val_mse, val_std_mse, val_maape, val_std_maape, val_r2, test_mse_before, test_std_mse_before, test_maape_before, test_std_maape_before, test_r2_before


def transfer_learning(best_rf, X_source, y_source, X_train, y_train, X_test, y_test, refit_metric):
    """Perform transfer learning by re-fitting worst trees and return best TL metrics and the trained model."""
    tree_scores = []
    for tree in best_rf.estimators_:
        y_pred_tree = tree.predict(X_test.values)
        if refit_metric == 'mse':
            score = -mean_squared_error(y_test, y_pred_tree)
        elif refit_metric == 'maape':
            score = -mean_arctangent_absolute_percentage_error(y_test, y_pred_tree)
        else:
            score = r2_score(y_test, y_pred_tree)
        tree_scores.append(score)
    sorted_tree_indices = np.argsort(tree_scores)
    best_r2 = -np.inf
    best_maape = np.inf
    best_maape_std = None
    best_mse = np.inf
    best_mse_std = None
    best_n = None
    best_y_pred = None
    model_returned = None
    n_worst_trees = [1, 3, 5, 10, 15, 20, 25, 30]
    for n_tree in n_worst_trees:
        # re-fit the n worst trees
        best_rf_copy = copy.deepcopy(best_rf)
        combined_X = pd.concat([X_source, X_train], axis=0)
        combined_y = pd.concat([y_source, y_train], axis=0)
        for tree_idx in sorted_tree_indices[:n_tree]:
            best_rf_copy.estimators_[tree_idx].fit(combined_X, combined_y)
        
        y_pred_after = best_rf_copy.predict(X_test)
        new_r2 = r2_score(y_test, y_pred_after)
        new_mse = mean_squared_error(y_test, y_pred_after)
        new_std_mse = np.std((y_test - y_pred_after) ** 2)
        new_maape = mean_arctangent_absolute_percentage_error(y_test, y_pred_after)
        new_std_maape = np.std(aape(y_test, y_pred_after))

        if refit_metric == 'mse':
            if new_mse < best_mse:
                best_mse, best_mse_std, best_maape, best_maape_std, best_r2, best_n, best_y_pred = new_mse, new_std_mse, new_maape, new_std_maape, new_r2, n_tree, y_pred_after
                model_returned = copy.deepcopy(best_rf_copy)
        elif refit_metric == 'maape':
            if new_maape < best_maape:
                best_mse, best_mse_std, best_maape, best_maape_std, best_r2, best_n, best_y_pred = new_mse, new_std_mse, new_maape, new_std_maape, new_r2, n_tree, y_pred_after
                model_returned = copy.deepcopy(best_rf_copy)
        else:   # refit_metric == 'r2'
            if new_r2 > best_r2:
                best_mse, best_mse_std, best_maape, best_maape_std, best_r2, best_n, best_y_pred = new_mse, new_std_mse, new_maape, new_std_maape, new_r2, n_tree, y_pred_after
                model_returned = copy.deepcopy(best_rf_copy)

    print(f"Test R² score after TL with {best_n} worst trees:", best_r2)
    print(f"Test maape after TL with {best_n} worst trees:", best_maape)
    print(f"Test MSE after TL with {best_n} worst trees:", best_mse)

    return best_n, best_r2, best_mse, best_mse_std, best_maape, best_maape_std, best_y_pred, model_returned