"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.array(X)
    X_clear = X.copy()
    N, F = X.shape
    nan_mask = np.isnan(X)
    mean = np.nanmean(X_clear, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    for col in range(F):
        X_clear[nan_mask[:,col], col] = mean[col]
    return X_clear

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.array(X)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3-q1
    lower = q1-k*iqr
    upper = q3+k*iqr
    return (lower, upper)

# Step 3 - clip_columns
import numpy as np

def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X = np.array(X)
    lower = np.array(lower)
    upper = np.array(upper)
    X_clipped = X.copy()
    N, F = X.shape
    for col in range(F):
        X_clipped[:,col] = np.clip(X_clipped[:, col], a_min=lower[col], a_max=upper[col])
    return X_clipped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X=np.array(X)
    col = np.array(col)
    X_appended = np.column_stack((X, col))
    return X_appended

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.array(labels)
    N = labels.shape[0]
    uniq_labels = np.unique(labels)
    C = uniq_labels.shape[0]
    label2idx={}
    for i in range(C):
        lab = uniq_labels[i]
        label2idx[lab]=i

    label_indices = [label2idx[lab] for lab in labels]
    one_hot_mat=np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        one_hot_mat[i, label_indices[i]] = 1.0

    return one_hot_mat

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X=np.array(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std=np.where(std != 0, std, 1.0)
    return (mean, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X_norm = (X-mean)/std
    return X_norm

# Step 9 - add_bias_column
import numpy as np
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    x = np.ones((X.shape[0],))
    X_bias = np.column_stack((x, X))
    return X_bias

# Step 10 - make_shuffled_indices
import numpy as np
import random
def make_shuffled_indices(n_samples, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    return indices

# Step 11 - partition_indices
import numpy as np
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    indices = np.array(indices)
    N = indices.shape[0]
    n_train = int(np.floor(N*train_ratio))
    n_val = int(np.floor(N*val_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return (train_idx, val_idx, test_idx)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = X[indices,:]
    y_sub = y[indices]
    return (X_sub, y_sub)

# Step 13 - ols_fit
import numpy as np
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    preds = X@theta
    return preds

# Step 15 - mean_absolute_error
import numpy as np
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true-y_pred))
    return mae

# Step 16 - root_mean_squared_error
import numpy as np
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    return rmse

# Step 17 - r_squared
import numpy as np
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean = np.mean(y_true)
    
    SS_tot = np.sum((y_true-mean)**2)
    if SS_tot == 0:
        return 0.0
    SS_res = np.sum((y_true-y_pred)**2)
    return 1-SS_res/SS_tot

# Step 18 - residual_summary
import numpy as np
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r = y_true - y_pred 
    mean = np.mean(r)
    std = np.std(r, ddof=0)
    median = np.median(np.abs(r))
    
    return {
        'mean': mean,
        'std':std,
        'median_abs':median
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_clean = X.copy()
    X_clean = impute_nan_with_mean(X_clean)
    lower, upper = compute_iqr_bounds(X_clean, k = iqr_k)
    X_clean = clip_columns(X_clean, lower, upper)
    return X_clean

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num_cols = X_num[:, ratio_num_idx]
    den_cols = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num_cols, den_cols)
    X_ass = X_num.copy()
    X_ass = append_column(X_ass, ratio)

    if cat_labels is not None:
        one_hot_labels = one_hot_encode(cat_labels)
        X_ass = append_column(X_ass, one_hot_labels)

    return X_ass

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    N = X.shape[0]
    indices = make_shuffled_indices(N, seed)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X_train = splits['X_train']
    y_train = splits['y_train']
    mean, std = fit_standardizer(X_train)
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    X_train = apply_standardizer(X_train, mean, std)
    X_train = add_bias_column(X_train)
    X_val = apply_standardizer(X_val, mean, std)
    X_val = add_bias_column(X_val)
    X_test = apply_standardizer(X_test, mean, std)
    X_test = add_bias_column(X_test)
    std_spilts = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test 
    }

    return std_spilts, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    rs = r_squared(y_true, y_pred)
    res_sum = residual_summary(y_true, y_pred)
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': rs,
        'residual_summary': res_sum
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X = prepare_cleaned_features(X, iqr_k)
    X = assemble_feature_matrix(X, ratio_num_idx, ratio_den_idx, cat_labels)
    splits = make_train_val_test(X, y, train_ratio, val_ratio, seed)
    splits, mean, std = standardize_and_add_bias(splits)
    X_train = splits['X_train']
    y_train = splits['y_train']
    w = ols_fit(X_train, y_train)
    X_val = splits['X_val']
    y_val = splits['y_val']
    y_val_pred = ols_predict(X_val)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    X_test = splits['X_test']
    y_test = splits['y_test']
    y_test_pred = ols_predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    return {
        'theta': w,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics
    }

