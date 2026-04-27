
#!/usr/bin/env python3
"""
Cayley-Hamilton-Guided Krylov Learning Pipeline
===============================================

This script implements a fully reproducible experimental workflow for the article:

    "Cayley–Hamilton-Guided Krylov Regularization for Machine Learning:
     Theory, Computational Architecture, and Empirical Evidence"

The code is intentionally verbose and heavily commented so that every design
decision can be audited, replicated, and adapted. It creates three result
subdirectories:

    results/5.1  -> exactness / convergence diagnostics
    results/5.2  -> predictive performance and efficiency
    results/5.3  -> robustness and interpretability analyses

The pipeline uses only widely available scientific Python libraries:
NumPy, pandas, matplotlib, and scikit-learn.

Conceptual summary
------------------
The article formulates regularized least-squares classification as the solution
of linear systems of the form:

    (X^T X + λ I) W = X^T Y

where X is the design matrix, λ > 0 is a regularization parameter, Y is a
one-hot response matrix, and W contains class-wise regression weights.

The Cayley-Hamilton theorem guarantees that any square matrix satisfies its own
characteristic polynomial. In finite dimension, this means matrix inverse-like
operators admit a finite polynomial representation in the matrix itself. In
practice, rather than explicitly computing potentially unstable monomial
coefficients, the code exploits the theorem through a numerically stable Krylov
subspace implementation based on conjugate gradients (CG). The resulting
Cayley-Hamilton-guided view motivates the use of polynomial iterative solvers
instead of repeated dense inversion.

Outputs
-------
The script saves:
- CSV summaries for every major result table.
- PNG images for the tables (easy to embed into a manuscript).
- Figures showing residual decay, accuracy comparisons, robustness, etc.
- A README and requirements file.
- A ZIP package with everything needed for replication.

The script is safe to execute repeatedly. Existing result files are overwritten.
"""

from __future__ import annotations

import os
import time
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Utility helpers
# =============================================================================

def set_global_seed(seed: int = 42) -> np.random.Generator:
    """
    Set deterministic random behavior for NumPy and return a generator.

    A dedicated generator is used instead of relying only on the global random
    state, because it makes the code easier to reason about when multiple
    sampling operations are performed.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: Path) -> None:
    """
    Create a directory (and parent directories) if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, content: str) -> None:
    """
    Save a plain text file using UTF-8 encoding.
    """
    path.write_text(content, encoding="utf-8")


def format_mean_std(values: Iterable[float], decimals: int = 3) -> str:
    """
    Format a sequence as 'mean ± std' with a fixed number of decimals.
    """
    values = list(values)
    mean_v = np.mean(values)
    std_v = np.std(values, ddof=0)
    return f"{mean_v:.{decimals}f} ± {std_v:.{decimals}f}"


# =============================================================================
# Data preparation
# =============================================================================

def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Load all benchmark datasets used in the manuscript.

    Returns
    -------
    dict
        Mapping from dataset name to:
        (X, y, feature_names)

    Notes
    -----
    All datasets come from scikit-learn and are therefore available offline.
    This makes the pipeline self-contained and easy to reproduce in restricted
    environments.
    """
    datasets = {}

    iris = load_iris()
    datasets["iris"] = (iris.data.astype(float), iris.target.astype(int), list(iris.feature_names))

    wine = load_wine()
    datasets["wine"] = (wine.data.astype(float), wine.target.astype(int), list(wine.feature_names))

    breast = load_breast_cancer()
    datasets["breast_cancer"] = (
        breast.data.astype(float),
        breast.target.astype(int),
        list(breast.feature_names),
    )

    digits = load_digits()
    # Generate human-readable feature names for the 8x8 pixel grid.
    digit_feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
    datasets["digits"] = (digits.data.astype(float), digits.target.astype(int), digit_feature_names)

    return datasets


def one_hot(y: np.ndarray) -> np.ndarray:
    """
    Convert a class vector to a one-hot response matrix.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Integer class labels.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_classes)
        One-hot encoding of y.
    """
    classes = np.unique(y)
    Y = np.zeros((len(y), len(classes)), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def augment_bias(X: np.ndarray) -> np.ndarray:
    """
    Add a constant bias column to the design matrix.

    This is done explicitly because the regularized least-squares formulation
    used here treats the model as a linear operator in augmented coordinates.
    """
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])


# =============================================================================
# Linear algebra and learning algorithms
# =============================================================================

def regularized_system(Xa: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the regularized normal-equation system:

        A W = B
        A = X^T X + λ I
        B = X^T Y

    Parameters
    ----------
    Xa : ndarray
        Augmented design matrix.
    Y : ndarray
        One-hot target matrix.
    lam : float
        Positive regularization strength.

    Returns
    -------
    A, B : ndarrays
        Linear system matrix and right-hand side.
    """
    d = Xa.shape[1]
    A = Xa.T @ Xa + lam * np.eye(d)
    B = Xa.T @ Y
    return A, B


def direct_rls_fit(X_train: np.ndarray, y_train: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    """
    Fit a regularized least-squares classifier using a direct linear solve.

    The model solves:
        (X^T X + λ I) W = X^T Y

    Returns a dictionary with everything needed for prediction and analysis.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Xa = augment_bias(Xs)
    Y = one_hot(y_train)
    A, B = regularized_system(Xa, Y, lam)
    W = np.linalg.solve(A, B)

    return {
        "method": "direct",
        "lambda": lam,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "weights": W,
        "classes": np.unique(y_train),
        "A": A,
        "B": B,
    }


def cg(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int | None = None,
    tol: float = 1e-10,
    x0: np.ndarray | None = None,
    return_history: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Conjugate gradient solver for symmetric positive definite systems.

    Parameters
    ----------
    A : ndarray of shape (d, d)
        Symmetric positive definite matrix.
    b : ndarray of shape (d,)
        Right-hand side.
    max_iter : int or None
        Maximum number of iterations. If None, uses d.
    tol : float
        Relative tolerance based on the residual norm.
    x0 : ndarray or None
        Optional initial guess. Defaults to zero vector.
    return_history : bool
        If True, returns a residual history for convergence visualization.

    Returns
    -------
    x : ndarray
        Approximate solution.
    info : dict
        Metadata including:
        - iterations
        - final_residual
        - converged
        - residual_history (optional)

    Notes
    -----
    Although the article is motivated by the Cayley-Hamilton theorem, the solver
    itself does not explicitly compute characteristic polynomial coefficients.
    That would be numerically fragile in moderate to high dimension. Instead,
    CG constructs polynomial approximations in Krylov subspaces, which is the
    stable operational interpretation adopted by the manuscript.
    """
    d = A.shape[0]
    if max_iter is None:
        # In exact arithmetic, CG terminates in at most d iterations for a
        # d-dimensional SPD system. In floating-point arithmetic, however,
        # ill-conditioned problems can benefit from extra refinement steps.
        max_iter = 5 * d

    if x0 is None:
        x = np.zeros(d, dtype=float)
    else:
        x = x0.astype(float).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        info = {
            "iterations": 0,
            "final_residual": 0.0,
            "converged": True,
            "residual_history": [0.0] if return_history else None,
        }
        return x, info

    residual_history = [math.sqrt(rs_old) / b_norm]

    converged = False
    iterations = 0

    for k in range(max_iter):
        Ap = A @ p
        denom = float(p @ Ap)

        # A tiny safeguard prevents division-by-zero in pathological cases.
        if abs(denom) < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)

        iterations = k + 1
        rel_res = math.sqrt(rs_new) / b_norm
        residual_history.append(rel_res)

        if rel_res < tol:
            converged = True
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    info = {
        "iterations": iterations,
        "final_residual": residual_history[-1],
        "converged": converged,
    }
    if return_history:
        info["residual_history"] = residual_history
    return x, info


def cg_multi(
    A: np.ndarray,
    B: np.ndarray,
    max_iter: int | None = None,
    tol: float = 1e-10,
    return_histories: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Solve multiple right-hand sides with CG by treating each class column
    independently.

    Parameters
    ----------
    A : ndarray of shape (d, d)
        System matrix.
    B : ndarray of shape (d, n_classes)
        Right-hand sides.
    max_iter : int or None
        Maximum number of CG iterations.
    tol : float
        Residual tolerance.
    return_histories : bool
        Whether to collect residual histories.

    Returns
    -------
    W : ndarray of shape (d, n_classes)
        Weight matrix.
    infos : list of dict
        Solver metadata for each class-wise solve.
    """
    cols = []
    infos = []
    for j in range(B.shape[1]):
        xj, info = cg(
            A,
            B[:, j],
            max_iter=max_iter,
            tol=tol,
            return_history=return_histories,
        )
        cols.append(xj)
        infos.append(info)
    W = np.column_stack(cols)
    return W, infos


def cg_rls_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam: float,
    max_iter: int | None = None,
    tol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Fit the same regularized least-squares classifier, but solve the underlying
    linear systems with conjugate gradients instead of dense inversion.

    When max_iter is None (or sufficiently large), the solution should match the
    direct solve to numerical precision. When max_iter is small, the model
    becomes a truncated polynomial approximation, which is the operational form
    used for the TCG variant.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Xa = augment_bias(Xs)
    Y = one_hot(y_train)
    A, B = regularized_system(Xa, Y, lam)
    W, infos = cg_multi(A, B, max_iter=max_iter, tol=tol, return_histories=False)

    return {
        "method": "cg" if max_iter is None else "tcg",
        "lambda": lam,
        "max_iter": max_iter,
        "tolerance": tol,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "weights": W,
        "classes": np.unique(y_train),
        "A": A,
        "B": B,
        "infos": infos,
    }


def predict_scores(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """
    Compute class scores for a fitted regularized least-squares model.
    """
    mean_ = model["scaler_mean"]
    scale_ = model["scaler_scale"]
    Xs = (X - mean_) / scale_
    Xa = augment_bias(Xs)
    return Xa @ model["weights"]


def predict_labels(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """
    Predict labels from class scores by argmax.
    """
    scores = predict_scores(model, X)
    return np.argmax(scores, axis=1)


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    max_iter: int = 2000,
) -> Dict[str, object]:
    """
    Fit a multinomial/ovr logistic regression baseline.

    This baseline is not included to claim universal superiority of the proposed
    operator view. It is included to provide a familiar benchmark and to show
    that the Cayley-Hamilton-guided architecture remains competitive while
    offering strong algebraic transparency.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(Xs, y_train)

    return {
        "method": "logreg",
        "C": C,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "model": model,
        "classes": np.unique(y_train),
    }


def predict_logreg(model: Dict[str, object], X: np.ndarray) -> np.ndarray:
    """
    Predict labels for the logistic regression baseline.
    """
    mean_ = model["scaler_mean"]
    scale_ = model["scaler_scale"]
    Xs = (X - mean_) / scale_
    return model["model"].predict(Xs)


# =============================================================================
# Model selection
# =============================================================================

def validation_split_indices(y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single stratified train/validation split inside a training fold.

    This inner split is used for hyperparameter selection. The design keeps the
    computational load reasonable while still making hyperparameter choices data
    driven and reproducible.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(splitter.split(np.zeros_like(y), y))
    return train_idx, val_idx


def tune_direct_rls(X_train: np.ndarray, y_train: np.ndarray, lambdas: List[float]) -> float:
    """
    Select the best regularization strength for the direct RLS model using
    validation macro-F1.
    """
    inner_train_idx, val_idx = validation_split_indices(y_train, random_state=42)
    X_inner, y_inner = X_train[inner_train_idx], y_train[inner_train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_lam = None
    best_score = -np.inf

    for lam in lambdas:
        model = direct_rls_fit(X_inner, y_inner, lam)
        pred = predict_labels(model, X_val)
        score = f1_score(y_val, pred, average="macro")
        if score > best_score:
            best_score = score
            best_lam = lam

    return float(best_lam)


def tune_cg_rls(X_train: np.ndarray, y_train: np.ndarray, lambdas: List[float]) -> float:
    """
    Select the best lambda for the full CG version.

    Because full CG is intended to recover the direct solution, only lambda is
    tuned; the iteration budget is left unrestricted.
    """
    inner_train_idx, val_idx = validation_split_indices(y_train, random_state=42)
    X_inner, y_inner = X_train[inner_train_idx], y_train[inner_train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_lam = None
    best_score = -np.inf

    for lam in lambdas:
        model = cg_rls_fit(X_inner, y_inner, lam, max_iter=None, tol=1e-10)
        pred = predict_labels(model, X_val)
        score = f1_score(y_val, pred, average="macro")
        if score > best_score:
            best_score = score
            best_lam = lam

    return float(best_lam)


def tune_tcg_rls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: List[float],
    iteration_grid: List[int],
) -> Tuple[float, int]:
    """
    Tune both lambda and the truncation depth for the truncated CG model.

    Early stopping turns the polynomial solver into a practical regularizer:
    lower iteration budgets bias the hypothesis space toward smoother operator
    approximations while reducing computational cost.
    """
    inner_train_idx, val_idx = validation_split_indices(y_train, random_state=42)
    X_inner, y_inner = X_train[inner_train_idx], y_train[inner_train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_combo = None
    best_score = -np.inf

    for lam in lambdas:
        for k in iteration_grid:
            model = cg_rls_fit(X_inner, y_inner, lam, max_iter=k, tol=1e-10)
            pred = predict_labels(model, X_val)
            score = f1_score(y_val, pred, average="macro")
            if score > best_score:
                best_score = score
                best_combo = (lam, k)

    return float(best_combo[0]), int(best_combo[1])


def tune_logreg(X_train: np.ndarray, y_train: np.ndarray, C_grid: List[float]) -> float:
    """
    Select the logistic-regression inverse regularization parameter C.
    """
    inner_train_idx, val_idx = validation_split_indices(y_train, random_state=42)
    X_inner, y_inner = X_train[inner_train_idx], y_train[inner_train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    best_C = None
    best_score = -np.inf

    for C in C_grid:
        model = fit_logistic_regression(X_inner, y_inner, C=C, max_iter=2000)
        pred = predict_logreg(model, X_val)
        score = f1_score(y_val, pred, average="macro")
        if score > best_score:
            best_score = score
            best_C = C

    return float(best_C)


# =============================================================================
# Plotting helpers
# =============================================================================

def dataframe_to_png(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    font_size: int = 9,
    scale_x: float = 1.2,
    scale_y: float = 1.3,
) -> None:
    """
    Render a pandas DataFrame as a PNG image using matplotlib.

    This is helpful because journals often require convenient figure/table
    assets that can be inserted manually if needed.
    """
    # A size heuristic based on rows/columns avoids tiny unreadable images.
    nrows, ncols = df.shape
    fig_width = max(8, ncols * 1.4)
    fig_height = max(1.8, 0.55 * (nrows + 1) + (0.6 if title else 0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(title, pad=16)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(scale_x, scale_y)

    # Improve readability by giving headers a slightly stronger appearance.
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    """
    Save a matplotlib figure with consistent export settings.
    """
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Section 5.1: Algebraic exactness and convergence behavior
# =============================================================================

def spectral_summary(A: np.ndarray, eps: float = 1e-8) -> Tuple[int, float]:
    """
    Estimate the number of distinct eigenvalues and the condition number.

    Parameters
    ----------
    A : ndarray
        Symmetric positive definite matrix.
    eps : float
        Tolerance used to cluster eigenvalues numerically.

    Returns
    -------
    distinct_count : int
        Estimated number of distinct eigenvalues.
    cond_number : float
        Spectral condition number.
    """
    eigvals = np.linalg.eigvalsh(A)
    cond_number = float(eigvals.max() / eigvals.min())

    distinct = [eigvals[0]]
    for val in eigvals[1:]:
        if abs(val - distinct[-1]) > eps:
            distinct.append(val)
    return len(distinct), cond_number


def exactness_probe(
    A: np.ndarray,
    n_rhs: int = 5,
    tol: float = 1e-12,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Probe how closely CG reproduces the direct solution for several random
    right-hand sides.

    Returns averaged diagnostics that are more numerically meaningful than an
    explicit monomial-coefficient reconstruction.
    """
    rng = np.random.default_rng(seed)
    rel_errors = []
    final_residuals = []
    iterations = []
    histories = []

    for _ in range(n_rhs):
        b = rng.normal(size=A.shape[0])
        x_star = np.linalg.solve(A, b)
        x_cg, info = cg(A, b, max_iter=None, tol=tol, return_history=True)
        rel_err = np.linalg.norm(x_cg - x_star) / (np.linalg.norm(x_star) + 1e-15)

        rel_errors.append(rel_err)
        final_residuals.append(info["final_residual"])
        iterations.append(info["iterations"])
        histories.append(info["residual_history"])

    distinct_eigs, cond_number = spectral_summary(A)
    return {
        "distinct_eigenvalues": distinct_eigs,
        "condition_number": cond_number,
        "mean_cg_iterations": float(np.mean(iterations)),
        "std_cg_iterations": float(np.std(iterations)),
        "mean_relative_solution_error": float(np.mean(rel_errors)),
        "mean_final_residual": float(np.mean(final_residuals)),
        "residual_histories": histories,
    }


def build_exactness_outputs(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Generate all outputs for subsection 5.1.

    Files produced
    --------------
    - exactness_summary.csv / .png
    - residual_decay_curves.png
    - error_vs_iteration_breast_cancer.png
    """
    rows = []
    residual_curves = {}
    error_curve_reference = None

    for dataset_name, (X, y, feature_names) in datasets.items():
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xa = augment_bias(Xs)
        Y = one_hot(y)

        # A moderate lambda is used to build a well-conditioned system for the
        # algebraic probe. This is not a tuned predictive setting; it is an
        # operator analysis setting.
        lam = 1.0
        A, B = regularized_system(Xa, Y, lam)
        probe = exactness_probe(A, n_rhs=5, tol=1e-12, seed=42)

        rows.append({
            "Dataset": dataset_name,
            "Augmented dimension": Xa.shape[1],
            "Distinct eigenvalues": probe["distinct_eigenvalues"],
            "Condition number": probe["condition_number"],
            "Mean CG iterations": probe["mean_cg_iterations"],
            "Mean relative solution error": probe["mean_relative_solution_error"],
            "Mean final residual": probe["mean_final_residual"],
        })

        # Store one representative residual history per dataset for plotting.
        residual_curves[dataset_name] = probe["residual_histories"][0]

        # Build an error-vs-iteration curve for one representative dataset.
        if dataset_name == "breast_cancer":
            rng = np.random.default_rng(42)
            b = rng.normal(size=A.shape[0])
            x_star = np.linalg.solve(A, b)
            errors = []
            residuals = []
            max_iter = A.shape[0]
            for k in range(1, max_iter + 1):
                x_k, info_k = cg(A, b, max_iter=k, tol=0.0, return_history=False)
                rel_err = np.linalg.norm(x_k - x_star) / (np.linalg.norm(x_star) + 1e-15)
                errors.append(rel_err)
                residuals.append(np.linalg.norm(b - A @ x_k) / (np.linalg.norm(b) + 1e-15))
            error_curve_reference = {
                "dataset": dataset_name,
                "errors": errors,
                "residuals": residuals,
            }

    summary_df = pd.DataFrame(rows)
    summary_df["Condition number"] = summary_df["Condition number"].map(lambda v: f"{v:.2f}")
    summary_df["Mean CG iterations"] = summary_df["Mean CG iterations"].map(lambda v: f"{v:.1f}")
    summary_df["Mean relative solution error"] = summary_df["Mean relative solution error"].map(lambda v: f"{v:.2e}")
    summary_df["Mean final residual"] = summary_df["Mean final residual"].map(lambda v: f"{v:.2e}")

    summary_csv = out_dir / "table_1_exactness_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    dataframe_to_png(summary_df, out_dir / "table_1_exactness_summary.png", title="Table 1. Exactness and convergence diagnostics")

    # Figure 1: semilog residual decay curves.
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    for dataset_name, history in residual_curves.items():
        ax1.semilogy(range(len(history)), history, marker="o", markersize=3, label=dataset_name)
    ax1.set_xlabel("CG iteration")
    ax1.set_ylabel("Relative residual norm")
    ax1.set_title("Figure 1. Residual decay across benchmark datasets")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    save_figure(fig1, out_dir / "figure_1_residual_decay_curves.png")

    # Figure 2: error vs. iteration on the most ill-conditioned representative case.
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    ax2.semilogy(range(1, len(error_curve_reference["errors"]) + 1), error_curve_reference["errors"], label="Relative solution error")
    ax2.semilogy(range(1, len(error_curve_reference["residuals"]) + 1), error_curve_reference["residuals"], label="Relative residual")
    ax2.set_xlabel("Iteration budget")
    ax2.set_ylabel("Relative magnitude")
    ax2.set_title("Figure 2. Error and residual trajectories on breast_cancer")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    save_figure(fig2, out_dir / "figure_2_error_vs_iteration_breast_cancer.png")

    return summary_df


# =============================================================================
# Section 5.2: Predictive performance and efficiency
# =============================================================================

def evaluate_predictive_performance(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    out_dir: Path,
    outer_splits: int = 5,
    outer_repeats: int = 2,
) -> pd.DataFrame:
    """
    Run repeated stratified cross-validation for all methods.

    Methods
    -------
    - direct : dense regularized least squares
    - cg     : full conjugate-gradient regularized least squares
    - tcg    : truncated conjugate-gradient regularized least squares
    - logreg : logistic-regression baseline

    Returns
    -------
    results_df : DataFrame
        Fold-level performance results.
    """
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    tcg_iterations = [1, 2, 3, 5, 8, 13, 21]
    C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    rows = []

    for dataset_name, (X, y, feature_names) in datasets.items():
        cv = RepeatedStratifiedKFold(
            n_splits=outer_splits,
            n_repeats=outer_repeats,
            random_state=42,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # -----------------------------------------------------------------
            # Direct regularized least squares
            # -----------------------------------------------------------------
            best_lam_direct = tune_direct_rls(X_train, y_train, lambdas)
            start = time.perf_counter()
            model_direct = direct_rls_fit(X_train, y_train, best_lam_direct)
            fit_time = time.perf_counter() - start
            pred_direct = predict_labels(model_direct, X_test)

            rows.append({
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": "direct",
                "accuracy": accuracy_score(y_test, pred_direct),
                "macro_f1": f1_score(y_test, pred_direct, average="macro"),
                "fit_time": fit_time,
                "lambda": best_lam_direct,
                "iterations": np.nan,
            })

            # -----------------------------------------------------------------
            # Full CG regularized least squares
            # -----------------------------------------------------------------
            best_lam_cg = tune_cg_rls(X_train, y_train, lambdas)
            start = time.perf_counter()
            model_cg = cg_rls_fit(X_train, y_train, best_lam_cg, max_iter=None, tol=1e-10)
            fit_time = time.perf_counter() - start
            pred_cg = predict_labels(model_cg, X_test)
            mean_iters = float(np.mean([info["iterations"] for info in model_cg["infos"]]))

            rows.append({
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": "cg",
                "accuracy": accuracy_score(y_test, pred_cg),
                "macro_f1": f1_score(y_test, pred_cg, average="macro"),
                "fit_time": fit_time,
                "lambda": best_lam_cg,
                "iterations": mean_iters,
            })

            # -----------------------------------------------------------------
            # Truncated CG regularized least squares
            # -----------------------------------------------------------------
            best_lam_tcg, best_k_tcg = tune_tcg_rls(X_train, y_train, lambdas, tcg_iterations)
            start = time.perf_counter()
            model_tcg = cg_rls_fit(X_train, y_train, best_lam_tcg, max_iter=best_k_tcg, tol=1e-10)
            fit_time = time.perf_counter() - start
            pred_tcg = predict_labels(model_tcg, X_test)

            rows.append({
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": "tcg",
                "accuracy": accuracy_score(y_test, pred_tcg),
                "macro_f1": f1_score(y_test, pred_tcg, average="macro"),
                "fit_time": fit_time,
                "lambda": best_lam_tcg,
                "iterations": best_k_tcg,
            })

            # -----------------------------------------------------------------
            # Logistic regression baseline
            # -----------------------------------------------------------------
            best_C = tune_logreg(X_train, y_train, C_grid)
            start = time.perf_counter()
            model_lr = fit_logistic_regression(X_train, y_train, C=best_C, max_iter=2000)
            fit_time = time.perf_counter() - start
            pred_lr = predict_logreg(model_lr, X_test)

            rows.append({
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": "logreg",
                "accuracy": accuracy_score(y_test, pred_lr),
                "macro_f1": f1_score(y_test, pred_lr, average="macro"),
                "fit_time": fit_time,
                "lambda": np.nan,
                "iterations": np.nan,
                "C": best_C,
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "fold_level_results.csv", index=False)

    # Create manuscript-ready summary tables.
    perf_rows = []
    for (dataset_name, method), grp in results_df.groupby(["dataset", "method"]):
        perf_rows.append({
            "Dataset": dataset_name,
            "Method": method,
            "Accuracy": format_mean_std(grp["accuracy"], decimals=3),
            "Macro-F1": format_mean_std(grp["macro_f1"], decimals=3),
        })
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(out_dir / "table_2_predictive_performance.csv", index=False)
    dataframe_to_png(perf_df, out_dir / "table_2_predictive_performance.png", title="Table 2. Predictive performance across methods")

    time_rows = []
    for (dataset_name, method), grp in results_df.groupby(["dataset", "method"]):
        time_rows.append({
            "Dataset": dataset_name,
            "Method": method,
            "Fit time (s)": format_mean_std(grp["fit_time"], decimals=4),
            "Mean iterations": "-" if grp["iterations"].isna().all() else format_mean_std(grp["iterations"].dropna(), decimals=1),
        })
    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(out_dir / "table_3_efficiency_summary.csv", index=False)
    dataframe_to_png(time_df, out_dir / "table_3_efficiency_summary.png", title="Table 3. Computational efficiency summary")

    # Figure 3: mean accuracy by method and dataset.
    mean_acc = results_df.groupby(["dataset", "method"])["accuracy"].mean().unstack("method")
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    mean_acc.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Mean accuracy")
    ax3.set_xlabel("Dataset")
    ax3.set_title("Figure 3. Mean classification accuracy across methods")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend(title="Method")
    save_figure(fig3, out_dir / "figure_3_mean_accuracy_by_method.png")

    return results_df


# =============================================================================
# Section 5.3: Stability, robustness, and interpretability
# =============================================================================

def add_gaussian_noise_standardized(
    X: np.ndarray,
    mean_: np.ndarray,
    scale_: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise in standardized feature space and map the result back to
    the original raw feature space.

    Why this formulation?
    ---------------------
    Adding the same raw-scale perturbation to every feature can produce
    misleading robustness conclusions when variables have very different units
    or dispersions. Because the predictive models are trained after
    standardization, a perturbation defined in standardized coordinates is more
    comparable across features and datasets.
    """
    Xs = (X - mean_) / scale_
    Xs_noisy = Xs + rng.normal(loc=0.0, scale=noise_std, size=Xs.shape)
    return mean_ + Xs_noisy * scale_


def robustness_analysis(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Evaluate how prediction accuracy changes under additive Gaussian noise.

    The analysis is intentionally straightforward:
    - fixed stratified train/test split
    - hyperparameters selected on the clean training set
    - noisy evaluation performed on the test set only

    This design isolates robustness effects without entangling them with
    repeated resampling noise.
    """
    rng = np.random.default_rng(123)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    tcg_iterations = [1, 2, 3, 5, 8, 13, 21]

    rows = []
    trained_models = {}

    for dataset_name, (X, y, feature_names) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=123
        )

        # Tune and train direct model.
        best_lam_direct = tune_direct_rls(X_train, y_train, lambdas)
        model_direct = direct_rls_fit(X_train, y_train, best_lam_direct)

        # Tune and train full CG model.
        best_lam_cg = tune_cg_rls(X_train, y_train, lambdas)
        model_cg = cg_rls_fit(X_train, y_train, best_lam_cg, max_iter=None, tol=1e-10)

        # Tune and train truncated CG model.
        best_lam_tcg, best_k_tcg = tune_tcg_rls(X_train, y_train, lambdas, tcg_iterations)
        model_tcg = cg_rls_fit(X_train, y_train, best_lam_tcg, max_iter=best_k_tcg, tol=1e-10)

        trained_models[dataset_name] = {
            "direct": model_direct,
            "cg": model_cg,
            "tcg": model_tcg,
            "feature_names": feature_names,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        # Noise is injected in standardized coordinates so that the perturbation
        # magnitude is comparable across features with different physical scales.
        # The same corrupted test set is used for all methods to make the
        # robustness comparison fair.
        X_test_noisy = add_gaussian_noise_standardized(
            X_test,
            mean_=model_direct["scaler_mean"],
            scale_=model_direct["scaler_scale"],
            noise_std=0.25,
            rng=rng,
        )

        for method_name, model in [("direct", model_direct), ("cg", model_cg), ("tcg", model_tcg)]:
            clean_pred = predict_labels(model, X_test)
            noisy_pred = predict_labels(model, X_test_noisy)

            clean_acc = accuracy_score(y_test, clean_pred)
            noisy_acc = accuracy_score(y_test, noisy_pred)

            rows.append({
                "Dataset": dataset_name,
                "Method": method_name,
                "Clean accuracy": clean_acc,
                "Noisy accuracy": noisy_acc,
                "Absolute drop": clean_acc - noisy_acc,
            })

    robustness_df = pd.DataFrame(rows)
    robustness_df["Clean accuracy"] = robustness_df["Clean accuracy"].map(lambda v: f"{v:.3f}")
    robustness_df["Noisy accuracy"] = robustness_df["Noisy accuracy"].map(lambda v: f"{v:.3f}")
    robustness_df["Absolute drop"] = robustness_df["Absolute drop"].map(lambda v: f"{v:.3f}")
    robustness_df.to_csv(out_dir / "table_4_robustness_summary.csv", index=False)
    dataframe_to_png(robustness_df, out_dir / "table_4_robustness_summary.png", title="Table 4. Robustness under additive Gaussian noise")

    return robustness_df, trained_models


def tcg_iteration_curve(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Create an accuracy-vs-iteration curve for the truncated CG model on a
    representative dataset.

    The manuscript uses the breast_cancer dataset because it combines practical
    relevance, moderate dimensionality, and a noticeably nontrivial condition
    number.
    """
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    iteration_grid = list(range(1, 31))

    curve_rows = []
    # Use the best lambda according to the truncated model's tuning rule.
    tuned_lam, _ = tune_tcg_rls(X_train, y_train, lambdas, [1, 2, 3, 5, 8, 13, 21])

    for k in iteration_grid:
        model = cg_rls_fit(X_train, y_train, tuned_lam, max_iter=k, tol=1e-10)
        pred = predict_labels(model, X_test)
        acc = accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro")
        curve_rows.append({
            "Iteration": k,
            "Accuracy": acc,
            "Macro-F1": macro,
        })

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"tcg_iteration_curve_{dataset_name}.csv", index=False)

    fig4, ax4 = plt.subplots(figsize=(8, 5.5))
    ax4.plot(curve_df["Iteration"], curve_df["Accuracy"], marker="o", label="Accuracy")
    ax4.plot(curve_df["Iteration"], curve_df["Macro-F1"], marker="s", label="Macro-F1")
    ax4.set_xlabel("TCG iteration budget")
    ax4.set_ylabel("Score")
    ax4.set_title(f"Figure 4. Truncated-CG performance trajectory on {dataset_name}")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    save_figure(fig4, out_dir / f"figure_4_tcg_iteration_curve_{dataset_name}.png")

    return curve_df


def feature_importance_figure(
    dataset_name: str,
    model: Dict[str, np.ndarray],
    feature_names: List[str],
    out_dir: Path,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Build a simple interpretability figure from absolute weight magnitudes.

    For multiclass problems, the average absolute coefficient across classes is
    used. For binary problems encoded through a two-column target matrix, the
    same averaging rule still provides a stable importance proxy.
    """
    # Exclude the bias row because it is not an input feature.
    W = model["weights"][:-1, :]
    importance = np.mean(np.abs(W), axis=1)

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False).head(top_k)

    imp_df.to_csv(out_dir / f"feature_importance_{dataset_name}.csv", index=False)

    fig5, ax5 = plt.subplots(figsize=(9, 5.5))
    ax5.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1])
    ax5.set_xlabel("Mean absolute coefficient magnitude")
    ax5.set_title(f"Figure 5. Top {top_k} feature importances on {dataset_name}")
    ax5.grid(axis="x", alpha=0.3)
    save_figure(fig5, out_dir / f"figure_5_feature_importance_{dataset_name}.png")

    return imp_df


# =============================================================================
# README and packaging
# =============================================================================

def build_readme(root_dir: Path) -> None:
    """
    Write a detailed README describing the generated assets.
    """
    content = """
# Cayley–Hamilton-Guided Krylov Learning Package

This package contains the code and results that support the article
"Cayley–Hamilton-Guided Krylov Regularization for Machine Learning:
Theory, Computational Architecture, and Empirical Evidence".

## Folder structure

- `results/5.1/`
  - Algebraic exactness and convergence diagnostics.
  - Includes a summary table, residual decay curves, and an error-vs-iteration figure.

- `results/5.2/`
  - Predictive performance and computational efficiency.
  - Includes fold-level CSV data, summary tables, and a comparative accuracy figure.

- `results/5.3/`
  - Stability, robustness, and interpretability.
  - Includes a noise-robustness table, a truncated-CG iteration curve,
    and a feature-importance figure.

## Main code file

- `ch_krylov_learning_pipeline.py`
  - End-to-end reproducible implementation.
  - Generates all result tables and figures.
  - Creates the ZIP archive.

## Reproduction

Run:

```bash
python ch_krylov_learning_pipeline.py
```

All outputs will be written inside the `results/` directory located next to the script.

## Notes

1. The implementation uses scikit-learn benchmark datasets available offline.
2. The conjugate-gradient solver is the computational realization of the
   finite polynomial / Krylov interpretation motivated by the Cayley-Hamilton theorem.
3. The explicit computation of characteristic-polynomial coefficients is not used
   as the primary numerical mechanism because it is less stable in moderate to
   high dimension than Krylov-space iteration.

## Files intended for manuscript integration

- `results/5.1/table_1_exactness_summary.csv`
- `results/5.1/figure_1_residual_decay_curves.png`
- `results/5.1/figure_2_error_vs_iteration_breast_cancer.png`

- `results/5.2/table_2_predictive_performance.csv`
- `results/5.2/table_3_efficiency_summary.csv`
- `results/5.2/figure_3_mean_accuracy_by_method.png`

- `results/5.3/table_4_robustness_summary.csv`
- `results/5.3/figure_4_tcg_iteration_curve_breast_cancer.png`
- `results/5.3/figure_5_feature_importance_breast_cancer.png`
"""
    save_text(root_dir / "README.md", content.strip() + "\n")


def build_requirements(root_dir: Path) -> None:
    """
    Save the package requirements.
    """
    content = """
numpy
pandas
matplotlib
scikit-learn
"""
    save_text(root_dir / "requirements.txt", content.strip() + "\n")


def zip_project(root_dir: Path, zip_path: Path) -> None:
    """
    Create a ZIP archive containing the complete project package.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in root_dir.rglob("*"):
            if file_path == zip_path:
                continue
            zf.write(file_path, arcname=file_path.relative_to(root_dir))


# =============================================================================
# Main orchestration
# =============================================================================

def main() -> None:
    """
    End-to-end orchestration of the full experimental pipeline.
    """
    rng = set_global_seed(42)

    # Establish a clean directory structure.
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent / "ch_krylov_package"
    if root_dir.exists():
        shutil.rmtree(root_dir)
    ensure_dir(root_dir)

    results_dir = root_dir / "results"
    out_51 = results_dir / "5.1"
    out_52 = results_dir / "5.2"
    out_53 = results_dir / "5.3"

    for p in [results_dir, out_51, out_52, out_53]:
        ensure_dir(p)

    # Load data once.
    datasets = load_datasets()

    # Generate results for section 5.1.
    exactness_df = build_exactness_outputs(datasets, out_51)

    # Generate results for section 5.2.
    predictive_df = evaluate_predictive_performance(datasets, out_52, outer_splits=5, outer_repeats=2)

    # Generate results for section 5.3.
    robustness_df, trained_models = robustness_analysis(datasets, out_53)

    # Representative truncated-CG iteration curve.
    breast_bundle = trained_models["breast_cancer"]
    curve_df = tcg_iteration_curve(
        "breast_cancer",
        breast_bundle["X_train"],
        breast_bundle["y_train"],
        breast_bundle["X_test"],
        breast_bundle["y_test"],
        out_53,
    )

    # Feature importance figure for the representative dataset.
    importance_df = feature_importance_figure(
        "breast_cancer",
        breast_bundle["direct"],
        breast_bundle["feature_names"],
        out_53,
        top_k=10,
    )

    # Save compact JSON metadata to facilitate manuscript writing.
    metadata = {
        "datasets": list(datasets.keys()),
        "files_generated": sorted([str(p.relative_to(root_dir)) for p in root_dir.rglob("*") if p.is_file()]),
    }
    save_text(root_dir / "metadata.json", json.dumps(metadata, indent=2))

    # Bundle source code and documentation.
    shutil.copy2(script_path, root_dir / script_path.name)
    build_readme(root_dir)
    build_requirements(root_dir)

    # Package everything in a ZIP archive.
    zip_project(root_dir, root_dir / "ch_krylov_results_package.zip")

    print("Pipeline completed successfully.")
    print(f"Package root: {root_dir}")
    print(f"ZIP archive: {root_dir / 'ch_krylov_results_package.zip'}")


if __name__ == "__main__":
    main()
