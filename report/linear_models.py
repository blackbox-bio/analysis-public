from itertools import combinations
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .utils import summary_qc
from palmreader_analysis.events import Palmreader

def _plot_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    import matplotlib.transforms as transforms
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, theta, fill=False, **kwargs)
    ax.add_patch(ellipse)

def fit_lda_model(
    df: pd.DataFrame,
    group_variable: str,
    shrinkage: str | float = 'auto',
):
    """
        Fit a shrinkage-regularized LDA model on numeric features and compute
        feature importance at global and per-class levels.

        Parameters
        ----------
        df : pd.DataFrame
            Must include numeric feature columns and one categorical group column.
        group_variable : str
            Column containing group labels.
        shrinkage : str | float
            LDA shrinkage parameter ('auto' = Ledoit–Wolf regularization).

        Returns
        -------
        results : dict
            {
              "scaler": fitted StandardScaler,
              "lda": fitted LDA model,
              "scores_LD": array (n_samples, n_classes-1),
              "posteriors": class probabilities,
              "correlation": global correlation matrix of the features (DataFrame),
              "classes": class labels,
              "X_scaled": standardized feature matrix,
              "feature_names": feature names,
              "coefficients": global LDA coefficients (DataFrame),
              "class_means": per-class means (DataFrame),
              "class_importance": per-class feature importance (DataFrame)
            }
        """

    # --- clean up the dataframe ---
    df = summary_qc(df, group_variable)

    # --- Extract X, y ---
    y = df[group_variable].values
    X = df.drop(columns=[group_variable]).select_dtypes(include=np.number)
    feature_names = X.columns

    #--- check and make sure the group variable contains more than one group ---
    # TODO: come back to this and make the palmreader software handle this situation better
    if df[group_variable].nunique() < 2:
        Palmreader.warning(
            f"The group variable {group_variable} must contain more than one group. Exiting.")
        print(f"Warning: The group variable {group_variable} must contain more than one group. Exiting.")

        return None

    # --- Standardize features ---
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # --- Fit shrinkage-regularized LDA ---
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage, store_covariance=True)
    lda.fit(X_scaled, y)

    # --- Projected scores & posteriors ---
    scores = lda.transform(X_scaled)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    posteriors = lda.predict_proba(X_scaled)

    # --- LDA global correlation matrix ---
    cov = lda.covariance_
    corr_df = pd.DataFrame(cov, index=feature_names, columns=feature_names).corr()

    # --- LDA coefficients (global importance) ---
    coef = lda.coef_.T  # shape (n_features, n_classes-1)
    coef_df = pd.DataFrame(
        coef,
        index=feature_names,
        columns=[f"LD{i + 1}" for i in range(coef.shape[1])],
    )
    coef_df["abs_LD1"] = coef_df["LD1"].abs()
    coef_df = coef_df.sort_values("abs_LD1", ascending=False)

    # --- Per-class means & pairwise differences ---
    mean_df = (
        df.groupby(group_variable)[feature_names]
        .mean()
        .T  # features as rows, groups as columns
    )

    # --- Class-specific feature importance ---
    # 1. Average coefficients across all discriminant axes
    mean_coef = coef.mean(axis=1)  # (n_features,)

    # 2. Standardize class means
    mean_std = pd.DataFrame(
        scaler.transform(mean_df.T),
        columns=feature_names,
        index=mean_df.columns,
    ).T  # features × classes

    # 3. Compute per-class feature importance
    #    multiply each feature’s standardized mean by its global discriminant weight
    class_importance_df = mean_std.mul(mean_coef, axis=0)

    return dict(
        scaler=scaler,
        lda=lda,
        scores_LD=scores,
        posteriors=posteriors,
        correlation=corr_df,
        classes=lda.classes_,
        X_scaled=X_scaled,
        y=y,
        feature_names=feature_names,
        coefficients=coef_df,
        class_means=mean_df,
        class_importance=class_importance_df,
    )


def plot_lda_projection(lda_result: dict,
                        point_size: int = 60,
                        ring_std: float = 1.0,
                        dest_path: Optional[str] = None,
):
    """
    Visualize LDA results in 2D with Gaussian ellipse contours around group means.

    Parameters
    ----------
    lda_result : dict
        Output of `fit_lda_model`.
    point_size : int
        Scatter point size.
    ring_std : float
        Radius of the ellipse in standard deviations (1 = 1σ ellipse).
    dest_path : str, optional
        Destination path for the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """

    y = lda_result["y"]
    classes = lda_result["classes"]
    scores = lda_result["scores_LD"]
    post = lda_result["posteriors"]
    K = len(classes)
    palette = sns.color_palette("husl", K)

    fig, ax = plt.subplots(figsize=(7, 6))

    if K >= 3:
        x, y2 = scores[:, 0], scores[:, 1] if scores.shape[1] >= 2 else np.zeros_like(scores[:, 0])

        for idx, cls in enumerate(classes):
            mask = (y == cls)
            color = palette[idx]

            # Scatter points
            ax.scatter(x[mask], y2[mask], s=point_size, alpha=0.75, label=str(cls), c=[color])

            # Mean and covariance
            mean = np.mean(np.column_stack((x[mask], y2[mask])), axis=0)
            cov = np.cov(x[mask], y2[mask])

            # Eigen decomposition for ellipse orientation and axes
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * ring_std * np.sqrt(vals)

            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=theta,
                edgecolor=color,
                facecolor='none',
                lw=2,
                alpha=0.9,
            )
            ax.add_patch(ellipse)

            # Centroid marker
            ax.scatter(*mean, c=[color], s=140, edgecolors="k", marker="X", zorder=5)

        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA (LD1 vs LD2, Gaussian Ellipses)")

    else:
        # Binary: LD1 vs logit
        ld1 = scores[:, 0]
        p1 = post[:, 1]
        logit = np.log((p1 + 1e-6) / (1 - p1 + 1e-6))

        for idx, cls in enumerate(classes):
            mask = (y == cls)
            color = palette[idx]
            ax.scatter(ld1[mask], logit[mask], s=point_size, alpha=0.75, label=str(cls), c=[color])

            # Mean ± std band
            mean = ld1[mask].mean()
            std = ld1[mask].std()
            ax.axvline(mean, color=color, lw=2, alpha=0.8)
            ax.fill_betweenx([logit.min(), logit.max()],
                             mean - std, mean + std,
                             color=color, alpha=0.15)

        ax.set_xlabel("LD1")
        ax.set_ylabel(f"logit P({classes[1]})")
        ax.set_title("Binary LDA: LD1 vs logit")

    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    # save fig if a dest_path is provided
    if dest_path:
        fig.savefig(dest_path,dpi=300, bbox_inches="tight")
        plt.close()

        return

    return fig


def plot_lda_class_feature_importance(
    lda_result: dict,
    top_n: int = 20,
    normalize: bool = True,
    figsize_per_class=(6, 5),
    cmap_pos="crimson",
    cmap_neg="steelblue",
):
    """
    Plot top discriminative features for each class from an LDA model.

    Parameters
    ----------
    lda_result : dict
        Output from `fit_lda_model()`. Must include `class_importance`.
    top_n : int
        Number of top features to display for each class.
    normalize : bool
        If True, normalize feature importance magnitudes per class (max = 1).
        Useful for comparing across classes.
    figsize_per_class : tuple
        (width, height) of each subplot. Total figure height scales with #classes.
    cmap_pos, cmap_neg : str
        Colors for positive vs negative weights.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    imp_df = lda_result["class_importance"]
    classes = imp_df.columns
    n_classes = len(classes)

    fig, axes = plt.subplots(
        n_classes, 1,
        figsize=(figsize_per_class[0], figsize_per_class[1] * n_classes),
        sharex=False
    )
    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        sub = imp_df[cls].copy()
        # Normalize magnitudes for better comparability
        if normalize and sub.abs().max() > 0:
            sub = sub / sub.abs().max()
        # Select top N by absolute magnitude
        sub = sub.reindex(sub.abs().sort_values(ascending=False).index)[:top_n]
        palette = [cmap_pos if val > 0 else cmap_neg for val in sub.values]

        sns.barplot(
            x=sub.values,
            y=sub.index,
            palette=palette,
            ax=axes[i]
        )
        axes[i].axvline(0, color="k", lw=1)
        axes[i].set_title(f"{cls}: top {top_n} features")
        axes[i].set_ylabel("")
        axes[i].set_xlabel("Normalized feature importance" if normalize else "Feature importance")

    plt.tight_layout()
    plt.show()
    return fig