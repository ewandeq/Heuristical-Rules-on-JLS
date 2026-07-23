from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def _subset_rows(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def _get_feature_names(X, feature_names):
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"f{i}" for i in range(X.shape[1])]


def run_agent_tree(X, y_agent, out_dir=".", feature_names=None, class_labels=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = pd.Series(y_agent)
    labels = list(class_labels) if class_labels is not None else sorted(pd.unique(y))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    base_model = DecisionTreeClassifier(class_weight="balanced", random_state=0)
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }

    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        refit="f1_macro",
        cv=cv,
    )
    grid.fit(X, y)

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(out_dir / "cv_results_agent.csv", index=False)

    best_idx = grid.best_index_
    metrics_df = pd.DataFrame(
        [
            {
                "best_params": json.dumps(grid.best_params_, sort_keys=True),
                "mean_f1_macro": cv_results.loc[best_idx, "mean_test_f1_macro"],
                "std_f1_macro": cv_results.loc[best_idx, "std_test_f1_macro"],
                "mean_accuracy": cv_results.loc[best_idx, "mean_test_accuracy"],
                "std_accuracy": cv_results.loc[best_idx, "std_test_accuracy"],
            }
        ]
    )
    metrics_df.to_csv(out_dir / "metrics_agent.csv", index=False)
    print("Best CV metrics:")
    print(metrics_df.to_string(index=False))

    cm_total = np.zeros((len(labels), len(labels)), dtype=int)
    oof_true = []
    oof_pred = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = _subset_rows(X, train_idx)
        X_test = _subset_rows(X, test_idx)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = DecisionTreeClassifier(
            class_weight="balanced", random_state=0, **grid.best_params_
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oof_true.extend(y_test)
        oof_pred.extend(preds)
        cm_total += confusion_matrix(y_test, preds, labels=labels)

    print("classification_report (CV aggregated):")
    print(classification_report(oof_true, oof_pred, labels=labels, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix (CV aggregated)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_agent_confusion.png", dpi=150)
    plt.close(fig)

    final_model = DecisionTreeClassifier(
        class_weight="balanced", random_state=0, **grid.best_params_
    )
    final_model.fit(X, y)

    feature_names = _get_feature_names(X, feature_names)
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    top15 = importances.head(15)
    print("Top 15 feature importances:")
    print(top15.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances (Agent)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_agent_importances.png", dpi=150)
    plt.close(fig)

    dot_path = out_dir / "tree_agent.dot"
    export_graphviz(
        final_model,
        out_file=str(dot_path),
        feature_names=feature_names,
        class_names=labels,
        filled=True,
        rounded=True,
        special_characters=False,
    )
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(out_dir / "tree_agent.png")],
            check=True,
        )
    except Exception as exc:
        print(f"Graphviz export failed: {exc}. Dot file saved to {dot_path}")

    return {
        "grid": grid,
        "cv_results": cv_results,
        "metrics": metrics_df,
        "confusion_matrix": cm_total,
        "final_model": final_model,
        "feature_importances": importances,
    }
