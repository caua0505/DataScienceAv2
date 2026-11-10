import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PATH,
    FEATURES,
    K_RANGE,
    MODEL_PATH,
    N_INIT,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.preprocessing import build_preprocessor, centroids_to_original_scale, load_dataset


def make_kmeans(n_clusters: int) -> KMeans:
    return KMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=RANDOM_STATE)


def compute_elbow(X_transformed: np.ndarray, k_values=K_RANGE) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = make_kmeans(k).fit(X_transformed)
        silhouette = silhouette_score(X_transformed, model.labels_) if k >= 2 else np.nan
        rows.append({"k": k, "inercia_wcss": model.inertia_, "silhouette": silhouette})
    return pd.DataFrame(rows)


def find_elbow_k(k_values, inertias) -> int:
    """Ponto de cotovelo: maior distância da curva normalizada à reta entre os extremos."""
    k = np.asarray(k_values, dtype=float)
    w = np.asarray(inertias, dtype=float)
    k_norm = (k - k.min()) / (k.max() - k.min())
    w_norm = (w - w.min()) / (w.max() - w.min())
    distance_below_chord = (1.0 - k_norm) - w_norm
    return int(k[np.argmax(distance_below_chord)])


def plot_elbow(elbow: pd.DataFrame, chosen_k: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow["k"], elbow["inercia_wcss"], marker="o", color="#1f5f99")
    chosen = elbow.loc[elbow["k"] == chosen_k].iloc[0]
    ax.plot([elbow["k"].iloc[0], elbow["k"].iloc[-1]],
            [elbow["inercia_wcss"].iloc[0], elbow["inercia_wcss"].iloc[-1]],
            linestyle="--", color="#999999", linewidth=1, label="Reta entre extremos")
    ax.scatter([chosen_k], [chosen["inercia_wcss"]], s=180, facecolors="none",
               edgecolors="#d1495b", linewidths=2, zorder=3, label=f"Cotovelo: k = {chosen_k}")
    ax.set_xticks(elbow["k"])
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inércia (WCSS)")
    ax.set_title("Método do cotovelo - K-Means")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_silhouette(elbow: pd.DataFrame, chosen_k: int, path: Path) -> None:
    data = elbow.dropna(subset=["silhouette"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["k"], data["silhouette"], marker="o", color="#2a9d8f")
    chosen = data.loc[data["k"] == chosen_k, "silhouette"]
    if not chosen.empty:
        ax.scatter([chosen_k], chosen, s=180, facecolors="none", edgecolors="#d1495b",
                   linewidths=2, zorder=3, label=f"k escolhido = {chosen_k}")
        ax.legend()
    ax.set_xticks(data["k"])
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Coeficiente de Silhouette")
    ax.set_title("Silhouette por k (métrica complementar)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cluster_distribution(counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([f"Cluster {c}" for c in counts.index], counts.values, color="#1f5f99")
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value} ({value / total:.1%})", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Quantidade de residências")
    ax.set_title("Distribuição dos registros por cluster")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_cluster_profiles(pipeline: Pipeline, X_transformed: np.ndarray, target: pd.Series) -> pd.DataFrame:
    kmeans = pipeline.named_steps["kmeans"]
    labels = kmeans.labels_
    profiles = centroids_to_original_scale(pipeline.named_steps["preprocessor"], kmeans.cluster_centers_)
    profiles.insert(0, "cluster", range(kmeans.n_clusters))

    member_distance = kmeans.transform(X_transformed)[np.arange(len(labels)), labels]
    stats = pd.DataFrame({"cluster": labels, "dist": member_distance, "medv": target.to_numpy()})
    grouped = stats.groupby("cluster")
    profiles.insert(1, "n_registros", grouped.size().reindex(profiles["cluster"]).to_numpy())
    profiles.insert(2, "percentual", profiles["n_registros"] / len(labels))
    profiles["distancia_media_membros"] = grouped["dist"].mean().reindex(profiles["cluster"]).to_numpy()
    profiles[f"{TARGET_COLUMN}_media_informativa"] = grouped["medv"].mean().reindex(profiles["cluster"]).to_numpy()
    return profiles.set_index("cluster", drop=False)


def train(
    n_clusters: int | None = None,
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    model_path, reports_dir = Path(model_path), Path(reports_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X = df[FEATURES]
    target = df[TARGET_COLUMN]

    X_elbow = build_preprocessor().fit_transform(X)
    elbow = compute_elbow(X_elbow)
    elbow_k = find_elbow_k(elbow["k"], elbow["inercia_wcss"])
    chosen_k = n_clusters or elbow_k
    if chosen_k not in set(elbow["k"]):
        raise ValueError(f"k deve estar entre {min(K_RANGE)} e {max(K_RANGE)} (recebido: {chosen_k})")
    elbow["reducao_inercia_pct"] = -elbow["inercia_wcss"].pct_change() * 100

    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("kmeans", make_kmeans(chosen_k))])
    pipeline.fit(X)
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)
    profiles = build_cluster_profiles(pipeline, X_transformed, target)

    elbow.to_csv(reports_dir / "elbow_results.csv", index=False)
    profiles.to_csv(reports_dir / "cluster_centroids.csv", index=False)
    plot_elbow(elbow, chosen_k, reports_dir / "elbow_method.png")
    plot_silhouette(elbow, chosen_k, reports_dir / "silhouette_scores.png")
    plot_cluster_distribution(profiles["n_registros"], reports_dir / "cluster_distribution.png")

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_clusters": chosen_k,
        "elbow_k": elbow_k,
        "cluster_profiles": profiles,
        "elbow_results": elbow,
        "feature_ranges": {f: (float(X[f].min()), float(X[f].max())) for f in NUMERIC_FEATURES},
        "n_samples": len(df),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, model_path)
    return artifact


def print_summary(artifact: dict, model_path: Path, reports_dir: Path) -> None:
    elbow = artifact["elbow_results"]
    profiles = artifact["cluster_profiles"]
    print("=" * 72)
    print("TREINAMENTO K-MEANS - BOSTON HOUSING")
    print("=" * 72)
    print(f"Registros utilizados: {artifact['n_samples']} | Features: {len(artifact['features'])} "
          f"(MEDV excluída)")
    print("\nMétodo do cotovelo:")
    print(elbow.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\nCotovelo detectado: k = {artifact['elbow_k']}")
    print(f"k utilizado no modelo final: k = {artifact['n_clusters']}")
    print("\nPerfis dos clusters (centroides na escala original):")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(profiles.drop(columns="cluster").round(3).to_string())
    print(f"\nModelo salvo em: {model_path}")
    print(f"Relatórios salvos em: {reports_dir}")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Treina o K-Means sobre a base Boston Housing.")
    parser.add_argument("--k", type=int, default=None,
                        help="Força um número de clusters (padrão: cotovelo detectado).")
    args = parser.parse_args(argv)
    try:
        artifact = train(n_clusters=args.k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}")
        return 1
    print_summary(artifact, MODEL_PATH, REPORTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
