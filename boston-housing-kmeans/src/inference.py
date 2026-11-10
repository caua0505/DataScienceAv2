import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessing import InvalidInputError, validate_field, validate_house

REQUIRED_ARTIFACT_KEYS = {"pipeline", "features", "cluster_profiles", "n_clusters"}


def load_artifact(path: Path = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Execute o treinamento: python -m src.train"
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not REQUIRED_ARTIFACT_KEYS.issubset(artifact):
        raise ValueError(f"Arquivo {path} não contém um modelo válido deste projeto.")
    if list(artifact["features"]) != FEATURES:
        raise ValueError("As features do modelo salvo não correspondem à configuração atual. Retreine o modelo.")
    return artifact


def predict_house(artifact: dict, house: dict) -> dict:
    X, warnings = validate_house(house)
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    kmeans = artifact["pipeline"].named_steps["kmeans"]

    X_transformed = preprocessor.transform(X)
    cluster = int(kmeans.predict(X_transformed)[0])
    distances = kmeans.transform(X_transformed)[0]
    profile = artifact["cluster_profiles"].loc[cluster]

    for feature, (low, high) in artifact.get("feature_ranges", {}).items():
        value = X.iloc[0][feature]
        if value < low or value > high:
            warnings.append(
                f"{feature}={value:g} está fora da faixa observada na base de treino "
                f"[{low:g}, {high:g}]; a atribuição ao cluster pode ser pouco representativa."
            )

    other = [(i, d) for i, d in enumerate(distances) if i != cluster]
    nearest_other = min(other, key=lambda item: item[1]) if other else None

    return {
        "id": house.get("id", "residencia_sem_id"),
        "observacao": house.get("observacao"),
        "cluster": cluster,
        "distancia_centroide": float(distances[cluster]),
        "distancia_media_membros": float(profile["distancia_media_membros"]),
        "segundo_cluster_mais_proximo": (
            {"cluster": int(nearest_other[0]), "distancia": float(nearest_other[1])} if nearest_other else None
        ),
        "entrada": X.iloc[0].to_dict(),
        "centroide": profile.to_dict(),
        "avisos": warnings,
    }


def format_result(result: dict) -> str:
    centroid = result["centroide"]
    lines = [
        "=" * 78,
        f"Residência: {result['id']}",
    ]
    if result.get("observacao"):
        lines.append(f"Observação: {result['observacao']}")
    lines += [
        "=" * 78,
        f"Cluster atribuído: {result['cluster']}  "
        f"({int(centroid['n_registros'])} residências da base, {centroid['percentual']:.1%})",
        f"Distância euclidiana ao centroide (espaço normalizado): {result['distancia_centroide']:.3f}",
        f"Distância média dos membros do cluster ao centroide:    {result['distancia_media_membros']:.3f}",
    ]
    second = result["segundo_cluster_mais_proximo"]
    if second:
        lines.append(f"Segundo cluster mais próximo: {second['cluster']} (distância {second['distancia']:.3f})")

    lines += [
        "",
        "Centroide do cluster na escala original (perfil médio do grupo):",
        f"{'Variável':<9} {'Residência':>12} {'Centroide':>12}  Descrição",
        "-" * 78,
    ]
    for feature in FEATURES:
        house_value = result["entrada"][feature]
        if feature in CATEGORICAL_FEATURES:
            predominant = int(centroid[f"{feature}_predominante"])
            share = centroid[f"{feature}_prop_{predominant}"]
            centroid_text = f"{predominant} ({share:.0%})"
            lines.append(f"{feature:<9} {int(house_value):>12} {centroid_text:>12}  {FEATURE_DESCRIPTIONS[feature]}")
        else:
            lines.append(f"{feature:<9} {house_value:>12.3f} {centroid[feature]:>12.3f}  {FEATURE_DESCRIPTIONS[feature]}")

    lines += [
        "-" * 78,
        "CHAS no centroide: categoria predominante e proporção de membros nessa categoria.",
        f"{TARGET_COLUMN} médio do cluster (informativo, não usado no treino): "
        f"US$ {centroid[f'{TARGET_COLUMN}_media_informativa'] * 1000:,.0f}",
    ]
    for warning in result["avisos"]:
        lines.append(f"Aviso: {warning}")
    return "\n".join(lines)


def read_houses_from_json(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"JSON inválido em {path}: {exc}") from None
    houses = content if isinstance(content, list) else [content]
    if not houses:
        raise InvalidInputError("O arquivo JSON não contém nenhuma residência.")
    return houses


def prompt_house_interactively() -> dict:
    print("Informe as características da nova residência (use ponto ou vírgula como separador decimal).")
    house = {"id": input("Identificação da residência [nova_residencia]: ").strip() or "nova_residencia"}
    for feature in FEATURES:
        while True:
            raw = input(f"{feature} - {FEATURE_DESCRIPTIONS[feature]}: ")
            try:
                house[feature] = validate_field(feature, raw)
                break
            except InvalidInputError as exc:
                print(f"  Valor inválido -> {exc}")
    return house


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Classifica uma nova residência em um cluster do modelo K-Means treinado."
    )
    parser.add_argument("--input", "-i", type=Path,
                        help="Arquivo JSON com uma residência (objeto) ou várias (lista). "
                             "Sem este argumento, a entrada é interativa.")
    parser.add_argument("--model", "-m", type=Path, default=MODEL_PATH, help="Caminho do modelo .joblib.")
    args = parser.parse_args(argv)

    try:
        artifact = load_artifact(args.model)
        houses = read_houses_from_json(args.input) if args.input else [prompt_house_interactively()]
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nEntrada cancelada.", file=sys.stderr)
        return 1

    exit_code = 0
    for index, house in enumerate(houses, start=1):
        try:
            print(format_result(predict_house(artifact, house)))
        except InvalidInputError as exc:
            label = house.get("id", f"#{index}") if isinstance(house, dict) else f"#{index}"
            print(f"Erro na residência {label}: {exc}", file=sys.stderr)
            exit_code = 1
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
