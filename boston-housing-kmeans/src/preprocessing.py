import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_BOUNDS,
    FEATURES,
    INTEGER_FEATURES,
    METADATA_KEYS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


class InvalidInputError(ValueError):
    """Dados de entrada de uma residência que não podem ser processados."""


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Base de dados não encontrada em: {path}")
    df = pd.read_csv(path)
    missing_columns = [c for c in FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes na base: {', '.join(missing_columns)}")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="error", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def centroids_to_original_scale(preprocessor: ColumnTransformer, centroids: np.ndarray) -> pd.DataFrame:
    """Converte centroides do espaço transformado para a escala original.

    Variáveis numéricas passam pelo inverse_transform do StandardScaler.
    Para as categóricas, cada coordenada one-hot do centroide é a proporção
    de membros do cluster naquela categoria, então reportamos proporções e a
    categoria predominante em vez de uma "média" sem significado.
    """
    names = list(preprocessor.get_feature_names_out())
    centroids = np.atleast_2d(centroids)

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    numeric_idx = [names.index(f) for f in NUMERIC_FEATURES]
    numeric_values = scaler.inverse_transform(centroids[:, numeric_idx])
    # Arredonda ruído de ponto flutuante (ex.: -1e-15) e elimina o sinal de -0.0.
    numeric_values = np.round(numeric_values, 10) + 0.0
    result = pd.DataFrame(numeric_values, columns=NUMERIC_FEATURES)

    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        block_idx = [names.index(f"{feature}_{c}") for c in categories]
        block = np.clip(centroids[:, block_idx], 0.0, 1.0)
        for col, category in enumerate(categories):
            result[f"{feature}_prop_{int(category)}"] = block[:, col]
        result[f"{feature}_predominante"] = categories[block.argmax(axis=1)].astype(int)

    return result


def validate_field(name: str, value) -> float:
    if isinstance(value, bool) or value is None:
        raise InvalidInputError(f"{name}: valor obrigatório e numérico (recebido: {value!r})")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidInputError(f"{name}: valor não numérico (recebido: {value!r})") from None
    if not math.isfinite(number):
        raise InvalidInputError(f"{name}: valor deve ser finito (recebido: {value!r})")

    if name in INTEGER_FEATURES and not number.is_integer():
        raise InvalidInputError(f"{name}: deve ser um número inteiro (recebido: {value!r})")
    if name == "CHAS" and number not in (0.0, 1.0):
        raise InvalidInputError(f"CHAS: deve ser 0 ou 1 (recebido: {value!r})")

    lower, upper = FEATURE_BOUNDS[name]
    if number < lower or (upper is not None and number > upper):
        limit = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
        raise InvalidInputError(f"{name}: fora do intervalo válido {limit} (recebido: {number})")
    return number


def validate_house(raw: dict) -> tuple[pd.DataFrame, list[str]]:
    """Valida os dados de uma residência e devolve (DataFrame de 1 linha, avisos)."""
    if not isinstance(raw, dict):
        raise InvalidInputError("A residência deve ser um objeto JSON com as características.")

    warnings = []
    if TARGET_COLUMN in raw:
        warnings.append(f"{TARGET_COLUMN} foi informado e será ignorado (não é usado na clusterização).")

    unknown = sorted(k for k in raw if k not in FEATURES and k not in METADATA_KEYS and k != TARGET_COLUMN)
    errors = [f"campo desconhecido: {k}" for k in unknown]
    errors += [f"{f}: campo obrigatório ausente" for f in FEATURES if f not in raw]

    values = {}
    for feature in FEATURES:
        if feature not in raw:
            continue
        try:
            values[feature] = validate_field(feature, raw[feature])
        except InvalidInputError as exc:
            errors.append(str(exc))

    if errors:
        raise InvalidInputError("Dados inválidos:\n  - " + "\n  - ".join(errors))

    return pd.DataFrame([values], columns=FEATURES, dtype=float), warnings
