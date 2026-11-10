from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "HousingData.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "kmeans_pipeline.joblib"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "nova_residencia.json"

RANDOM_STATE = 42
N_INIT = 10
K_RANGE = range(1, 11)

TARGET_COLUMN = "MEDV"

FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
CATEGORICAL_FEATURES = ["CHAS"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

FEATURE_DESCRIPTIONS = {
    "CRIM": "Taxa de criminalidade per capita da região",
    "ZN": "% de terrenos residenciais com lotes acima de 25.000 pés²",
    "INDUS": "% de área de comércio não varejista",
    "CHAS": "Margeia o rio Charles (1 = sim, 0 = não)",
    "NOX": "Concentração de óxidos nítricos (partes por 10 milhões)",
    "RM": "Número médio de cômodos por residência",
    "AGE": "% de imóveis ocupados construídos antes de 1940",
    "DIS": "Distância ponderada a 5 centros de emprego de Boston",
    "RAD": "Índice de acessibilidade a rodovias radiais (inteiro)",
    "TAX": "Imposto predial por US$ 10.000",
    "PTRATIO": "Razão aluno/professor",
    "B": "1000(Bk - 0.63)², Bk = proporção de população negra",
    "LSTAT": "% da população de baixa renda",
}

# Limites físicos/definicionais das variáveis, usados apenas para validar entradas.
FEATURE_BOUNDS = {
    "CRIM": (0.0, None),
    "ZN": (0.0, 100.0),
    "INDUS": (0.0, 100.0),
    "CHAS": (0.0, 1.0),
    "NOX": (0.0, None),
    "RM": (0.0, None),
    "AGE": (0.0, 100.0),
    "DIS": (0.0, None),
    "RAD": (1.0, None),
    "TAX": (0.0, None),
    "PTRATIO": (0.0, None),
    "B": (0.0, 396.9),
    "LSTAT": (0.0, 100.0),
}
INTEGER_FEATURES = {"CHAS", "RAD"}

METADATA_KEYS = {"id", "observacao"}
