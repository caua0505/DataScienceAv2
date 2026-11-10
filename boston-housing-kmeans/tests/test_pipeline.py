import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXAMPLE_PATH,
    FEATURES,
    K_RANGE,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.inference import load_artifact, main as inference_main, predict_house
from src.preprocessing import (
    InvalidInputError,
    build_preprocessor,
    centroids_to_original_scale,
    load_dataset,
    validate_house,
)
from src.train import compute_elbow, find_elbow_k, train


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artefatos")
    model_path = out / "models" / "kmeans_pipeline.joblib"
    reports_dir = out / "reports"
    artifact = train(model_path=model_path, reports_dir=reports_dir)
    return {"artifact": artifact, "model_path": model_path, "reports_dir": reports_dir}


@pytest.fixture
def valid_house():
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


# ---------- Leitura e pré-processamento ----------

def test_dataset_loads_with_expected_shape(dataset):
    assert dataset.shape == (506, 14)
    assert set(FEATURES + [TARGET_COLUMN]) == set(dataset.columns)
    assert dataset[FEATURES].isna().sum().sum() > 0


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "inexistente.csv")


def test_preprocessor_imputes_and_scales(dataset):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataset[FEATURES])
    names = list(preprocessor.get_feature_names_out())

    assert not np.isnan(transformed).any()
    assert transformed.shape == (506, len(NUMERIC_FEATURES) + 2)
    assert TARGET_COLUMN not in names

    numeric = transformed[:, [names.index(f) for f in NUMERIC_FEATURES]]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-9)


def test_chas_is_one_hot_encoded_not_scaled(dataset):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataset[FEATURES])
    names = list(preprocessor.get_feature_names_out())
    chas_cols = [i for i, n in enumerate(names) if n.startswith("CHAS_")]

    assert len(chas_cols) == 2
    block = transformed[:, chas_cols]
    assert set(np.unique(block)) <= {0.0, 1.0}
    np.testing.assert_array_equal(block.sum(axis=1), 1.0)


def test_missing_chas_imputed_with_most_frequent(dataset):
    preprocessor = build_preprocessor().fit(dataset[FEATURES])
    imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
    assert imputer.statistics_[0] == 0.0


# ---------- Método do cotovelo ----------

def test_elbow_inertia_decreases_over_k_range(dataset):
    X = build_preprocessor().fit_transform(dataset[FEATURES])
    elbow = compute_elbow(X)
    assert list(elbow["k"]) == list(K_RANGE)
    assert (elbow["inercia_wcss"].diff().dropna() < 0).all()
    assert np.isnan(elbow.loc[0, "silhouette"])
    assert elbow["silhouette"].dropna().between(-1, 1).all()


def test_find_elbow_k_on_known_curve():
    k = list(range(1, 11))
    inertia = [100, 40, 20, 17, 15, 14, 13, 12, 11, 10]
    assert find_elbow_k(k, inertia) == 3


# ---------- Treinamento e persistência ----------

def test_training_produces_model_and_reports(trained):
    artifact, reports = trained["artifact"], trained["reports_dir"]
    assert trained["model_path"].exists()
    for name in ("elbow_method.png", "silhouette_scores.png", "cluster_distribution.png",
                 "cluster_centroids.csv", "elbow_results.csv"):
        assert (reports / name).exists(), name

    kmeans = artifact["pipeline"].named_steps["kmeans"]
    assert kmeans.n_clusters == artifact["n_clusters"] == artifact["elbow_k"]
    assert artifact["cluster_profiles"]["n_registros"].sum() == 506


def test_training_is_reproducible(trained, tmp_path):
    again = train(model_path=tmp_path / "m.joblib", reports_dir=tmp_path / "r")
    first = trained["artifact"]["pipeline"].named_steps["kmeans"]
    second = again["pipeline"].named_steps["kmeans"]
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)


def test_saved_model_loads_and_matches(trained, dataset):
    loaded = load_artifact(trained["model_path"])
    original_labels = trained["artifact"]["pipeline"].named_steps["kmeans"].labels_
    np.testing.assert_array_equal(loaded["pipeline"].predict(dataset[FEATURES]), original_labels)


def test_load_artifact_missing_file_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_artifact(tmp_path / "nao_existe.joblib")


# ---------- Desnormalização ----------

def test_centroids_inverse_transform_equals_member_means(trained, dataset):
    pipeline = trained["artifact"]["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    kmeans = pipeline.named_steps["kmeans"]

    centroids = centroids_to_original_scale(preprocessor, kmeans.cluster_centers_)
    imputed = pd.DataFrame(
        preprocessor.named_transformers_["num"].named_steps["imputer"].transform(dataset[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
    )
    member_means = imputed.groupby(kmeans.labels_).mean()
    np.testing.assert_allclose(centroids[NUMERIC_FEATURES].to_numpy(), member_means.to_numpy(), rtol=1e-6, atol=1e-6)


def test_categorical_centroid_reports_proportions(trained, dataset):
    profiles = trained["artifact"]["cluster_profiles"]
    labels = trained["artifact"]["pipeline"].named_steps["kmeans"].labels_
    np.testing.assert_allclose(profiles["CHAS_prop_0"] + profiles["CHAS_prop_1"], 1.0)
    assert profiles["CHAS_predominante"].isin([0, 1]).all()

    chas_imputed = dataset["CHAS"].fillna(0.0)
    expected = chas_imputed.groupby(labels).mean().to_numpy()
    np.testing.assert_allclose(profiles["CHAS_prop_1"].to_numpy(), expected, atol=1e-6)


# ---------- Inferência ----------

def test_predict_new_house(trained, valid_house):
    result = predict_house(trained["artifact"], valid_house)
    n_clusters = trained["artifact"]["n_clusters"]

    assert 0 <= result["cluster"] < n_clusters
    assert result["distancia_centroide"] >= 0
    assert result["id"] == valid_house["id"]
    assert result["avisos"] == []
    for feature in NUMERIC_FEATURES:
        assert feature in result["centroide"]


def test_training_row_is_assigned_to_its_own_cluster(trained, dataset):
    row = dataset.loc[0, FEATURES].to_dict()
    expected = trained["artifact"]["pipeline"].named_steps["kmeans"].labels_[0]
    assert predict_house(trained["artifact"], row)["cluster"] == expected


def test_assigned_cluster_is_the_nearest(trained, valid_house):
    result = predict_house(trained["artifact"], valid_house)
    second = result["segundo_cluster_mais_proximo"]
    assert second is None or result["distancia_centroide"] <= second["distancia"]


def test_medv_is_not_required_and_is_ignored(trained, valid_house):
    without = predict_house(trained["artifact"], valid_house)
    with_medv = predict_house(trained["artifact"], {**valid_house, "MEDV": 50.0})
    assert without["cluster"] == with_medv["cluster"]
    assert any("MEDV" in w for w in with_medv["avisos"])


def test_out_of_range_value_generates_warning(trained, valid_house):
    result = predict_house(trained["artifact"], {**valid_house, "PTRATIO": 40})
    assert any("PTRATIO" in w for w in result["avisos"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"CRIM": "abc"}, "CRIM: valor não numérico"),
        ({"CHAS": 2}, "CHAS: deve ser 0 ou 1"),
        ({"ZN": -5}, "ZN: fora do intervalo"),
        ({"AGE": 120}, "AGE: fora do intervalo"),
        ({"RAD": 4.5}, "RAD: deve ser um número inteiro"),
        ({"NOX": None}, "NOX: valor obrigatório"),
        ({"RM": float("nan")}, "RM: valor deve ser finito"),
        ({"EXTRA": 1}, "campo desconhecido: EXTRA"),
    ],
)
def test_invalid_inputs_raise_clear_errors(valid_house, change, message):
    with pytest.raises(InvalidInputError, match=message):
        validate_house({**valid_house, **change})


def test_missing_field_is_reported(valid_house):
    house = {k: v for k, v in valid_house.items() if k != "LSTAT"}
    with pytest.raises(InvalidInputError, match="LSTAT: campo obrigatório ausente"):
        validate_house(house)


def test_string_numbers_with_comma_are_accepted(valid_house):
    X, _ = validate_house({**valid_house, "RM": "6,4"})
    assert X.loc[0, "RM"] == pytest.approx(6.4)
    assert list(X.columns) == FEATURES


# ---------- Interface de linha de comando ----------

def test_cli_with_json_file(trained, capsys):
    exit_code = inference_main(["--input", str(EXAMPLE_PATH), "--model", str(trained["model_path"])])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Cluster atribuído" in output
    assert "Centroide do cluster na escala original" in output


def test_cli_with_list_of_houses_and_one_invalid(trained, valid_house, tmp_path, capsys):
    path = tmp_path / "lote.json"
    path.write_text(json.dumps([valid_house, {**valid_house, "id": "ruim", "CHAS": 7}]), encoding="utf-8")
    exit_code = inference_main(["-i", str(path), "-m", str(trained["model_path"])])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.count("Cluster atribuído") == 1
    assert "Erro na residência ruim" in captured.err


def test_cli_with_malformed_json(trained, tmp_path, capsys):
    path = tmp_path / "quebrado.json"
    path.write_text("{ isto não é json", encoding="utf-8")
    assert inference_main(["-i", str(path), "-m", str(trained["model_path"])]) == 1
    assert "JSON inválido" in capsys.readouterr().err


def test_cli_interactive_mode(trained, monkeypatch, capsys):
    answers = iter(["teste_interativo", "0.05", "80", "2,5", "1", "0.41", "xyz", "7.1",
                    "20", "7.5", "3", "250", "15", "396", "4"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    exit_code = inference_main(["-m", str(trained["model_path"])])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Valor inválido -> RM" in output
    assert "Residência: teste_interativo" in output
