# Projeto: Clusterizacao com K-Means - Boston Housing

## Contexto

Este projeto e a **avaliacao do 2o bimestre da disciplina de Data Science**. O objetivo e treinar um modelo de clusterizacao **K-Means** utilizando a base **Boston Housing** e construir dois modulos independentes: um de **treinamento** e outro de **inferencia**.

---

## Enunciado da Avaliacao

> Treinar um clusterizador com K-Means usando a base Boston Housing.

### Criterios exigidos

#### Modulo de Treinamento
1. Normalizar a base com cuidado sobre dados categoricos e dados quantitativos
2. Nao ha necessidade de balancear a base
3. Determinar o numero otimo de grupos com o metodo do cotovelo (distancias)
4. Treinar e salvar o modelo

#### Modulo de Inferencia
1. Receber dados de uma nova residencia
2. Normalizar os dados da nova residencia
3. Inferir a qual grupo essa nova residencia pertence
4. Exibir os dados desnormalizados do centroide inferido

---

## Como cada criterio foi atendido

### Modulo de Treinamento (`src/train.py`)

| Criterio | Onde esta implementado | Detalhes |
|---|---|---|
| **Normalizacao com cuidado sobre categoricos e quantitativos** | `src/preprocessing.py` - funcao `build_preprocessor()` | Variaveis quantitativas (12): `SimpleImputer(median)` + `StandardScaler`. Variavel categorica (CHAS): `SimpleImputer(most_frequent)` + `OneHotEncoder`. Tudo dentro de um `ColumnTransformer` do scikit-learn. |
| **Nao balancear a base** | Nao ha nenhum passo de balanceamento | Em clusterizacao nao supervisionada nao existe classe-alvo a ser balanceada. |
| **Metodo do cotovelo (distancias)** | `src/train.py` - funcoes `compute_elbow()` e `find_elbow_k()` | K-Means executado para k = 1 a 10, registrando a inercia (WCSS) de cada. O ponto de cotovelo e identificado como o k cuja distancia a reta entre os extremos e maxima. Resultado: **k = 3**. |
| **Treinar e salvar o modelo** | `src/train.py` - funcao `train()` | Pipeline `[preprocessor -> KMeans]` treinado e salvo em `models/kmeans_pipeline.joblib` via joblib, incluindo metadados (features, k, perfis, versao, data). |

### Modulo de Inferencia (`src/inference.py`)

| Criterio | Onde esta implementado | Detalhes |
|---|---|---|
| **Receber dados de uma nova residencia** | `src/inference.py` - funcoes `read_houses_from_json()` e `prompt_house_interactively()` | Aceita entrada via arquivo JSON (uma ou varias residencias) ou via terminal interativo. |
| **Normalizar os dados da nova residencia** | `src/inference.py` - funcao `predict_house()` | Usa `preprocessor.transform(X)` com o mesmo pre-processador salvo no treinamento (medias, desvios e encoder ja ajustados). |
| **Inferir a qual grupo pertence** | `src/inference.py` - funcao `predict_house()` | Usa `KMeans.predict()` para atribuir a residencia ao centroide mais proximo. |
| **Exibir dados desnormalizados do centroide** | `src/preprocessing.py` - funcao `centroids_to_original_scale()` e `src/inference.py` - funcao `format_result()` | Quantitativas: `StandardScaler.inverse_transform`. Categorica CHAS: mostra a categoria predominante e a proporcao de membros. A saida mostra coluna a coluna os valores da residencia vs. centroide na escala original. |

---

## Base de Dados

- **Nome:** Boston Housing (`HousingData.csv`)
- **Registros:** 506
- **Colunas:** 14 (13 features + MEDV como alvo informativo)
- **MEDV** (valor mediano das residencias) **nao participa da clusterizacao** - e calculado apenas como informacao descritiva apos o treino

### Variaveis

| Variavel | Tipo | Descricao |
|---|---|---|
| CRIM | Quantitativa | Taxa de criminalidade per capita |
| ZN | Quantitativa | % de terrenos residenciais com lotes > 25.000 pes^2 |
| INDUS | Quantitativa | % de area de comercio nao varejista |
| **CHAS** | **Categorica binaria** | Margeia o rio Charles (0/1) |
| NOX | Quantitativa | Concentracao de oxidos nitricos |
| RM | Quantitativa | Numero medio de comodos |
| AGE | Quantitativa | % de imoveis construidos antes de 1940 |
| DIS | Quantitativa | Distancia ponderada a centros de emprego |
| RAD | Numerica discreta | Indice de acesso a rodovias radiais |
| TAX | Quantitativa | Imposto predial por US$ 10.000 |
| PTRATIO | Quantitativa | Razao aluno/professor |
| B | Quantitativa | 1000(Bk - 0.63)^2, Bk = proporcao de populacao negra |
| LSTAT | Quantitativa | % da populacao de baixa renda |
| **MEDV** | **Alvo (fora da clusterizacao)** | Valor mediano das residencias (US$ 1.000) |

---

## Resultados

O metodo do cotovelo detectou **k = 3** como numero otimo de clusters:

| Cluster | Perfil | Residencias | MEDV medio (informativo) |
|---|---|---:|---:|
| 0 | Suburbano residencial | 98 (19,4%) | US$ 28.650 |
| 1 | Intermediario | 245 (48,4%) | US$ 24.280 |
| 2 | Urbano/industrial | 163 (32,2%) | US$ 16.230 |

Os graficos e tabelas detalhadas estao em `reports/`.

---

## Entrega

- Repositorio dedicado exclusivamente a esta avaliacao
- Link do repositorio entregue conforme solicitado pelo professor
