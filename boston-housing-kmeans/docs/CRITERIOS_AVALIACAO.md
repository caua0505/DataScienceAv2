# Checklist dos Criterios da Avaliacao

Este documento mapeia cada criterio exigido pelo professor ao arquivo e trecho de codigo correspondente.

---

## Modulo de Treinamento

### 1. Normalizar a base com cuidado sobre dados categoricos e dados quantitativos

**Status:** Atendido

**Arquivo:** `src/preprocessing.py`, funcao `build_preprocessor()` (linhas 38-54)

**O que foi feito:**
- Utiliza `ColumnTransformer` do scikit-learn para tratar cada tipo de variavel de forma diferente
- **12 variaveis quantitativas** (CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT):
  - Imputacao de valores ausentes pela **mediana** (`SimpleImputer(strategy="median")`)
  - Padronizacao com **StandardScaler** (media=0, desvio padrao=1)
- **1 variavel categorica** (CHAS - binaria 0/1):
  - Imputacao de valores ausentes pela **moda** (`SimpleImputer(strategy="most_frequent")`)
  - Codificacao com **OneHotEncoder** (gera CHAS_0 e CHAS_1, sem padronizacao)
- A separacao entre categoricas e quantitativas esta definida em `src/config.py` (linhas 20-21)

**Por que mediana e nao media para as quantitativas:**
- Variaveis como CRIM sao muito assimetricas (media 3,61 vs. mediana 0,25)
- Imputar pela media deslocaria registros para valores atipicos

**Por que moda para CHAS:**
- CHAS e binaria (0 ou 1), e a categoria mais frequente e 0

**Por que CHAS nao e padronizada:**
- Ela e um indicador (sim/nao), nao uma grandeza continua
- Padronizar uma variavel binaria com StandardScaler distorceria sua natureza

---

### 2. Nao ha necessidade de balancear a base

**Status:** Atendido

**O que foi feito:**
- Nenhum passo de balanceamento foi implementado (corretamente)
- Em clusterizacao nao supervisionada, nao existe classe-alvo a ser balanceada
- O K-Means agrupa dados pela proximidade no espaco de features, nao por uma variavel de resposta

---

### 3. Determinar o numero otimo de grupos com o metodo do cotovelo (distancias)

**Status:** Atendido

**Arquivo:** `src/train.py`, funcoes `compute_elbow()` (linhas 37-43) e `find_elbow_k()` (linhas 46-53)

**O que foi feito:**
- K-Means executado para **k = 1 a 10** com `n_init=10` e `random_state=42`
- Para cada k, registrada a **inercia (WCSS)** - soma das distancias quadraticas ao centroide
- Para k >= 2, calculado tambem o **coeficiente de Silhouette** como metrica complementar
- O ponto de cotovelo e detectado automaticamente: normaliza a curva em [0,1] nos dois eixos e encontra o k com maior distancia a reta entre os extremos
- **Resultado: k = 3**

**Artefatos gerados:**
- `reports/elbow_method.png` - Grafico do cotovelo com o ponto marcado
- `reports/silhouette_scores.png` - Grafico de silhouette
- `reports/elbow_results.csv` - Tabela com inercia, silhouette e reducao percentual

---

### 4. Treinar e salvar o modelo

**Status:** Atendido

**Arquivo:** `src/train.py`, funcao `train()` (linhas 126-175)

**O que foi feito:**
- Pipeline scikit-learn com dois passos: `[preprocessor -> KMeans(n_clusters=3)]`
- O pipeline e treinado com `.fit(X)` sobre as 13 features (MEDV excluido)
- Modelo salvo em `models/kmeans_pipeline.joblib` via `joblib.dump()`
- O arquivo salvo contem: pipeline completo, lista de features, k escolhido, perfis dos clusters, resultados do cotovelo, faixas observadas, versao do scikit-learn e data do treino
- O treinamento e **deterministico** (random_state=42) - executar novamente produz os mesmos resultados

---

## Modulo de Inferencia

### 1. Receber dados de uma nova residencia

**Status:** Atendido

**Arquivo:** `src/inference.py`, funcoes `read_houses_from_json()` (linhas 118-129) e `prompt_house_interactively()` (linhas 132-143)

**O que foi feito:**
- **Via arquivo JSON:** aceita um objeto (uma residencia) ou uma lista de objetos
  - Comando: `python -m src.inference --input examples/nova_residencia.json`
- **Via terminal interativo:** solicita cada uma das 13 caracteristicas com descricao
  - Comando: `python -m src.inference`
  - Aceita tanto ponto quanto virgula como separador decimal
  - Valores invalidos sao pedidos novamente
- Todas as 13 features sao obrigatorias; campos `id` e `observacao` sao opcionais

---

### 2. Normalizar os dados da nova residencia

**Status:** Atendido

**Arquivo:** `src/inference.py`, funcao `predict_house()` (linha 41)

**O que foi feito:**
- Usa `preprocessor.transform(X)` - o **mesmo pre-processador** salvo no treinamento
- As medias, desvios padrao e categorias do encoder sao os aprendidos na base de treino
- O pre-processador **nao e reajustado** (nao faz `.fit()` novamente)
- Isso garante que a nova residencia e normalizada na **mesma escala** dos dados de treino

---

### 3. Inferir a qual grupo essa nova residencia pertence

**Status:** Atendido

**Arquivo:** `src/inference.py`, funcao `predict_house()` (linha 42)

**O que foi feito:**
- Usa `KMeans.predict(X_transformed)` para atribuir a residencia ao centroide mais proximo
- Calcula a distancia euclidiana ao centroide atribuido e a todos os outros
- Informa tambem o segundo cluster mais proximo e a distancia media dos membros do cluster

---

### 4. Exibir os dados desnormalizados do centroide inferido

**Status:** Atendido

**Arquivos:**
- `src/preprocessing.py`, funcao `centroids_to_original_scale()` (linhas 57-83)
- `src/inference.py`, funcao `format_result()` (linhas 72-115)

**O que foi feito:**
- **Quantitativas:** `StandardScaler.inverse_transform()` devolve os valores na escala original
- **Categorica CHAS:** cada coordenada one-hot do centroide e a proporcao de membros naquela categoria; exibe a categoria predominante e o percentual
- A saida mostra uma tabela comparativa: para cada uma das 13 variaveis, mostra o **valor da residencia** ao lado do **valor do centroide** na escala original, com a descricao da variavel
- Inclui tambem: MEDV medio informativo, distancia ao centroide, distancia media dos membros e segundo cluster mais proximo

---

## Resumo

| # | Criterio | Status | Arquivo Principal |
|---|---|---|---|
| T1 | Normalizacao categoricos/quantitativos | Atendido | `src/preprocessing.py` |
| T2 | Sem balanceamento | Atendido | (nenhum passo de balanceamento) |
| T3 | Metodo do cotovelo | Atendido | `src/train.py` |
| T4 | Treinar e salvar modelo | Atendido | `src/train.py` |
| I1 | Receber dados nova residencia | Atendido | `src/inference.py` |
| I2 | Normalizar nova residencia | Atendido | `src/inference.py` |
| I3 | Inferir grupo | Atendido | `src/inference.py` |
| I4 | Exibir centroide desnormalizado | Atendido | `src/preprocessing.py` + `src/inference.py` |
