# Arquitetura do Projeto

## Estrutura de Diretorios

```
boston-housing-kmeans/
├── data/
│   └── HousingData.csv              # Base original (506 registros x 14 colunas)
├── docs/
│   ├── PROJETO.md                   # Descricao geral e mapeamento dos criterios
│   ├── ARQUITETURA.md               # Este arquivo - estrutura e fluxo do codigo
│   ├── CRITERIOS_AVALIACAO.md       # Checklist detalhado dos criterios da avaliacao
│   └── GUIA_EXECUCAO.md             # Passo a passo para executar o projeto
├── examples/
│   └── nova_residencia.json         # Exemplo de entrada para inferencia
├── models/
│   └── kmeans_pipeline.joblib       # Modelo treinado (preprocessador + KMeans + metadados)
├── reports/
│   ├── elbow_method.png             # Grafico do metodo do cotovelo
│   ├── elbow_results.csv            # Inercia, silhouette e reducao % para k=1..10
│   ├── silhouette_scores.png        # Grafico de silhouette (metrica complementar)
│   ├── cluster_distribution.png     # Distribuicao de registros por cluster
│   └── cluster_centroids.csv        # Centroides na escala original
├── src/
│   ├── __init__.py                  # Marca o diretorio como pacote Python
│   ├── config.py                    # Configuracoes: caminhos, features, hiperparametros
│   ├── preprocessing.py             # Carregamento dos dados, pipeline de transformacao,
│   │                                  validacao de entrada, desnormalizacao de centroides
│   ├── train.py                     # Modulo de treinamento completo
│   └── inference.py                 # Modulo de inferencia (JSON e interativo)
├── tests/
│   └── test_pipeline.py             # 32 testes automatizados (pytest)
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Modulos do Codigo-Fonte (`src/`)

### `config.py` - Configuracao Central

Concentra todas as constantes do projeto:
- **Caminhos:** base de dados, diretorio de modelos, relatorios, exemplos
- **Features:** lista das 13 features, separacao entre categoricas (`CHAS`) e numericas
- **Hiperparametros:** `RANDOM_STATE = 42`, `N_INIT = 10`, `K_RANGE = range(1, 11)`
- **Descricoes:** dicionario com a descricao em portugues de cada variavel
- **Limites:** faixas validas para cada variavel (usadas na validacao de entrada)

### `preprocessing.py` - Pre-processamento e Validacao

Responsabilidades:
1. **`load_dataset()`** - Carrega o CSV e verifica se todas as colunas esperadas existem
2. **`build_preprocessor()`** - Constroi o `ColumnTransformer` com dois pipelines:
   - Quantitativas: `SimpleImputer(median)` -> `StandardScaler`
   - Categorica (CHAS): `SimpleImputer(most_frequent)` -> `OneHotEncoder`
3. **`centroids_to_original_scale()`** - Desnormaliza centroides para a escala original
4. **`validate_house()`** / **`validate_field()`** - Valida entrada de novas residencias

### `train.py` - Modulo de Treinamento

Fluxo de execucao:
1. Carrega `data/HousingData.csv` e separa as 13 features de MEDV
2. Ajusta o pre-processador e calcula inercia + silhouette para k = 1 a 10
3. Detecta o cotovelo (`find_elbow_k`) e gera graficos
4. Treina o pipeline definitivo `[preprocessor -> KMeans(k=3)]`
5. Calcula perfis dos clusters (centroides desnormalizados + estatisticas)
6. Salva tudo em `models/kmeans_pipeline.joblib`

Funcoes principais:
- `compute_elbow()` - Calcula inercia e silhouette para cada k
- `find_elbow_k()` - Algoritmo do cotovelo por maxima distancia a reta
- `plot_elbow()` / `plot_silhouette()` / `plot_cluster_distribution()` - Graficos
- `build_cluster_profiles()` - Perfis com centroides desnormalizados e estatisticas
- `train()` - Funcao principal de treinamento
- `main()` - Ponto de entrada via CLI (`python -m src.train`)

### `inference.py` - Modulo de Inferencia

Fluxo de execucao:
1. Carrega o modelo salvo (`kmeans_pipeline.joblib`)
2. Recebe dados via JSON ou terminal interativo
3. Valida todas as 13 features (tipo, faixa, obrigatoriedade)
4. Transforma com o pre-processador salvo (mesmo fit do treinamento)
5. Classifica com `KMeans.predict()`
6. Recupera e desnormaliza o centroide do cluster atribuido
7. Exibe comparacao residencia vs. centroide na escala original

Funcoes principais:
- `load_artifact()` - Carrega e valida o modelo salvo
- `predict_house()` - Pipeline completo de inferencia para uma residencia
- `format_result()` - Formata a saida com comparacao detalhada
- `read_houses_from_json()` - Leitura de entrada via arquivo
- `prompt_house_interactively()` - Entrada interativa pelo terminal
- `main()` - Ponto de entrada via CLI (`python -m src.inference`)

---

## Fluxo de Dados

```
                    TREINAMENTO
                    ===========
HousingData.csv
      |
      v
  load_dataset()  -->  DataFrame (506 x 14)
      |
      v
  build_preprocessor().fit_transform(X)
      |    Quantitativas: imputacao(mediana) + padronizacao (media=0, std=1)
      |    Categorica CHAS: imputacao(moda) + one-hot encoding
      v
  X_transformed (506 x 14 dimensoes)
      |
      v
  compute_elbow() --> inercia + silhouette para k=1..10
  find_elbow_k()  --> k otimo = 3
      |
      v
  Pipeline [preprocessor -> KMeans(k=3)].fit(X)
      |
      v
  joblib.dump() --> models/kmeans_pipeline.joblib
  plot_*()      --> reports/*.png
  to_csv()      --> reports/*.csv


                    INFERENCIA
                    ==========
nova_residencia.json  OU  terminal interativo
      |
      v
  validate_house()  -->  DataFrame (1 x 13)
      |
      v
  preprocessor.transform(X)   (usa medias/desvios do treino)
      |
      v
  KMeans.predict()  -->  cluster atribuido
      |
      v
  centroids_to_original_scale()  -->  centroide desnormalizado
      |
      v
  format_result()  -->  saida formatada (residencia vs. centroide)
```

---

## Tecnologias

| Biblioteca | Versao | Uso |
|---|---|---|
| Python | 3.14 | Linguagem |
| scikit-learn | 1.9.1 | Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, KMeans, silhouette_score |
| pandas | 3.1.1 | Leitura e manipulacao dos dados |
| numpy | 2.5.4 | Calculos numericos |
| matplotlib | 3.11.2 | Graficos |
| joblib | 1.6.0 | Persistencia do modelo |
| pytest | 9.2.0 | Testes automatizados |
