# Guia de Execucao

Passo a passo para instalar, treinar o modelo e executar a inferencia.

---

## Pre-requisitos

- Python 3.10 ou superior (desenvolvido em Python 3.14)
- Git

---

## 1. Clonar o repositorio

```bash
git clone <url-do-repositorio>
cd boston-housing-kmeans
```

---

## 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
```

**Ativar:**

- Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

- Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```

- Linux/macOS:
```bash
source .venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias instaladas:
- scikit-learn 1.9.1
- pandas 3.1.1
- numpy 2.5.4
- matplotlib 3.11.2
- joblib 1.6.0
- pytest 9.2.0

---

## 4. Executar o treinamento

```bash
python -m src.train
```

**O que acontece:**
1. Carrega `data/HousingData.csv` (506 registros)
2. Pre-processa: imputacao + normalizacao (quantitativas) e one-hot encoding (CHAS)
3. Executa o metodo do cotovelo para k = 1 a 10
4. Detecta k otimo = 3
5. Treina o modelo K-Means com k = 3
6. Salva em `models/kmeans_pipeline.joblib`
7. Gera graficos e tabelas em `reports/`

**Saida no terminal:** resumo com a tabela do cotovelo, k escolhido e perfis dos clusters.

**Para forcar outro k (comparacao):**
```bash
python -m src.train --k 4
```

---

## 5. Executar a inferencia

### Opcao A: Via arquivo JSON

```bash
python -m src.inference --input examples/nova_residencia.json
```

O arquivo `examples/nova_residencia.json` contem um exemplo pronto:
```json
{
  "id": "exemplo_demonstracao_001",
  "CRIM": 0.25, "ZN": 12.5, "INDUS": 7.0, "CHAS": 0, "NOX": 0.52, "RM": 6.4,
  "AGE": 65.0, "DIS": 4.2, "RAD": 5, "TAX": 300, "PTRATIO": 18.0, "B": 390.0, "LSTAT": 10.5
}
```

O arquivo JSON pode conter:
- Um objeto (uma residencia)
- Uma lista de objetos (varias residencias)

Campos opcionais: `id` e `observacao`.

### Opcao B: Via terminal interativo

```bash
python -m src.inference
```

O programa solicita cada uma das 13 caracteristicas com sua descricao. Aceita ponto e virgula como separador decimal. Valores invalidos sao pedidos novamente.

### Opcao C: Com outro modelo

```bash
python -m src.inference --model caminho/para/outro_modelo.joblib
```

---

## 6. Executar os testes

```bash
pytest
```

**Resultado esperado:** 32 testes aprovados.

Os testes cobrem:
- Leitura da base e erro quando o arquivo nao existe
- Imputacao (nenhum NaN apos pre-processamento; CHAS imputada com 0)
- Padronizacao (media 0, desvio 1) e OneHot de CHAS
- Metodo do cotovelo e deteccao automatica
- Geracao do modelo e reprodutibilidade do treino
- Carregamento e verificacao do modelo salvo
- Desnormalizacao dos centroides
- Inferencia (residencia da base cai no proprio cluster, MEDV ignorado, avisos)
- Validacao de entradas invalidas
- Interface de linha de comando (JSON, lote, JSON malformado, modo interativo)

---

## 7. Saida da inferencia (exemplo)

```
==============================================================================
Residencia: exemplo_demonstracao_001
==============================================================================
Cluster atribuido: 1  (245 residencias da base, 48.4%)
Distancia euclidiana ao centroide (espaco normalizado): 0.594
Distancia media dos membros do cluster ao centroide:    2.090
Segundo cluster mais proximo: 0 (distancia 2.619)

Centroide do cluster na escala original (perfil medio do grupo):
Variavel    Residencia    Centroide  Descricao
------------------------------------------------------------------------------
CRIM             0.250        0.382  Taxa de criminalidade per capita da regiao
ZN              12.500        3.000  % de terrenos residenciais com lotes acima de 25.000 pes^2
INDUS            7.000        8.918  % de area de comercio nao varejista
CHAS                 0      0 (93%)  Margeia o rio Charles (1 = sim, 0 = nao)
NOX              0.520        0.519  Concentracao de oxidos nitricos
RM               6.400        6.341  Numero medio de comodos por residencia
AGE             65.000       69.276  % de imoveis ocupados construidos antes de 1940
DIS              4.200        3.811  Distancia ponderada a 5 centros de emprego de Boston
RAD              5.000        4.543  Indice de acessibilidade a rodovias radiais
TAX            300.000      305.090  Imposto predial por US$ 10.000
PTRATIO         18.000       18.162  Razao aluno/professor
B              390.000      382.849  1000(Bk - 0.63)^2
LSTAT           10.500       11.136  % da populacao de baixa renda
------------------------------------------------------------------------------
MEDV medio do cluster (informativo, nao usado no treino): US$ 24,281
```

### Como interpretar

- **Cluster atribuido:** o perfil de regiao mais parecido com a residencia informada
- **Distancia ao centroide:** valores menores que a distancia media dos membros indicam residencia tipica do grupo
- **Segundo cluster mais proximo:** se as distancias forem parecidas, a residencia esta na fronteira entre dois perfis
- **Tabela comparativa:** mostra em que a residencia se parece ou difere do perfil medio do grupo
