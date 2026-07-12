# Otimizaçào de campanhas com modelo hierárquico bayesiano

<p align="justify">

Projeto de Ciência de Dados que demonstra como utilizar **Estatística Bayesiana Hierárquica** para otimizar decisões de orçamento em campanhas de marketing digital. O objetivo é reduzir decisões incorretas causadas por amostras pequenas, estimando a taxa real de conversão de cada campanha e quantificando sua incerteza.

Em vez de utilizar apenas a taxa observada (`cliques / impressões`), o projeto modela todas as campanhas simultaneamente através de um modelo **Beta-Binomial Hierárquico**, permitindo que campanhas com poucos dados sejam "encolhidas" (*shrinkage*) em direção à média global da conta.

</p>

---

# 1. Problema de Negócio

<p align="justify">

Em plataformas de anúncios é comum administrar dezenas ou centenas de campanhas simultaneamente.

Nos primeiros dias de execução existe pouca informação disponível. Como consequência, a taxa de conversão observada pode variar drasticamente apenas por efeito do acaso.

Isso pode levar a duas decisões ruins:

* escalar campanhas aparentemente excelentes que tiveram apenas sorte;
* pausar campanhas promissoras que tiveram poucos eventos.

O objetivo deste projeto é utilizar um modelo probabilístico para estimar a verdadeira taxa de conversão de cada campanha considerando toda a informação disponível na conta.

</p>

---

# 2. Objetivos

<p align="justify">

O projeto busca:

* modelar campanhas utilizando uma abordagem Bayesiana Hierárquica;
* reduzir a variabilidade causada por pequenas amostras;
* estimar intervalos de credibilidade para cada campanha;
* calcular probabilidades diretamente interpretáveis para decisões de negócio;
* transformar estimativas estatísticas em ações operacionais.

</p>

---

# 3. Arquitetura

```text
Dados das campanhas
        │
        ▼
Pré-processamento
        │
        ▼
Modelo Beta-Binomial Hierárquico
(PyMC + MCMC)
        │
        ▼
Distribuições Posteriores
        │
        ▼
Probabilidade de superar a média
        │
        ▼
ESCALAR
MANTER
PAUSAR
```

---

# 4. Base de Dados

<p align="justify">

Cada campanha possui:

* quantidade de impressões;
* quantidade de cliques;
* taxa observada.

O modelo considera que todas as campanhas pertencem à mesma conta publicitária, compartilhando uma distribuição global de conversão.

</p>

---

# 5. Modelo Estatístico

<p align="justify">

O projeto utiliza um modelo **Beta-Binomial Hierárquico**.

Os hiperparâmetros aprendem automaticamente qual é a taxa típica da conta.

Cada campanha possui sua própria distribuição de probabilidade, sendo estimada a partir da informação individual e também da informação compartilhada entre todas as campanhas.

Essa estrutura produz o conhecido efeito de **shrinkage**, reduzindo estimativas exageradas provenientes de campanhas com poucas observações.

</p>

---

# 6. Código Principal

```python
with pm.Model() as modelo_ads:

    alpha = pm.Exponential("alpha", 1.0)
    beta = pm.Exponential("beta", 1.0)

    taxa = pm.Beta(
        "taxa",
        alpha=alpha,
        beta=beta,
        shape=len(df)
    )

    pm.Binomial(
        "obs",
        n=df["impressoes"],
        p=taxa,
        observed=df["cliques"]
    )

    trace = pm.sample(
        2000,
        tune=1000,
        target_accept=0.9,
        random_seed=42
    )
```

---

# 7. Comparação com Estatística Clássica (statsmodels)

<p align="justify">

Como referência, uma abordagem clássica pode ser implementada utilizando o **statsmodels** para estimar a taxa de conversão individual e seus respectivos intervalos de confiança.

Entretanto, cada campanha é tratada de forma independente, sem compartilhar informação entre campanhas.

</p>

### Código principal (statsmodels)

```python
from statsmodels.stats.proportion import proportion_confint

df["taxa_bruta"] = (
    df["cliques"] /
    df["impressoes"]
)

df[["ic_low", "ic_high"]] = df.apply(
    lambda row: proportion_confint(
        count=row["cliques"],
        nobs=row["impressoes"],
        alpha=0.05,
        method="wilson"
    ),
    axis=1,
    result_type="expand"
)
```

<p align="justify">

Enquanto o método clássico fornece estimativas independentes para cada campanha, o modelo Bayesiano Hierárquico reduz naturalmente a instabilidade das campanhas com poucas observações ao compartilhar informação entre todos os grupos.

</p>

---

# 8. Resultados

<p align="justify">

Após a inferência MCMC são obtidas:

* média posterior da taxa de conversão;
* intervalo de credibilidade;
* distribuição completa da taxa;
* probabilidade de superar a média global.

Essas informações permitem substituir decisões baseadas apenas na taxa observada por decisões probabilísticas.

</p>

---

# 9. Regra de Negócio

<p align="justify">

Cada campanha recebe uma probabilidade de possuir desempenho superior à média global da conta.

Com base nessa probabilidade, o sistema classifica automaticamente cada campanha em uma das seguintes ações:

* ESCALAR;
* MANTER E COLETAR MAIS DADOS;
* PAUSAR.

Essa estratégia reduz decisões precipitadas e melhora a alocação do orçamento.

</p>

---

# 10. Comparação entre Abordagens

| Característica                     | Estatística Clássica (statsmodels) | Bayesiano Hierárquico      |
| ---------------------------------- | ---------------------------------- | -------------------------- |
| Campanhas independentes            | Sim                                | Não                        |
| Compartilha informação             | Não                                | Sim                        |
| Funciona bem com poucos dados      | Limitado                           | Sim                        |
| Intervalo de confiança             | Sim                                | Intervalo de credibilidade |
| Probabilidade direta de ser melhor | Não                                | Sim                        |
| Efeito de shrinkage                | Não                                | Sim                        |

---

# 11. Tecnologias

* Python
* PyMC
* ArviZ
* NumPy
* Pandas
* Matplotlib
* statsmodels

---

# 12. Aplicações

<p align="justify">

A metodologia pode ser aplicada em diversos problemas de negócio, incluindo:

* otimização de campanhas de marketing;
* testes de produtos digitais;
* análise de conversão por país;
* comparação entre vendedores;
* avaliação de desempenho de lojas;
* monitoramento de modelos de Machine Learning.

</p>

---

# 13. Conclusão

<p align="justify">

Este projeto demonstra como modelos Bayesianos Hierárquicos podem produzir estimativas mais robustas do que abordagens clássicas quando existem múltiplos grupos com diferentes quantidades de dados. Ao incorporar a incerteza diretamente na tomada de decisão, o modelo reduz o risco de escalar campanhas devido ao acaso ou interromper campanhas promissoras prematuramente. Além de apresentar uma aplicação prática de Estatística Bayesiana, o projeto evidencia competências em modelagem probabilística, inferência por MCMC, comunicação de resultados e transformação de análises estatísticas em decisões de negócio orientadas por dados.

</p>

