# Otimização de campanhas com modelo hierárquico bayesiano

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

- escalar campanhas aparentemente excelentes que tiveram apenas sorte;
- pausar campanhas promissoras que tiveram poucos eventos.

O objetivo deste projeto é utilizar um modelo probabilístico para estimar a verdadeira taxa de conversão de cada campanha considerando toda a informação disponível na conta.

</p>

---

# 2. Objetivos

<p align="justify">

O projeto busca:

- Modelar campanhas utilizando uma abordagem Bayesiana Hierárquica.

- Reduzir a variabilidade causada por pequenas amostras.

- Estimar intervalos de credibilidade para cada campanha.

- Calcular probabilidades diretamente interpretáveis para decisões de negócio.

- Transformar estimativas estatísticas em ações operacionais.

</p>

---

# 3. Arquitetura do Projeto

```text
                  Dados das Campanhas
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
        Probabilidade de Superar a Média
                          │
                          ▼
        ESCALAR | MANTER | PAUSAR
```

---

# 4. Fundamentação Estatística

<p align="justify">

O projeto utiliza um **Modelo Beta-Binomial Hierárquico**, amplamente empregado em problemas de inferência Bayesiana envolvendo proporções.

Cada campanha possui uma taxa de conversão desconhecida, modelada por uma distribuição Beta. Ao mesmo tempo, todas as campanhas compartilham hiperparâmetros comuns que representam o comportamento médio da conta.

Essa estrutura permite que campanhas com poucas observações sejam naturalmente aproximadas da média global, reduzindo estimativas extremas produzidas apenas pelo acaso.

Esse fenômeno é conhecido como **Shrinkage Bayesiano**, uma das principais vantagens dos modelos hierárquicos.

</p>

---

# 5. Modelo Matemático

<p align="justify">

Para cada campanha \(i\):

\[
Clicks_i \sim Binomial(Impressions_i,\theta_i)
\]

onde

\[
\theta_i \sim Beta(\alpha,\beta)
\]

e os hiperparâmetros são estimados a partir dos próprios dados:

\[
\alpha,\beta \sim Exponential(1)
\]

Após a inferência MCMC obtém-se a distribuição posterior de cada taxa de conversão.

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
        draws=2000,
        tune=1000,
        target_accept=0.9,
        random_seed=42
    )
```

---

# 7. Diagnóstico do Modelo

<p align="justify">

Após a inferência são avaliados indicadores de convergência, como:

- R-hat próximo de 1;

- Effective Sample Size (ESS);

- Trace Plots;

- Distribuições posteriores.

Essas análises garantem que o algoritmo MCMC convergiu adequadamente para a distribuição posterior.

</p>

---

# 8. Comparação com Estatística Clássica (statsmodels)

<p align="justify">

Como referência, uma abordagem clássica pode ser construída utilizando o pacote **statsmodels**.

Nesse caso, cada campanha é analisada individualmente através de um intervalo de confiança para proporções, sem compartilhar informação com as demais campanhas.

</p>

### Código Principal

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

Embora o método clássico produza intervalos de confiança adequados para cada campanha individualmente, ele não compartilha informação entre campanhas.

Como consequência, campanhas com poucas observações continuam apresentando elevada variabilidade.

Já o modelo Bayesiano Hierárquico utiliza informação global da conta para reduzir essa instabilidade.

</p>

---

# 9. Comparação entre Abordagens

| Característica | Estatística Clássica | Bayesiano Hierárquico |
|----------------|----------------------|------------------------|
| Compartilha informação | Não | Sim |
| Funciona bem com poucos dados | Limitado | Sim |
| Intervalo de confiança | Sim | Intervalo de credibilidade |
| Probabilidade direta de ser melhor | Não | Sim |
| Shrinkage | Não | Sim |
| Distribuição posterior | Não | Sim |

---

# 10. Visualizações Produzidas

<p align="justify">

O notebook gera visualizações que facilitam a interpretação dos resultados, incluindo:

- gráfico de efeito de shrinkage;

- comparação entre taxa observada e taxa posterior;

- intervalos de credibilidade;

- trace plots;

- distribuições posteriores;

- probabilidade de cada campanha superar a média global.

</p>

---

# 11. Decisão de Negócio

<p align="justify">

Após estimar a distribuição posterior de cada campanha, calcula-se a probabilidade de sua taxa de conversão ser superior à média global da conta.

Com base nessa probabilidade, define-se automaticamente uma ação operacional.

Exemplo:

- Probabilidade superior a 80% → **ESCALAR**

- Probabilidade inferior a 20% → **PAUSAR**

- Demais casos → **MANTER E COLETAR MAIS DADOS**

Essa estratégia substitui decisões baseadas apenas na taxa observada por decisões fundamentadas em probabilidade.

</p>

---

# 12. Tecnologias Utilizadas

- Python

- PyMC

- ArviZ

- NumPy

- Pandas

- Matplotlib

- statsmodels

---

# 13. Aplicações

<p align="justify">

A metodologia apresentada pode ser utilizada em diversos problemas reais, incluindo:

- otimização de campanhas de marketing;

- testes A/B com múltiplas variantes;

- recomendação de produtos;

- comparação entre vendedores;

- análise de desempenho de lojas;

- monitoramento de modelos de Machine Learning;

- experimentação online;

- sistemas de recomendação.

</p>

---

# 14. Conclusão

<p align="justify">

Este projeto demonstra como modelos Bayesianos Hierárquicos podem produzir estimativas significativamente mais robustas do que abordagens clássicas quando existem múltiplos grupos com diferentes quantidades de dados.

Ao incorporar explicitamente a incerteza na tomada de decisão, o modelo reduz o risco de escalar campanhas devido ao acaso ou interromper campanhas promissoras prematuramente.

Além de apresentar uma aplicação prática de Estatística Bayesiana, o projeto evidencia conhecimentos em modelagem probabilística, inferência por Cadeias de Markov Monte Carlo (MCMC), diagnóstico de convergência, comunicação de resultados e utilização de modelos estatísticos para suporte à decisão em problemas reais de negócio.

</p>
