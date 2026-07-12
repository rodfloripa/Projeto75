# Otimização de campanhas com modelo hierárquico bayesiano


<p align="justify">Este projeto apresenta uma aplicação de Estatística Bayesiana Hierárquica para otimização de orçamento em campanhas de marketing digital. O objetivo é reduzir decisões incorretas causadas por pequenas amostras, estimando de forma probabilística a verdadeira taxa de conversão de cada campanha e quantificando explicitamente a incerteza associada às estimativas.</p>

<p align="justify">Em vez de utilizar apenas a taxa observada de cliques por impressões, o modelo considera que todas as campanhas pertencem à mesma conta publicitária e compartilham características em comum. Essa estrutura permite que campanhas com poucos dados sejam naturalmente aproximadas da média global da conta, reduzindo estimativas extremas produzidas apenas pelo acaso.</p>

---

# 1. Problema de Negócio

<p align="justify">Gestores de tráfego frequentemente administram dezenas ou centenas de campanhas simultaneamente. Nos primeiros dias de execução existe pouca informação disponível, fazendo com que pequenas variações produzam taxas de conversão extremamente altas ou extremamente baixas.</p>

<p align="justify">Tomar decisões apenas com base nessas taxas observadas pode levar ao aumento de investimento em campanhas que tiveram apenas sorte ou ao encerramento prematuro de campanhas potencialmente lucrativas.</p>

<p align="justify">Este projeto propõe uma abordagem Bayesiana Hierárquica capaz de produzir estimativas mais robustas para cada campanha, reduzindo o impacto da variabilidade causada por pequenas amostras.</p>

---

# 2. Objetivos

- Modelar campanhas utilizando um Modelo Beta-Binomial Hierárquico.
- Reduzir o efeito de pequenas amostras.
- Estimar distribuições posteriores para cada campanha.
- Calcular probabilidades diretamente interpretáveis.
- Apoiar decisões automáticas de orçamento.

---

# 3. Arquitetura

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

<p align="justify">Cada campanha possui uma taxa de conversão desconhecida que é modelada por uma distribuição Beta. Os hiperparâmetros dessa distribuição também são estimados a partir dos dados, permitindo que o modelo aprenda automaticamente qual é a taxa média típica da conta.</p>

<p align="justify">Essa estrutura hierárquica permite compartilhar informação entre todas as campanhas. Como consequência, campanhas com poucas observações são naturalmente aproximadas da média global, enquanto campanhas com grande quantidade de dados permanecem praticamente inalteradas.</p>

<p align="justify">Esse comportamento é conhecido como <b>Shrinkage Bayesiano</b> e constitui uma das principais vantagens dos modelos hierárquicos.</p>

---

# 5. Modelo Matemático

<p align="justify">Para cada campanha, considera-se que o número de cliques segue uma distribuição Binomial condicionada à taxa de conversão desconhecida.</p>

\[
Clicks_i \sim Binomial(Impressions_i,\theta_i)
\]

<p align="justify">A taxa de conversão de cada campanha é modelada por uma distribuição Beta.</p>

\[
\theta_i \sim Beta(\alpha,\beta)
\]

<p align="justify">Os hiperparâmetros também são aprendidos pelo próprio modelo.</p>

\[
\alpha,\beta \sim Exponential(1)
\]

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

# 7. Diagnóstico da Inferência

<p align="justify">Após a amostragem MCMC são avaliadas métricas de convergência para verificar se a distribuição posterior foi corretamente estimada.</p>

- R-hat
- Effective Sample Size (ESS)
- Trace Plots
- Distribuições Posteriores

<p align="justify">Esses diagnósticos garantem que a inferência produzida pelo algoritmo representa adequadamente a distribuição posterior dos parâmetros.</p>

---

# 8. Comparação com Estatística Clássica (statsmodels)

<p align="justify">Como comparação, pode-se utilizar a Estatística Clássica por meio da biblioteca <b>statsmodels</b>, calculando intervalos de confiança para cada campanha individualmente.</p>

<p align="justify">Nesse caso, cada campanha é analisada separadamente e não compartilha informação com as demais campanhas da conta.</p>

### Código principal

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

<p align="justify">Embora os intervalos de confiança produzidos pelo método clássico sejam adequados para grandes amostras, campanhas com poucos dados continuam apresentando elevada variabilidade, pois não existe compartilhamento de informação entre campanhas.</p>

<p align="justify">No modelo Bayesiano Hierárquico, campanhas pequenas são naturalmente aproximadas da média global da conta, reduzindo decisões equivocadas provocadas pelo acaso.</p>

---

# 9. Comparação entre Abordagens

| Característica | Estatística Clássica | Bayesiano Hierárquico |
|----------------|----------------------|------------------------|
| Compartilha informação | Não | Sim |
| Funciona bem com poucos dados | Limitado | Sim |
| Intervalo de confiança | Sim | Intervalo de credibilidade |
| Probabilidade direta | Não | Sim |
| Shrinkage | Não | Sim |
| Distribuição posterior | Não | Sim |

---

# 10. Visualizações

<p align="justify">O notebook gera visualizações que permitem interpretar facilmente o comportamento do modelo probabilístico.</p>

- Comparação entre taxa observada e taxa posterior.
- Gráfico do efeito de Shrinkage.
- Intervalos de credibilidade.
- Trace Plots.
- Distribuições posteriores.
- Probabilidade de superar a média global.

---

# 11. Resultados

<p align="justify">Após estimar a distribuição posterior de cada campanha, calcula-se a probabilidade de sua taxa de conversão ser superior à média global da conta.</p>

<p align="justify">Essa probabilidade P(Melhor que Média) é utilizada para transformar diretamente o resultado estatístico em uma decisão operacional.</p>

- Probabilidade superior a 80% → **ESCALAR**
- Probabilidade inferior a 20% → **PAUSAR**
- Demais casos → **MANTER E COLETAR MAIS DADOS**

| Campanha | Impressões | Taxa Bruta | Taxa MCMC | P(Melhor que Média) | Ação |
| :--- | :--- | :--- | :--- | :--- | :--- |
| C | 2 | 0.500 | 0.271 | 0.82 | ESCALAR |
| E | 10 | 0.200 | 0.187 | 0.78 | MANTER E COLETAR |
| H | 8 | 0.125 | 0.138 | 0.57 | MANTER E COLETAR |
| D | 5 | 0.000 | 0.073 | 0.25 | MANTER E COLETAR |
| G | 50 | 0.060 | 0.068 | 0.15 | PAUSAR |
| A | 5000 | 0.060 | 0.060 | 0.00 | PAUSAR |
| B | 8000 | 0.056 | 0.056 | 0.00 | PAUSAR |
| F | 300 | 0.050 | 0.051 | 0.00 | PAUSAR |
| I | 700 | 0.057 | 0.058 | 0.00 | PAUSAR |
| J | 1000 | 0.060 | 0.060 | 0.00 | PAUSAR |

<p align="justify">Essa estratégia reduz decisões precipitadas e melhora a alocação do orçamento publicitário.</p>
<p align="center">
  <img src="https://github.com/rodfloripa/Projeto75/blob/main/fig.png">
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

<p align="justify">A metodologia apresentada pode ser aplicada em diversos cenários de Ciência de Dados e Inteligência Artificial.</p>

- Marketing Digital
- Sistemas de Recomendação
- Testes A/B
- Avaliação de Produtos
- Comparação entre Vendedores
- Análise de Lojas
- Experimentação Online
- Machine Learning

---

# 14. Conclusão

<p align="justify">Este projeto demonstra como modelos Bayesianos Hierárquicos podem produzir estimativas significativamente mais robustas do que abordagens clássicas quando existem múltiplos grupos com diferentes quantidades de dados.</p>

<p align="justify">Ao incorporar explicitamente a incerteza na tomada de decisão, o modelo reduz o risco de escalar campanhas devido ao acaso ou interromper campanhas promissoras prematuramente.</p>

<p align="justify">Além de apresentar uma aplicação prática de Estatística Bayesiana, o projeto evidencia conhecimentos em modelagem probabilística, inferência por Cadeias de Markov Monte Carlo (MCMC), diagnóstico de convergência, comunicação de resultados e utilização de modelos estatísticos para suporte à decisão em problemas reais de negócio.</p>
