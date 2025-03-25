# Relatório: Aproximação de Funções com Redes Neurais

## Introdução

Este relatório apresenta os resultados de um estudo sobre a capacidade de redes neurais artificiais em aproximar diferentes funções matemáticas. Utilizamos o framework scikit-learn para implementar e treinar Perceptrons Multicamadas (MLPs) com diferentes arquiteturas, avaliando seu desempenho na aproximação de funções de complexidade variada.

## Metodologia

### Conjuntos de Dados

Foram utilizados quatro conjuntos de dados diferentes para os testes:

- **Teste 2**: Função de complexidade básica
- **Teste 3**: Função de complexidade intermediária
- **Teste 4**: Função de complexidade elevada
- **Teste 5**: Função de complexidade muito elevada

Cada conjunto de dados consiste em pares de entrada-saída que representam pontos da função a ser aproximada.

### Arquiteturas de Rede Neural

Para cada conjunto de dados, testamos três arquiteturas diferentes de redes neurais, variando o número de camadas ocultas e neurônios:

**Teste 2**:
- Arquitetura 1: (10,) - Uma camada oculta com 10 neurônios
- Arquitetura 2: (15, 5) - Duas camadas ocultas com 15 e 5 neurônios
- Arquitetura 3: (20, 10, 5) - Três camadas ocultas com 20, 10 e 5 neurônios

**Teste 3**:
- Arquitetura 1: (20,) - Uma camada oculta com 20 neurônios
- Arquitetura 2: (30, 15) - Duas camadas ocultas com 30 e 15 neurônios
- Arquitetura 3: (40, 20, 10) - Três camadas ocultas com 40, 20 e 10 neurônios

**Teste 4**:
- Arquitetura 1: (25,) - Uma camada oculta com 25 neurônios
- Arquitetura 2: (35, 20) - Duas camadas ocultas com 35 e 20 neurônios
- Arquitetura 3: (50, 30, 15) - Três camadas ocultas com 50, 30 e 15 neurônios

**Teste 5**:
- Arquitetura 1: (30,) - Uma camada oculta com 30 neurônios
- Arquitetura 2: (40, 25) - Duas camadas ocultas com 40 e 25 neurônios
- Arquitetura 3: (60, 40, 20) - Três camadas ocultas com 60, 40 e 20 neurônios

### Parâmetros de Treinamento

Para todas as redes, utilizamos os seguintes parâmetros:
- Função de ativação: tangente hiperbólica (tanh)
- Otimizador: Adam
- Taxa de aprendizado: adaptativa
- Número máximo de iterações: 400
- Número de simulações por arquitetura: 10

## Resultados

### Gráficos a Incluir no Relatório


1. **Função Original**: Gráfico da função original que está sendo aproximada.

2. **Erro Médio com Desvio Padrão**: Gráfico de barras mostrando o erro médio e o desvio padrão para cada arquitetura testada.

3. **Descrições das Arquiteturas**: Tabela ou texto descrevendo cada arquitetura e seus respectivos erros médios e desvios padrão.

4. **Para cada arquitetura**:
   - **Curva de Aprendizado**: Gráfico mostrando a evolução do erro durante o treinamento.
   - **Aproximação da Função**: Gráfico comparando a função original (verde) com a função aproximada (azul).
   - **Distribuição de Erros**: Histograma mostrando a distribuição dos erros nas 10 simulações realizadas.

### Análise dos Resultados

Para cada conjunto de dados, analise:

1. Qual arquitetura obteve o menor erro médio?
2. Como o desvio padrão varia entre as diferentes arquiteturas?
3. Existe uma relação clara entre a complexidade da arquitetura e a qualidade da aproximação?
4. Como as curvas de aprendizado se comportam para cada arquitetura?

## Conclusões

Com base nos resultados obtidos, discuta:

1. A capacidade das redes neurais em aproximar funções de diferentes complexidades.
2. O impacto do número de camadas ocultas e neurônios no desempenho da aproximação.
3. O trade-off entre complexidade da rede e qualidade da aproximação.
4. Possíveis melhorias e trabalhos futuros.

## Referências

- Documentação do scikit-learn: [https://scikit-learn.org/stable/modules/neural_networks_supervised.html](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- Haykin, S. (2009). Neural networks and learning machines (3rd ed.). Pearson.
- Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.