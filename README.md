# Relatório: Aproximação de Funções com Redes Neurais

## Membros
* Cayque Cicarelli - 22.221.005-6

* Bruna Paz - 22.121.020-6

* Matheus Miranda - 22.22.0017-2


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

### Gráficos

1. **Função Original**: Gráfico da função original que está sendo aproximada.

![image](https://github.com/user-attachments/assets/8ebb5180-c29f-4385-bcde-a7f2198dec70)


2. **Erro Médio com Desvio Padrão**: 

![image](https://github.com/user-attachments/assets/3c8cccda-b178-43a4-9033-ec6d2df4dafe)


3. **Curva de Aprendizado**: 

![image](https://github.com/user-attachments/assets/48de3dda-3ecd-4b3e-a8b7-d02f7f8f13ba)


4. **Aproximação da Função**:

![image](https://github.com/user-attachments/assets/9a096135-c9d4-45e4-84c6-9e0f67dd72ed)

   
### Análise dos Resultados

Para cada conjunto de dados, analise:

* Qual arquitetura obteve o menor erro médio?

Arquitetura 3 com um erro médio de 0.001016 e desvio padrão de 0.000686.


