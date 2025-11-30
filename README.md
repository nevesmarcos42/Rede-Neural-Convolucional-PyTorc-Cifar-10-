# Rede Neural Convolucional - CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?style=for-the-badge&logo=nvidia)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

Modelo de rede neural convolucional (CNN) para classificação de imagens no conjunto de dados CIFAR-10 utilizando PyTorch. Atinge precisão de 86-88% com técnicas modernas de data augmentation e otimização.

[Funcionalidades](#funcionalidades) • [Tecnologias](#tecnologias) • [Instalação](#instalação) • [Uso](#uso) • [Resultados](#resultados) • [Contribuir](#contribuindo)

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias](#tecnologias)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [Resultados](#resultados)
- [Melhorias Futuras](#melhorias-futuras)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## Sobre o Projeto

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificação de imagens do conjunto de dados CIFAR-10. O CIFAR-10 consiste em 60.000 imagens coloridas de 32x32 pixels divididas em 10 classes diferentes.

### Principais Características

- **CNN Otimizada** - Arquitetura inspirada em VGG-16 com múltiplas camadas convolucionais
- **Data Augmentation** - Técnicas avançadas para aumentar a diversidade dos dados
- **Suporte a GPU** - Treinamento acelerado com CUDA
- **Visualizações** - Ferramentas para exploração de dados e resultados
- **Modular** - Código organizado e reutilizável
- **Jupyter Notebook** - Experimentação interativa incluída

### Classes do CIFAR-10

O modelo classifica imagens nas seguintes categorias:

1. Avião (airplane)
2. Automóvel (automobile)
3. Pássaro (bird)
4. Gato (cat)
5. Cervo (deer)
6. Cachorro (dog)
7. Sapo (frog)
8. Cavalo (horse)
9. Navio (ship)
10. Caminhão (truck)

## Funcionalidades

### Pré-processamento de Dados

- **Normalização** - Padronização dos valores de pixel
- **Data Augmentation** - Flip horizontal, rotação, jitter de cor
- **Recorte Aleatório** - Aumenta a robustez do modelo
- **Validação Cruzada** - Separação treino/teste

### Arquitetura da Rede

- **Camadas Convolucionais** - Extração hierárquica de características
- **Pooling** - Redução de dimensionalidade
- **Batch Normalization** - Estabilização do treinamento
- **Dropout** - Prevenção de overfitting
- **Fully Connected** - Classificação final

### Treinamento

- **Otimizador Adam** - Convergência eficiente
- **Learning Rate Scheduler** - Ajuste dinâmico da taxa de aprendizado
- **Early Stopping** - Prevenção de overfitting
- **Checkpoints** - Salvamento automático do melhor modelo
- **Métricas** - Acurácia, perda e matriz de confusão

### Visualização

- **Amostragem de Dados** - Visualização de imagens originais
- **Curvas de Aprendizado** - Perda e acurácia por época
- **Matriz de Confusão** - Análise de erros de classificação
- **Ativações** - Visualização de mapas de características

## Tecnologias

| Tecnologia  | Versão | Descrição                  |
| ----------- | ------ | -------------------------- |
| Python      | 3.8+   | Linguagem de programação   |
| PyTorch     | 2.0+   | Framework de deep learning |
| torchvision | 0.15+  | Datasets e transformações  |
| NumPy       | 1.24+  | Computação numérica        |
| Matplotlib  | 3.7+   | Visualização de dados      |
| Jupyter     | 1.0+   | Ambiente interativo        |
| CUDA        | 11.0+  | Aceleração GPU (opcional)  |

## Arquitetura

### Estrutura do Repositório

```
Rede-Neural-Convolucional-PyTorc-Cifar-10-/
├── dataset/              # Conjunto de dados CIFAR-10
├── models/               # Arquiteturas de CNN
├── notebooks/            # Jupyter Notebooks com experimentos
├── src/                  # Código principal do projeto
│   ├── train.py          # Script de treinamento do modelo
│   ├── evaluate.py       # Avaliação do modelo
│   ├── preprocess.py     # Funções de pré-processamento
│   └── visualize.py      # Visualizações de imagens e dados
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação
```

### Arquitetura da CNN

```
Input (32x32x3)
    ↓
Conv2D (64 filtros) → BatchNorm → ReLU
    ↓
Conv2D (64 filtros) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (128 filtros) → BatchNorm → ReLU
    ↓
Conv2D (128 filtros) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (256 filtros) → BatchNorm → ReLU
    ↓
Conv2D (256 filtros) → BatchNorm → ReLU → MaxPool
    ↓
Flatten
    ↓
Fully Connected (512) → Dropout → ReLU
    ↓
Fully Connected (10) → Softmax
    ↓
Output (10 classes)
```

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- GPU com CUDA (opcional, mas recomendado)

### Instalação Local

#### 1. Clone o repositório

```bash
git clone https://github.com/nevesmarcos42/Rede-Neural-Convolucional-PyTorc-Cifar-10-.git
cd Rede-Neural-Convolucional-PyTorc-Cifar-10-
```

#### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

#### 4. Baixe o conjunto CIFAR-10

```python
from torchvision.datasets import CIFAR10
CIFAR10(root='./dataset', download=True)
```

#### 5. Verifique a instalação

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

## Uso

### Treinamento do Modelo

#### Treinamento básico

```bash
python src/train.py
```

#### Treinamento com parâmetros customizados

```bash
python src/train.py --epochs 50 --batch-size 128 --lr 0.001 --device cuda
```

### Avaliação do Modelo

```bash
python src/evaluate.py --model-path models/cifar10_cnn.pth
```

### Exemplo de Uso no Código

```python
from models.cnn import CNNClassifier
import torch

# Carregar modelo treinado
model = CNNClassifier()
model.load_state_dict(torch.load("models/cifar10_cnn.pth"))
model.eval()

# Inferência em uma imagem do CIFAR-10
sample_image = torch.randn(1, 3, 32, 32)  # Simulação de entrada
prediction = model(sample_image).argmax(dim=1)
print(f"Classe prevista: {prediction.item()}")
```

### Usando o Jupyter Notebook

```bash
jupyter notebook notebooks/Rede_Neural_Convolucional_PyTorc_(Cifar_10).ipynb
```

### Visualização de Dados

```python
from src.visualize import plot_samples, plot_training_history

# Visualizar amostras do dataset
plot_samples(train_loader, num_samples=16)

# Visualizar histórico de treinamento
plot_training_history(history)
```

## Resultados

### Métricas de Performance

- **Acurácia no Treino**: ~92%
- **Acurácia no Teste**: 86-88%
- **Tempo de Treinamento**: ~20 minutos (GPU) / ~2 horas (CPU)
- **Parâmetros do Modelo**: ~2.5M

### Classes com Melhor Performance

- Avião: 91%
- Navio: 90%
- Caminhão: 88%

### Classes com Maior Dificuldade

- Gato vs Cachorro: 78%
- Cervo vs Cavalo: 80%

## Melhorias Futuras

### Arquitetura

- [ ] Implementar ResNet para melhor propagação de gradientes
- [ ] Testar EfficientNet para redução de parâmetros
- [ ] Adicionar Attention Mechanisms

### Data Augmentation

- [ ] Implementar CutMix
- [ ] Implementar MixUp
- [ ] Adicionar AutoAugment

### Otimização

- [ ] Ajuste fino de hiperparâmetros com Optuna
- [ ] Transfer Learning com modelos pré-treinados
- [ ] Quantização para inferência mais rápida

### Deployment

- [ ] API REST com FastAPI
- [ ] Interface web com Streamlit
- [ ] Containerização com Docker

## Contribuindo

Contribuições são bem-vindas! Siga os passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Padrões de Código

- Seguir PEP 8 para código Python
- Adicionar docstrings em todas as funções
- Escrever testes para novas funcionalidades
- Documentar mudanças significativas

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

Desenvolvido como projeto de estudo em Deep Learning e Computer Vision

**Versão**: 1.0.0

**Última Atualização**: Novembro 2025
