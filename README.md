# Rede-Neural-Convolucional-PyTorc-Cifar-10-
Este repositório contém um modelo de rede neural convolucional (CNN) para classificação de imagens no conjunto de dados CIFAR-10, utilizando PyTorch.


 Descrição do Projeto
O objetivo deste projeto é desenvolver um modelo de aprendizado profundo para classificar imagens em 10 categorias do conjunto CIFAR-10. A implementação inclui técnicas modernas de pré-processamento de dados, otimização e visualização, garantindo um modelo eficiente e preciso.

 Estrutura do Repositório
├── dataset/              # Conjunto de dados CIFAR-10
├── models/               # Arquiteturas de CNN
├── notebooks/            # Jupyter Notebooks com experimentos
├── src/                  # Código principal do projeto
│   ├── train.py          # Script de treinamento do modelo
│   ├── evaluate.py       # Avaliação do modelo
│   ├── preprocess.py     # Funções de pré-processamento
│   ├── visualize.py      # Visualizações de imagens e dados
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação


 Recursos do Projeto

 CNN aprimorada – Estrutura inspirada em redes como VGG-16 para melhor extração de características.

 Data Augmentation – Técnicas como flip horizontal, rotação, jitter de cor e recorte aleatório para aumentar a diversidade dos dados.

 Treinamento eficiente – Compatível com GPU (CUDA) e ajustável para execução em CPU.

 Otimização avançada – Uso de Adam optimizer e ajuste da taxa de aprendizado com Learning Rate Scheduler.

 Visualização de dados – Funções para explorar amostras antes e depois do pré-processamento.

 Instalação
1 Clone o repositório:
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio


2 Instale as dependências:
pip install -r requirements.txt


3️ Baixe o conjunto CIFAR-10:
from torchvision.datasets import CIFAR10
CIFAR10(download=True)


4️ Execute o treinamento do modelo:
python src/train.py


 Exemplo de Uso
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


 Resultados e Melhorias Futuras
A rede atinge uma acurácia de aproximadamente 86-88% no conjunto CIFAR-10.
Para melhorias futuras, podemos testar:
- Arquiteturas mais profundas como ResNet ou EfficientNet
- Ajuste fino dos hiperparâmetros (learning rate, número de filtros, etc.)
- Data Augmentation mais avançada com técnicas como CutMix ou MixUp
