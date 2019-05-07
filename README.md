# Projeto Demonstrativo 3
##### Aluno: Pedro Garcia
##### Matrícula: 15/0019891

* OS: Ubuntu 18.10
* Versão do Python: 3.7.2
* Versão do OpenCV: 3.4.5
* Pacotes Necessários: Numpy, Matplotlib, OpenCV (com suporte a SIFT)

## Repositório no GitHub
* https://github.com/aspectum/pvc-proj3

## Comando para execução
* Do diretório principal:
    - Para o requisito 1: `python3 src/pd3.py --req1`
    - Para o requisito 2: `python3 src/pd3.py --req2`
    - Para o requisito 2 (sem retificar as imagens): `python3 src/pd3.py --req2_non_rect`
* Para compilar e executar o algoritmo SAD implementado:
    - Compilar com `make` de dentro do diretório src
    - Executar do diretório principal `src/rectifiedSADmatcher`

## Arquivos
```
Pedro_Garcia
├── README.md
├── Pedro_Garcia.pdf
├── /relatorio
│   └── arquivos fontes do LaTeX
├── /src
│   └── pd3.py
│   └── req1.py
│   └── req2.py
│   └── req2_meu.py
│   └── rectifiedSADmatcher.cpp
|   └── Makefile
└── /data
    ├── /FurukawaPonce
    ├── /Middlebury
    |   └── /Jadeplant-perfect
    |   └── /Motorcycle-perfect
    └── /output
```

## Observações
* Imagens de entrada organizadas como no arquivo data.zip fornecido no Moodle.
* Já vão os resultados em output, principalmente porque não é viável a pessoa corrigindo rodar o req2_non_rect.
    - Esse requisito está pegando de um arquivo txt e apenas gerando a imagem.
    - Se quiser rodar de fato tem que descomentar a chamada da função.
    - Eu levei 6 horas pra rodar.
* O rectifiedSADmatcher eu levei 1 hora pra rodar.