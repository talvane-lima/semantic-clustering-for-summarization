# Semantic Clustering for Summarization

Este projeto visa testar o impacto da clusterização semântica de comentários no resultado final da sumarização por Modelos de Linguagem (LLMs). A ideia é organizar a entrada de forma estruturada baseada no sentido (clusters semânticos) antes de submetê-la ao processo de sumarização, para verificar melhorias na coesão, precisão e representatividade do resumo final.

## Etapa 1: Geração de Dados

Para realizar nossos testes, geramos um conjunto sintético de comentários de usuários focados num produto hipotético. Utilizamos o LLM local `qwen2.5:14b` (via API do Ollama). 

A geração está distribuída para simular um cenário ruidoso e diversificado de avaliações:
- **70%** focados no Preço / Custo-benefício.
- **20%** focados na Durabilidade.
- **5%** focados na Estética do produto.
- **2%** focados em problemas de Pós-venda.
- **2%** focados em problemas de Entrega.
- **1%** off-topic (comentários sem relação com o produto, como política, religião ou correntes de internet).

### Como gerar a base de comentários

Foi criado o script `src/generate_comments.py` usando as bibliotecas padrão do Python (dispensando pacotes externos no momento, com exceção da API Ollama que precisa estar ativa).

**Uso básico:**
```bash
python src/generate_comments.py
```

**Parâmetros suportados:**
- `--num_comments`: Define a quantidade de comentários (padrão: 500).
- `--output`: Define onde salvar o json de saída (padrão: comentarios.json).
- `--model`: Define a string do modelo no Ollama (padrão: qwen2.5:14b).

**Exemplo de geração customizada:**
```bash
python src/generate_comments.py --num_comments 1000 --output data/comentarios_1000.json
```

---

## Etapa 2: Geração de Embeddings Semânticos

O objetivo desta etapa é transformar os comentários de texto livre em vetores numéricos de alta qualidade utilizando modelos de embeddings da família Transformer, capturando assim as nuances semânticas de cada texto para realizar o agrupamento nos passos futuros.

Utilizamos a biblioteca `sentence-transformers` com suporte automático para aceleração via GPU (CUDA). O modelo padrão adotado é o `intfloat/multilingual-e5-base`, que é eficiente para português.

### Como gerar os embeddings

Execute o script abaixo que processa os comentários, valida anomalias e gera a representação multidimensional normalizada (L2=1).

```bash
python src/generate_embeddings.py
```

**Parâmetros suportados:**
- `--input`: O caminho do JSON de comentários criado na Etapa 1 (padrão: `comentarios.json`).
- `--output`: O diretório onde serão salvos os artefatos finais (padrão: `outputs/embeddings`).
- `--model`: Qual modelo de embedding usar no HuggingFace (padrão: `intfloat/multilingual-e5-base`).
- `--batch-size`: O tamanho do lote para passar pelo modelo (padrão: 64).

**Artefatos Gerados:**
- `embeddings.npy`: Matriz NumPy com os vetores normalizados (número de comentários x dimensão).
- `metadata.csv`: Mapeamento das linhas da matriz NumPy com o ID, tópico e texto.
- `embedding_info.json`: Dados sobre a geração (modelo, dimensões, metadados extras).

*Diário de Projeto atualizado ao decorrer dos testes.*
