# Semantic Clustering for Summarization

Este projeto explora o impacto da clusterização semântica de comentários no resultado final da sumarização por Modelos de Linguagem (LLMs). A proposta é agrupar logicamente os dados de entrada usando técnicas de NLP e clusterização antes de submetê-los à sumarização, criando resumos muito mais granulares, assertivos e representativos.

## Como Rodar o Projeto

### Pré-requisitos
Antes de executar qualquer etapa, instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```
*Observação: A geração de comentários e resumos utiliza o Ollama local (por padrão o modelo qwen2.5:14b e qwen2.5:3b). Certifique-se de que o Ollama esteja rodando.*

### A Sequência Completa (Full Pipeline)
Se quiser rodar o projeto inteiro do zero:
1. **`python src/generate_comments.py`**: Cria os comentários sintéticos simulados.
2. **`python src/generate_embeddings.py`**: Transforma os comentários em vetores semânticos.
3. **`python src/cluster_embeddings.py`**: Agrupa os vetores usando UMAP e algoritmos de clusterização.
4. **`python src/summarize_clusters.py`**: Gera os resumos baseline e clusterizados.

### Atalhos (Ponto de Partida)
* **Já possui os comentários (comentarios.json)?** Pule o passo 1 e comece rodando `generate_embeddings.py`.
* **Já possui os Embeddings (outputs/embeddings/)?** Comece a partir do passo 3 rodando `cluster_embeddings.py`.

## Arquivos do Projeto

Aqui está a explicação do que cada script principal realiza na nossa arquitetura:

* **`src/generate_comments.py`**: Responsável por gerar um conjunto de dados sintético via Ollama. Ele simula clientes reais comentando sobre serviços, assinaturas, cancelamentos, preços, etc.
* **`src/generate_embeddings.py`**: Pega o texto limpo e o transforma em representações matemáticas densas (embeddings) usando modelos sentence transformers. Essa etapa permite o cálculo de similaridade de contexto matemático.
* **`src/cluster_embeddings.py`**: O motor analítico. Ele reduz a dimensão espacial dos embeddings (com UMAP) e testa diversos algoritmos de agrupamento (K-Means, HDBSCAN, Hierárquico) extraindo as métricas e identificando as comunidades semânticas (Clusters).
* **`src/summarize_clusters.py`**: Responsável pela síntese via LLM. Ele atua em duas frentes: lendo a base global para criar um resumo tradicional (Baseline) e lendo cada grupo separadamente para criar o resumo avançado (Clustered).

## Estrutura de Outputs (Saídas)
Todos os resultados gerados pela pipeline são centralizados na pasta outputs:
* **`outputs/embeddings/`**: Contém a matriz vetorial embeddings.npy e metadados (metadata.csv).
* **`outputs/clustering/`**: Guarda a tabela de atribuição final de cada comentário e um arquivo .json com todas as métricas dos testes de clusterização.
* **`outputs/clustering/plots/`**: Salva os gráficos da separação dos modelos, Heatmaps de verificação e a projeção UMAP Comparativa com legendas de grupos visuais.
* **`outputs/summaries/`**: Repositório dos documentos Markdown com a sumarização LLM final gerada pelo Ollama.

## Resultados: Baseline vs Clustered Summarization

A diferença prática da nossa metodologia pode ser observada nos arquivos finais de saída:

### O Resumo Baseline (outputs/summaries/baseline_summary.md)
No resumo base (onde enviamos todos os comentários embolados diretamente ao LLM), a inteligência artificial consegue pegar o "tom geral" dos clientes que elogiam a rapidez ou o preço, mas os detalhes vitais se perdem na massa de texto.
**Ponto Crítico:** Ele **não fala absolutamente nada** sobre cancelamentos ou problemas pontuais, ocultando uma informação grave.

### O Resumo Clusterizado (outputs/summaries/clustered_summary.md)
Quando tratamos o texto cluster a cluster antes de resumir, conseguimos ter uma visão infinitamente mais rica e precisa do que cada microgrupo está falando. O LLM extrai e foca na essência de cada pilar de comunicação, capturando com exatidão o grupo isolado que está criticando o fluxo de devolução ou processo de cancelamento.

## Aplicações no Mundo Real e Casos de Uso

A possibilidade de sumarizar e atuar isoladamente em clusters semânticos vai muito além do simples resumo. Ela habilita:

* **Detecção de Anomalias:** Encontrar e separar imediatamente as "agulhas no palheiro" em meio a milhares de feedbacks elogiosos.
* **Isolamento de Contexto Específico:** Dar uma visão cirúrgica para a área de produto sem ruídos de outras métricas.
* **Tratamento Específico (Cancelamento/Devolução/Jurídico):** Direcionar clusters que contenham contexto sobre "judicialização" ou problemas graves de cancelamento imediatamente para a fila de atendimento prioritário.
* **Sistemas de Detecção Alarmística (Triggers):** Desenvolver um sistema que dispara e-mails preventivos para as lideranças caso o cluster semântico voltado a cancelamento de contas de repente comece a crescer rapidamente na base de processamento.
