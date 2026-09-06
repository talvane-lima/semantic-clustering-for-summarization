import argparse
import json
import logging
import time
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_comments(filepath):
    """Lê o arquivo JSON contendo os comentários."""
    logging.info(f"Carregando comentários de '{filepath}'")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def validate_comments(df):
    """Valida a presença de colunas e remove/anota registros com text inválido ou duplicado."""
    logging.info("Validando comentários...")
    required_cols = ['id', 'topic', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    # Remove textos vazios ou nulos
    initial_count = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    final_count = len(df)
    if initial_count != final_count:
        logging.warning(f"Removidos {initial_count - final_count} comentários com 'text' vazio ou nulo.")

    # Informar duplicatas exatas sem removê-las
    duplicates = df.duplicated(subset=['text'], keep=False).sum()
    if duplicates > 0:
        logging.info(f"Encontrados {duplicates} comentários com texto exatamente duplicado (mantidos no dataset).")

    return df.reset_index(drop=True)

def load_embedding_model(model_name):
    """Carrega o modelo do HuggingFace via SentenceTransformers com suporte a GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Carregando modelo '{model_name}' utilizando dispositivo '{device}'...")
    model = SentenceTransformer(model_name, device=device)
    return model, device

def generate_embeddings(model, texts, batch_size, is_e5_model=False):
    """Gera os embeddings em batches e já normaliza (L2)."""
    logging.info(f"Gerando embeddings para {len(texts)} textos com batch_size={batch_size}...")
    
    # Aplicação de prefixo exigido para modelos intfloat/multilingual-e5
    if is_e5_model:
        texts = ["passage: " + text for text in texts]
        
    start_time = time.time()
    # model.encode faz batching interno e normalização se solicitado
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    logging.info(f"Geração de embeddings concluída em {elapsed_time:.2f} segundos.")
    return embeddings, elapsed_time

def save_outputs(output_dir, embeddings, df, model_name, dim, normalized, metric):
    """Salva os embeddings, o metadata.csv e o arquivo de log JSON informacional."""
    logging.info(f"Salvando resultados no diretório '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Salvar embeddings.npy
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    
    # 2. Salvar metadata.csv
    df_meta = df[['id', 'topic', 'text']].copy()
    df_meta.index.name = 'index'
    df_meta.to_csv(os.path.join(output_dir, "metadata.csv"))
    
    # 3. Salvar embedding_info.json
    info = {
        "model": model_name,
        "embedding_dimension": int(dim),
        "normalized": normalized,
        "similarity_metric_recommended": metric,
        "number_of_comments": len(df)
    }
    with open(os.path.join(output_dir, "embedding_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)
        
    logging.info("Arquivos de saída salvos com sucesso.")

def validate_embeddings(embeddings, expected_count):
    """Verifica número de linhas, valores problemáticos, dimensões e normalização."""
    logging.info("Executando verificações automáticas nos embeddings...")
    
    # 1. Tamanho do conjunto de dados
    if embeddings.shape[0] != expected_count:
        raise ValueError(f"Erro de tamanho: esperados {expected_count} embeddings, obtidos {embeddings.shape[0]}.")
        
    # 2. Checagem de NaN e Infinito
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Existem valores NaN ou Infinitos nos vetores gerados!")
        
    # 3. Dimensão do embedding
    dim = embeddings.shape[1]
    logging.info(f"Dimensão detectada do embedding: {dim}")
    
    # 4. Checar Normalização L2
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    if not np.isclose(mean_norm, 1.0, atol=1e-4):
        logging.warning(f"Atenção: A norma média L2 dos vetores é {mean_norm:.4f} (esperado 1.0).")
    else:
        logging.info("Embeddings estão devidamente normalizados (norma L2 = 1.0).")
        
    return dim

def find_nearest_comments(embeddings, df, num_samples=3, num_neighbors=5):
    """Seleciona alguns comentários e apresenta seus vizinhos mais próximos."""
    logging.info("\n--- Teste Qualitativo de Similaridade ---")
    if len(df) <= num_neighbors:
        logging.info("Não há comentários suficientes para buscar vizinhos qualitativamente.")
        return
        
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), num_samples, replace=False)
    
    # Calcular similaridade de cosseno desses samples com toda a base
    sim_matrix = cosine_similarity(embeddings[sample_indices], embeddings)
    
    for i, idx in enumerate(sample_indices):
        orig_text = df.iloc[idx]['text']
        orig_topic = df.iloc[idx]['topic']
        
        sims = sim_matrix[i]
        nearest_indices = np.argsort(sims)[::-1]
        nearest_indices = [n for n in nearest_indices if n != idx][:num_neighbors]
        
        print(f"\nComentário original (topic={orig_topic}):\n{orig_text}")
        print("Vizinhos:")
        for rank, n_idx in enumerate(nearest_indices, 1):
            sim = sims[n_idx]
            topic = df.iloc[n_idx]['topic']
            text = df.iloc[n_idx]['text']
            print(f"{rank}. similarity={sim:.4f} | topic={topic} | {text}")
            
    print("-----------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Gera embeddings semânticos para os comentários gerados.")
    parser.add_argument("--input", type=str, default="comentarios.json", help="Arquivo JSON de entrada.")
    parser.add_argument("--output", type=str, default="outputs/embeddings", help="Diretório de saída.")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base", help="Modelo de embeddings do HuggingFace.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do batch para processamento no modelo.")
    
    args = parser.parse_args()
    
    # 1. Carregar dados
    df = load_comments(args.input)
    
    # 2. Validar integridade
    df = validate_comments(df)
    
    # 3. Inicializar o modelo de embeddings
    model, device = load_embedding_model(args.model)
    
    # 4. Gerar vetores
    # Adicionar o prefixo `passage: ` caso estejamos usando a família E5 conforme boas práticas do paper
    is_e5 = "e5" in args.model.lower()
    texts = df['text'].tolist()
    embeddings, elapsed_time = generate_embeddings(model, texts, args.batch_size, is_e5)
    
    # 5. Validações automáticas
    dim = validate_embeddings(embeddings, expected_count=len(df))
    
    # 6. Salvar todos os artefatos
    save_outputs(args.output, embeddings, df, args.model, dim, normalized=True, metric="cosine")
    
    # 7. Diagnóstico Qualitativo (Vizinhos)
    find_nearest_comments(embeddings, df)
    
    logging.info(f"Processo finalizado com sucesso! Processados {len(df)} comentários em {elapsed_time:.2f}s no dispositivo '{device}'.")

if __name__ == "__main__":
    main()
