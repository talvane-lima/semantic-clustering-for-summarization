import os
import json
import urllib.request
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

def generate_summary(prompt, temperature=0.8):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"Erro ao chamar Ollama: {e}")
        return "Erro na sumarização."

def build_prompt_baseline(texts):
    import random
    if len(texts) > 100:
        texts = random.sample(texts, 100)
    joined = "\n- ".join(texts)
    prompt = f"""You are a data analyst. Below are several customer reviews about a product, mixed together.
Provide a general and simple summary of the main points mentioned in a single cohesive paragraph. Do not split into pros and cons.
CRITICAL INSTRUCTION: YOU MUST ANSWER STRICTLY IN ENGLISH. BE CLEAR AND CONCISE.

REVIEWS:
- {joined}

SUMMARY:"""
    return prompt

def build_prompt_cluster(texts):
    import random
    if len(texts) > 100:
        texts = random.sample(texts, 100)
    joined = "\n- ".join(texts)
    prompt = f"""You are a data analyst. Below is a specific cluster of customer reviews that share a common theme or sentiment.
Analyze this cluster and provide a direct summary of the main topic discussed here (e.g., complaints about delivery, praise for battery life, etc.).
CRITICAL INSTRUCTION: YOU MUST ANSWER STRICTLY IN ENGLISH. BE CLEAR AND CONCISE.

CLUSTER REVIEWS:
- {joined}

CLUSTER SUMMARY:"""
    return prompt

def main():
    print("Iniciando a etapa de sumarização...\n")
    df = pd.read_csv("outputs/clustering/cluster_assignments.csv")
    
    out_dir = "outputs/summaries"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Baseline Summary
    print("Gerando Sumarização Global (Baseline)...")
    all_texts = df["text"].tolist()
    baseline_prompt = build_prompt_baseline(all_texts)
    baseline_summary = generate_summary(baseline_prompt)
    
    with open(os.path.join(out_dir, "baseline.md"), "w", encoding="utf-8") as f:
        f.write("# Sumarização Global (Sem Clusterização)\n\n")
        f.write(baseline_summary)
        f.write("\n")
    print("Baseline concluído!\n")
    
    # 2. Cluster Summaries (Usando o modelo: Hierarchical)
    print("Gerando Sumarização por Clusters (Modelo: Hierarchical)...")
    cluster_col = "Hierarchical"
    
    clusters = df[cluster_col].unique()
    clustered_results = []
    
    for c in sorted(clusters):
        print(f"  -> Sumarizando Cluster {c}...")
        cluster_texts = df[df[cluster_col] == c]["text"].tolist()
        cluster_prompt = build_prompt_cluster(cluster_texts)
        summary = generate_summary(cluster_prompt)
        
        clustered_results.append({
            "cluster_id": int(c),
            "size": len(cluster_texts),
            "summary": summary,
            "sample_texts": cluster_texts[:2] # Apenas para o log interno
        })
        
    with open(os.path.join(out_dir, "clustered_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Sumarização Baseada em Clusters\n\n")
        f.write(f"*Algoritmo utilizado para o agrupamento: {cluster_col}*\n\n")
        
        for res in clustered_results:
            f.write(f"## Cluster {res['cluster_id']} (Tamanho: {res['size']} comentários)\n")
            f.write(f"**Resumo do LLM:** {res['summary']}\n\n")
            f.write(f"**Exemplos Reais do Grupo:**\n")
            for t in res['sample_texts']:
                f.write(f"- *\"{t}\"*\n")
            f.write("\n---\n\n")
            
    print("Sumarização por clusters concluída!\n")
    print(f"Resultados salvos na pasta {out_dir}/")

if __name__ == "__main__":
    main()
