import argparse
import random
import json
import urllib.request
import urllib.error
import time

def generate_comment(topic_prompt, model="qwen2.5:14b", url="http://localhost:11434/api/generate"):
    """
    Chama a API local do Ollama para gerar um texto com base no prompt.
    """
    data = {
        "model": model,
        "prompt": topic_prompt,
        "stream": False
    }
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"\nErro ao conectar na API local (Ollama rodando na porta 11434?): {e}")
        return None
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Gera comentários simulados sobre produtos usando LLM local.")
    parser.add_argument("--num_comments", type=int, default=500, help="Quantidade total de comentários a serem gerados.")
    parser.add_argument("--output", type=str, default="comentarios.json", help="Arquivo de saída (.json).")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Modelo a ser utilizado (Ollama).")
    
    args = parser.parse_args()
    
    # Distribuição alvo
    distribution = {
        "price": 0.70,
        "durability": 0.20,
        "aesthetics": 0.05,
        "post_sales": 0.02,
        "delivery": 0.02,
        "off_topic": 0.01
    }

    # Prompts de contexto para o modelo gerar a avaliação com o viés correto
    prompts = {
        "price": "Você é um cliente avaliando um produto. Escreva um comentário curto (1 a 3 frases) focando EXCLUSIVAMENTE no PREÇO (pode ser sobre custo-benefício, ser muito caro, ou muito barato). Apenas o comentário, sem introduções.",
        "durability": "Você é um cliente avaliando um produto. Escreva um comentário curto (1 a 3 frases) focando EXCLUSIVAMENTE na DURABILIDADE (qualidade do material, se quebrou rápido, se dura muito). Apenas o comentário, sem introduções.",
        "aesthetics": "Você é um cliente avaliando um produto. Escreva um comentário curto (1 a 3 frases) focando EXCLUSIVAMENTE na ESTÉTICA (beleza, design, cor, feio/bonito). Apenas o comentário, sem introduções.",
        "post_sales": "Você é um cliente frustrado. Escreva um comentário curto (1 a 3 frases) reclamando EXCLUSIVAMENTE de problemas com o PÓS-VENDA (atendimento, dificuldade para trocar, suporte ruim). Apenas o comentário, sem introduções.",
        "delivery": "Você é um cliente frustrado. Escreva um comentário curto (1 a 3 frases) reclamando EXCLUSIVAMENTE de problemas na ENTREGA (atraso enorme, chegou quebrado pela transportadora). Apenas o comentário, sem introduções.",
        "off_topic": "Escreva um comentário curto de internet (1 a 3 frases) que NÃO tenha nada a ver com produto. Pode ser sobre política, religião, correntes, ou algo totalmente aleatório. Apenas o comentário, sem introduções."
    }

    print(f"Calculando a distribuição para {args.num_comments} comentários...")
    
    topics = []
    for topic, pct in distribution.items():
        count = int(args.num_comments * pct)
        topics.extend([topic] * count)
    
    # Caso o arredondamento resulte em quantidade menor ou maior que o exato
    while len(topics) < args.num_comments:
        topics.append(random.choice(list(distribution.keys())))
    while len(topics) > args.num_comments:
        topics.pop()
        
    random.shuffle(topics)
    
    generated_data = []
    
    print(f"Iniciando a geração de comentários usando o modelo '{args.model}'...")
    
    start_time = time.time()
    
    for i, topic in enumerate(topics):
        print(f"[{i+1:03d}/{args.num_comments:03d}] Gerando sobre '{topic}'...", end=" ", flush=True)
        comment = generate_comment(prompts[topic], model=args.model)
        
        if comment:
            generated_data.append({
                "id": i + 1,
                "topic": topic,
                "text": comment
            })
            print("OK")
        else:
            print("FALHA (abortando o restante para verificar conexão).")
            break
            
    # Salva o arquivo de qualquer forma, até onde parou
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
        
    elapsed = time.time() - start_time
    print(f"\nConcluído! {len(generated_data)} comentários salvos em '{args.output}'. Tempo gasto: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
