import argparse
import random
import json
import urllib.request
import urllib.error
import time

def generate_comment(topic_prompt, model="qwen2.5:14b", url="http://localhost:11434/api/generate", temperature=0.9):
    """
    Chama a API local do Ollama para gerar um texto com base no prompt.
    """
    data = {
        "model": model,
        "prompt": topic_prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
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
    parser.add_argument("--num_comments", type=int, default=50, help="Quantidade total de comentários a serem gerados.")
    parser.add_argument("--output", type=str, default="comentarios.json", help="Arquivo de saída (.json).")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Modelo a ser utilizado (Ollama).")
    
    args = parser.parse_args()
    
    # Distribuição alvo
    distribution = {
        "service_features": 0.95,
        "cancellation": 0.05
    }

    # Prompts de contexto para o modelo gerar a avaliação com o viés correto
    prompts = {
        "service_features": "You are a customer reviewing a service. Write a short comment (1 to 3 sentences) praising the service, talking about the price, or discussing any other characteristic of the service. Just the comment, no introductions.",
        "cancellation": "You are a frustrated customer. Write a short comment (1 to 3 sentences) complaining EXCLUSIVELY about the CANCELLATION process (difficulty canceling the service, unhelpful support). Just the comment, no introductions."
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
    
    styles = [
        "Use modern internet slang.", 
        "Write extremely formally.", 
        "Write in lowercase without punctuation.", 
        "Be super direct.", 
        "Sound like a confused elderly person with technology.", 
        "Sound like an impatient teenager.", 
        "Be dramatic and exaggerated."
    ]
    
    services = [
        "a streaming subscription", "an internet provider", "a software subscription", 
        "a web hosting service", "a VPN service"
    ]
    
    for i, topic in enumerate(topics):
        print(f"[{i+1:03d}/{args.num_comments:03d}] Gerando sobre '{topic}'...", end=" ", flush=True)
        
        base_prompt = prompts[topic]
        style = random.choice(styles)
        
        if topic != "off_topic":
            service = random.choice(services)
            final_prompt = f"{base_prompt} Target service: {service}. Required writing style: {style}. CRITICAL INSTRUCTION: ANSWER SOLELY AND EXCLUSIVELY IN ENGLISH. DO NOT USE CHINESE, PORTUGUESE OR ANY OTHER LANGUAGE. BE HIGHLY CREATIVE AND UNIQUE."
        else:
            final_prompt = f"{base_prompt} Required writing style: {style}. CRITICAL INSTRUCTION: ANSWER SOLELY AND EXCLUSIVELY IN ENGLISH. DO NOT USE CHINESE, PORTUGUESE OR ANY OTHER LANGUAGE. BE HIGHLY CREATIVE AND UNIQUE."
            
        comment = generate_comment(final_prompt, model=args.model, temperature=0.85)
        
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
