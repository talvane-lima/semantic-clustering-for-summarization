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
        "price": 0.70,
        "durability": 0.20,
        "post_sales": 0.10
    }

    # Prompts de contexto para o modelo gerar a avaliação com o viés correto
    prompts = {
        "price": "Você é um cliente avaliando um produto. Escreva um comentário curto (1 a 3 frases) focando EXCLUSIVAMENTE no PREÇO (pode ser sobre custo-benefício, ser muito caro, ou muito barato). Apenas o comentário, sem introduções.",
        "durability": "Você é um cliente avaliando um produto. Escreva um comentário curto (1 a 3 frases) focando EXCLUSIVAMENTE na DURABILIDADE (qualidade do material, se quebrou rápido, se dura muito). Apenas o comentário, sem introduções.",
        "post_sales": "Você é um cliente frustrado. Escreva um comentário curto (1 a 3 frases) reclamando EXCLUSIVAMENTE de problemas com o PÓS-VENDA (atendimento, dificuldade para trocar, suporte ruim). Apenas o comentário, sem introduções."
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
        "Use gírias da internet brasileiras atuais.", 
        "Escreva de forma extremamente formal.", 
        "Escreva em minúsculas sem pontuação.", 
        "Seja super direto e monossilábico.", 
        "Pareça uma pessoa idosa confusa com a tecnologia.", 
        "Pareça um adolescente impaciente.", 
        "Seja dramático e exagerado."
    ]
    
    products = [
        "um smartphone", "um tênis", "uma geladeira", "um livro", 
        "um fone bluetooth", "um carro", "um curso online", 
        "uma camiseta", "um liquidificador", "um notebook", 
        "um sofá", "uma smart TV", "um perfume", "uma mochila"
    ]
    
    for i, topic in enumerate(topics):
        print(f"[{i+1:03d}/{args.num_comments:03d}] Gerando sobre '{topic}'...", end=" ", flush=True)
        
        base_prompt = prompts[topic]
        style = random.choice(styles)
        
        if topic != "off_topic":
            product = random.choice(products)
            final_prompt = f"{base_prompt} Produto alvo: {product}. Estilo de escrita exigido: {style}. INSTRUÇÃO VITAL: RESPONDA ÚNICA E EXCLUSIVAMENTE EM PORTUGUÊS DO BRASIL. NÃO USE CHINÊS, INGLÊS OU OUTRO IDIOMA. SEJA ALTAMENTE CRIATIVO E INÉDITO. USE PALAVRAS DIFERENTES DOS COMENTÁRIOS COMUNS."
        else:
            final_prompt = f"{base_prompt} Estilo de escrita exigido: {style}. INSTRUÇÃO VITAL: RESPONDA ÚNICA E EXCLUSIVAMENTE EM PORTUGUÊS DO BRASIL. NÃO USE CHINÊS, INGLÊS OU OUTRO IDIOMA. SEJA ALTAMENTE CRIATIVO E INÉDITO."
            
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
