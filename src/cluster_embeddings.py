import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
import umap
import hdbscan

warnings.filterwarnings('ignore', category=FutureWarning)

def load_data(embedding_dir):
    emb_path = os.path.join(embedding_dir, "embeddings.npy")
    meta_path = os.path.join(embedding_dir, "metadata.csv")
    
    embeddings = np.load(emb_path)
    df = pd.read_csv(meta_path, index_col=0)
    return embeddings, df

def eval_ground_truth(true_labels, pred_labels):
    # Trata ruído (-1) do HDBSCAN como um cluster separado para avaliação limpa
    return {
        "ARI": float(adjusted_rand_score(true_labels, pred_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "Homogeneity": float(homogeneity_score(true_labels, pred_labels)),
        "Completeness": float(completeness_score(true_labels, pred_labels)),
        "V-Measure": float(v_measure_score(true_labels, pred_labels))
    }

def run_kmeans(embeddings, max_k=15, out_dir="plots"):
    print("\n--- Rodando K-Means Sweep ---")
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples // 2)
    
    inertias = []
    silhouettes = []
    chs = []
    dbs = []
    models = {}
    
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(embeddings, labels))
        chs.append(calinski_harabasz_score(embeddings, labels))
        dbs.append(davies_bouldin_score(embeddings, labels))
        models[k] = labels

    best_k = k_range[np.argmax(silhouettes)]
    print(f"Melhor K pelo Silhouette Score: {best_k}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(k_range, inertias, marker='o')
    axes[0, 0].set_title("Elbow Method (Inertia)")
    
    axes[0, 1].plot(k_range, silhouettes, marker='o', color='g')
    axes[0, 1].set_title("Silhouette Score (Maior é melhor)")
    
    axes[1, 0].plot(k_range, chs, marker='o', color='orange')
    axes[1, 0].set_title("Calinski-Harabasz (Maior é melhor)")
    
    axes[1, 1].plot(k_range, dbs, marker='o', color='red')
    axes[1, 1].set_title("Davies-Bouldin (Menor é melhor)")
    
    for ax in axes.flatten():
        ax.set_xlabel("Número de Clusters (K)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kmeans_metrics.png"))
    plt.close()
    
    return models[best_k], best_k

def run_hdbscan(embeddings, out_dir="plots"):
    print("\n--- Rodando HDBSCAN Sweep ---")
    
    n_samples = len(embeddings)
    min_cluster_sizes = [2, 3, 5, 7, 10]
    min_cluster_sizes = [s for s in min_cluster_sizes if s <= n_samples // 3]
    
    best_sil = -1
    best_labels = None
    best_params = {}
    
    results = []
    
    for mcs in min_cluster_sizes:
        for ms in [1, max(1, mcs//2), mcs]:
            if ms > n_samples: continue
            
            for c_method in ['eom', 'leaf']:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs, 
                    min_samples=ms, 
                    metric='euclidean',
                    cluster_selection_method=c_method
                )
                labels = clusterer.fit_predict(embeddings)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                
                # Compute silhouette on non-outliers if there are at least 2 clusters
                valid_idx = labels != -1
                if n_clusters >= 2 and sum(valid_idx) > 2:
                    sil = silhouette_score(embeddings[valid_idx], labels[valid_idx])
                else:
                    sil = -1
                    
                results.append({"mcs": mcs, "ms": ms, "method": c_method, "n_clusters": n_clusters, "outliers": n_outliers, "silhouette": sil})
                
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels
                    best_params = {"min_cluster_size": mcs, "min_samples": ms, "method": c_method}
                
    if best_labels is None: # Fallback if no config produced >=2 clusters
        print("HDBSCAN não encontrou múltiplos clusters bons. Usando fallback.")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        best_labels = clusterer.fit_predict(embeddings)
        best_params = {"min_cluster_size": 2}
        
    print(f"Melhor param: {best_params} | Clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)} | Outliers: {list(best_labels).count(-1)}")
    
    return best_labels, best_params


def run_hierarchical(embeddings, out_dir="plots", target_k=6):
    print("\n--- Rodando Clusterização Hierárquica ---")
    Z = linkage(embeddings, method='ward')
    
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title("Dendrograma Hierárquico (Ward)")
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.savefig(os.path.join(out_dir, "dendrogram.png"))
    plt.close()
    
    clusterer = AgglomerativeClustering(n_clusters=target_k, linkage='ward')
    labels = clusterer.fit_predict(embeddings)
    
    return labels, {"linkage": "ward", "n_clusters": target_k}

def generate_comparative_umap(embeddings, labels_dict, out_dir="plots"):
    print("\n--- Gerando Projeção UMAP Comparativa ---")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    u = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()
    
    for ax, (method, labels) in zip(axes, labels_dict.items()):
        scatter = ax.scatter(u[:, 0], u[:, 1], c=labels, cmap='tab20', s=30, alpha=0.8)
        ax.set_title(method)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        handles, leg_labels = scatter.legend_elements(prop="colors", alpha=0.6)
        
        # Se for HDBSCAN, destacar outliers e mesclar na legenda
        if method == "HDBSCAN" and -1 in labels:
            outliers = (labels == -1)
            ax.scatter(u[outliers, 0], u[outliers, 1], color='black', s=10, label='Outliers', marker='x')
            
            clean_handles, clean_labels = [], []
            for h, l in zip(handles, leg_labels):
                if '$-1$' not in l and '-1' not in l:
                    clean_handles.append(h)
                    clean_labels.append(l)
                    
            outlier_handles, outlier_labels = ax.get_legend_handles_labels()
            handles = clean_handles + outlier_handles
            leg_labels = clean_labels + outlier_labels

        ax.legend(handles, leg_labels, loc="best", title="Clusters", fontsize='x-small', ncol=2)
            
    # Remove eixos vazios se houver menos métodos do que subplots
    for i in range(len(labels_dict), len(axes)):
        fig.delaxes(axes[i])
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_comparative.png"))
    plt.close()

def generate_heatmaps(df_labels, true_topic_col, out_dir="plots"):
    for method in df_labels.columns:
        if method in ['id', 'topic', 'text']: continue
        
        crosstab = pd.crosstab(df_labels[method], df_labels[true_topic_col])
        plt.figure(figsize=(8, 6))
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Heatmap: {method} vs Topico Real")
        plt.ylabel("Cluster ID")
        plt.xlabel("Topic (Ground Truth)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_{method}.png"))
        plt.close()

def main():
    embedding_dir = "outputs/embeddings"
    out_dir = "outputs/clustering"
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Carregar Dados
    embeddings, df = load_data(embedding_dir)
    true_labels = df['topic'].values
    
    print("\n--- Redução de Dimensionalidade (UMAP) ---")
    print(f"Reduzindo de {embeddings.shape[1]} para 15 dimensões para otimizar os agrupamentos...")
    reducer = umap.UMAP(n_components=15, metric='cosine', random_state=42)
    clustering_embeddings = reducer.fit_transform(embeddings)
    
    # Dicionários de resultados
    all_labels = {}
    metrics_log = []
    summary_params = {}
    
    # 2. K-Means
    kmeans_labels, best_k = run_kmeans(clustering_embeddings, max_k=15, out_dir=plots_dir)
    all_labels["KMeans"] = kmeans_labels
    summary_params["KMeans"] = {"best_k": int(best_k)}
    
    # 3. HDBSCAN
    hdb_labels, hdb_params = run_hdbscan(clustering_embeddings, out_dir=plots_dir)
    all_labels["HDBSCAN"] = hdb_labels
    summary_params["HDBSCAN"] = hdb_params
    

    # 5. Hierárquico
    # Usaremos o best_k do k-means como referência heurística para o corte do dendrograma
    hier_labels, hier_params = run_hierarchical(clustering_embeddings, out_dir=plots_dir, target_k=best_k)
    all_labels["Hierarchical"] = hier_labels
    summary_params["Hierarchical"] = hier_params
    
    # 6. Avaliação Externa e UMAP
    for method, labels in all_labels.items():
        metrics = eval_ground_truth(true_labels, labels)
        metrics["Method"] = method
        metrics_log.append(metrics)
        
    generate_comparative_umap(embeddings, all_labels, out_dir=plots_dir)
    
    # 7. Compilar DataFrames
    df_labels = df.copy()
    for method, labels in all_labels.items():
        df_labels[method] = labels
        
    generate_heatmaps(df_labels, "topic", out_dir=plots_dir)
    
    # Representativos (Amotras de cada cluster no KMeans por ser comum)
    representatives = {}
    for c_id in np.unique(kmeans_labels):
        cluster_texts = df_labels[df_labels["KMeans"] == c_id]["text"].head(3).tolist()
        representatives[f"Cluster_{c_id}"] = cluster_texts
        
    summary_params["KMeans_Representatives"] = representatives
    summary_params["Total_Samples"] = len(df)
    
    # 8. Exportar tudo
    df_labels.to_csv(os.path.join(out_dir, "cluster_assignments.csv"), index=False)
    
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics = df_metrics[["Method", "ARI", "NMI", "Homogeneity", "Completeness", "V-Measure"]]
    df_metrics.to_csv(os.path.join(out_dir, "clustering_metrics.csv"), index=False)
    
    with open(os.path.join(out_dir, "clustering_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_params, f, indent=4, ensure_ascii=False)
        
    print("\n--- Clusterização finalizada! ---")
    print("Métricas:")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
