import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import networkx as nx
import warnings
import faiss
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict





warnings.filterwarnings("ignore")




def compute_cap_network_adaptive_faiss_hnsw(sampled_df, similarity_threshold=0.2):

    print("🔹 Constructing Adaptive CAP Network with FAISS + NetworkX...")

    # Build User-Cluster Matrix (Sparse)
    user_cluster_counts = sampled_df.groupby(['username', 'cluster']).size().unstack(fill_value=0)
    users = user_cluster_counts.index  # User IDs
    num_users = len(users)

    # set n_neighbors
    n_neighbors = max(10, min(100, num_users // 500))  # works about as good as 100% with major speedup


    user_cluster_matrix = csr_matrix(user_cluster_counts.values)  # sparse format

    # tfidf weighting (Still Sparse)
    tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    cap_matrix_sparse = tfidf_transformer.fit_transform(user_cluster_matrix)

    # faiss stuff
    cap_matrix_dense = cap_matrix_sparse.toarray().astype(np.float32)  
    d = cap_matrix_dense.shape[1]  
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200  
    index.hnsw.efSearch = 128  
    faiss.normalize_L2(cap_matrix_dense)
    index.add(cap_matrix_dense) 
    faiss.normalize_L2(cap_matrix_dense)
    
    # actual search
    distances, indices = index.search(cap_matrix_dense, n_neighbors)

    print("making graph using")

    G = nx.Graph()
    
    for i in tqdm(range(len(users)), desc="adding edges", dynamic_ncols=True):
        user_i = users[i]
        for j_idx in range(n_neighbors):
            j = indices[i][j_idx]  
            if j == i:  #  self-comparisons
                continue

            user_j = users[j]
            cosine_similarity = 1 - distances[i][j_idx]

            if cosine_similarity > similarity_threshold: 
                G.add_edge(user_i, user_j, weight=cosine_similarity)

    print(f" {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    return G



def compute_edge_weight_stats(graph, graph_name="Graph"):
    edge_weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    if len(edge_weights) == 0:
        return None
    stats = {
        "Graph": graph_name,
        "Number of Edges": len(edge_weights),
        "Min Weight": np.min(edge_weights),
        # "Min Weight": np.min(edge_weights),
        "Max Weight": np.max(edge_weights),
        # "Max Weight": np.max(edge_weights),
        "Mean Weight": np.mean(edge_weights),
        "Median Weight": np.median(edge_weights),
        "Standard Deviation": np.std(edge_weights),
        # "Standard Deviation": np.std(edge_weights),
        "10th Percentile": np.percentile(edge_weights, 10),
        "25th Percentile": np.percentile(edge_weights, 25),
        "50th Percentile (Median)": np.percentile(edge_weights, 50),
        "75th Percentile": np.percentile(edge_weights, 75),
        # "85th Percentile": np.percentile(edge_weights, 75),
        "90th Percentile": np.percentile(edge_weights, 90),
        "95th Percentile": np.percentile(edge_weights, 95),
        "99th Percentile": np.percentile(edge_weights, 99),
    }

    return stats




              
            

def print_edge_weight_stats(stats):
    if stats is None:
        return
    for key, value in stats.items():
        if key != "Graph":
            print(f"{key}: {value:.6f}")
            
            




def compute_temporal_cap_neighbors(sampled_df, time_col='time_bin', user_col='url-username', cluster_col='cluster', k=10):
    import faiss
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    import numpy as np
    import pandas as pd

    neighbors_by_time = {}
    time_bins = sorted(sampled_df[time_col].unique())

    for t in tqdm(time_bins, desc="timestps"):
        df_t = sampled_df[sampled_df[time_col] == t]
        user_cluster = df_t.groupby([user_col, cluster_col]).size().unstack(fill_value=0)
        users = user_cluster.index.tolist()
        
        if len(users) < 2:
            continue  # too small



        tfidf = TfidfTransformer()
        # faiss stuff
        cap_matrix = tfidf.fit_transform(csr_matrix(user_cluster.values))
        cap_matrix_dense = cap_matrix.toarray().astype(np.float32)
        faiss.normalize_L2(cap_matrix_dense)
        d = cap_matrix_dense.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(cap_matrix_dense)
        faiss.normalize_L2(cap_matrix_dense)
        distances, indices = index.search(cap_matrix_dense, min(k + 1, len(users)))

        neighbors_by_time[t] = {}
        for i, user in enumerate(users):
            neighbors = []
            for j_idx in range(1, min(k + 1, len(users))):  
                neighbor_idx = indices[i][j_idx]
                similarity = 1 - distances[i][j_idx]
                if similarity > 0:  # threshold
                    neighbors.append((users[neighbor_idx], similarity))
            neighbors_by_time[t][user] = neighbors

    return neighbors_by_time



def build_temporal_cap_graph(aggregated_scores, similarity_threshold=0.2):

    G = nx.Graph()
    for u in tqdm(aggregated_scores, desc="building graph"):
        for v, sim in aggregated_scores[u].items():
            if sim >= similarity_threshold:
                G.add_edge(u, v, weight=sim)
    return G








def aggregate_temporal_neighbors(neighbors_by_time, method='decay', lambda_=0.1):
    aggregated_scores = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))  

    sorted_timesteps = sorted(neighbors_by_time.keys())
    
    for step, t in enumerate(tqdm(sorted_timesteps, desc="aggregate over time")):
        if method == 'decay':
            decay_weight = math.exp(-lambda_ * (len(sorted_timesteps) - 1 - step))
        else:
            decay_weight = 1.0
        for u, neighbors in tqdm(neighbors_by_time[t].items(), desc=f"timestep {t}", leave=False):
            for v, sim in neighbors:
                if u == v:
                    continue
                if method in ['sum', 'decay']:
                    aggregated_scores[u][v] += sim * decay_weight
                elif method == 'average':
                    aggregated_scores[u][v] += sim
                    counts[u][v] += 1

                elif method == 'max':
                    aggregated_scores[u][v] = max(aggregated_scores[u][v], sim)
                elif method == 'stability':
                    counts[u][v] += 1  # just count how often u and v are paired

    if method == 'average':
        for u in tqdm(aggregated_scores, desc="avging scores"):
            for v in aggregated_scores[u]:
                aggregated_scores[u][v] /= counts[u][v]
    elif method == 'stability':
        total_bins = len(sorted_timesteps)
        for u in tqdm(counts, desc="stability scores"):
            for v in counts[u]:
                aggregated_scores[u][v] = counts[u][v] / total_bins

    return aggregated_scores




# NEEDS: CLUSTERED DF

file_path = "/data/pgerard/tenet-media-cross-platform/ts-tk-tw/all_clustered_filtered.parquet"
save_dir = "/data/pgerard/tenet-media-cross-platform/ts-tk-tw/cap-networks-with-telegram/"
combined_df = pd.read_parquet(file_path)
clustered_df = combined_df

clustered_df['timestamp'] = pd.to_datetime(clustered_df['timestamp'])
clustered_df['time_bin'] = clustered_df['timestamp'].dt.to_period('2W').astype(str) 



print('Creating cap graph...')
# cap_graph = compute_cap_network_adaptive(clustered_df)
cap_graph = compute_cap_network_adaptive_faiss_hnsw(clustered_df)
cap_stats = compute_edge_weight_stats(cap_graph, "CAP Edge Weights")
cap_graph_path = os.path.join(save_dir, f"cap-full.gpickle")
print("cap network stats:")
print_edge_weight_stats(cap_stats)

print(f"saving graphs..")
# nx.write_gpickle(cap_graph, cap_graph_path)
import pickle
with open(cap_graph_path, "wb") as f:
        pickle.dump(cap_graph, f)

neighbors_by_time = compute_temporal_cap_neighbors(
    clustered_df,
    time_col='time_bin',
    user_col='username',
    cluster_col='cluster',
    k=10
)
scores = aggregate_temporal_neighbors(neighbors_by_time, method='decay', lambda_=0.2)
print('got scores')
G = build_temporal_cap_graph(scores, similarity_threshold=0.)
cap_stats = compute_edge_weight_stats(G, "t-CAP Edge Weights")

print("\n🔹 CAP Network Statistics:")
print_edge_weight_stats(cap_stats)
cap_graph_path = os.path.join(save_dir, f"t-cap-full-decay_2w.gpickle")
with open(cap_graph_path, "wb") as f:
        pickle.dump(G, f)

    