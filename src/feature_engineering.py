import pandas as pd
import networkx as nx
import community as community_louvain # pip install python-louvain
import os

def load_graph_data():
    print("Loading nodes and relationships from processed CSVs...")
    nodes_df = pd.read_csv("data/processed/nodes.csv")
    rel_df = pd.read_csv("data/processed/relationships.csv")
    return nodes_df, rel_df

def build_network(nodes_df, rel_df):
    print(f"Building NetworkX graph with {len(nodes_df)} nodes and {len(rel_df)} edges...")
    
    # We use DiGraph because transactions are directional
    G = nx.from_pandas_edgelist(
        rel_df, 
        source='source', 
        target='target', 
        create_using=nx.DiGraph()
    )
    
    # Add nodes that might not have any edges
    G.add_nodes_from(nodes_df['txId'].tolist())
    
    print(f"Graph built successfully. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def compute_graph_features(G):
    print("Computing In-Degree Centrality...")
    in_degree = dict(G.in_degree())
    
    print("Computing Out-Degree Centrality...")
    out_degree = dict(G.out_degree())
    
    print("Computing PageRank (this might take a minute)...")
    pagerank = nx.pagerank(G, alpha=0.85)
    
    print("Computing Louvain Communities...")
    # Louvain needs an undirected graph
    undirected_G = G.to_undirected()
    communities = community_louvain.best_partition(undirected_G)
    
    return in_degree, out_degree, pagerank, communities

def build_feature_dataset(nodes_df, in_degree, out_degree, pagerank, communities, output_path):
    print("Merging graph features into tabular dataset for Machine Learning...")
    
    # Map features to the original nodes dataframe
    nodes_df['in_degree'] = nodes_df['txId'].map(in_degree).fillna(0)
    nodes_df['out_degree'] = nodes_df['txId'].map(out_degree).fillna(0)
    nodes_df['pagerank'] = nodes_df['txId'].map(pagerank).fillna(0.0)
    nodes_df['community_id'] = nodes_df['txId'].map(communities).fillna(-1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save target dataset
    nodes_df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully created at {output_path} with {len(nodes_df)} rows and {len(nodes_df.columns)} columns.")
    print(nodes_df.head(3))

if __name__ == "__main__":
    nodes_df, rel_df = load_graph_data()
    
    G = build_network(nodes_df, rel_df)
    
    in_degree, out_degree, pagerank, communities = compute_graph_features(G)
    
    output_path = "data/processed/graph_features.csv"
    build_feature_dataset(nodes_df, in_degree, out_degree, pagerank, communities, output_path)
