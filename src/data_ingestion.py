import polars as pl
import os

def load_and_clean_data(raw_dir: str, processed_dir: str):
    """
    Loads raw Elliptic dataset CSVs, cleans them, and prepares nodes and relationships for Neo4j.
    """
    print(f"Buscando datos en {raw_dir}...")
    
    classes_path = os.path.join(raw_dir, "elliptic_txs_classes.csv")
    features_path = os.path.join(raw_dir, "elliptic_txs_features.csv")
    edgelist_path = os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    
    if not all(os.path.exists(p) for p in [classes_path, features_path, edgelist_path]):
        print("❌ Error: Faltan archivos CSV en el directorio 'data/raw'.")
        print("Asegúrate de haber descargado el 'Elliptic Data Set' de Kaggle y descomprimido los tres archivos CSV en la carpeta 'data/raw'.")
        return False
        
    print("✅ Archivos encontrados. Procesando Nodos (Transacciones)...")
    
    # 1. Procesar Nodos (Clases + Features)
    # Las features no tienen cabecera. La col 0 es txId, col 1 is time_step, cols 2-166 son features
    col_names = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
    
    df_features = pl.read_csv(features_path, has_header=False, new_columns=col_names)
    df_classes = pl.read_csv(classes_path)
    
    # Unir clases y features por txId
    df_nodes = df_features.join(df_classes, on="txId", how="left")
    
    # Mapear las clases a un formato más legible
    # 1 = Illicit, 2 = Licit, unknown = Unknown
    class_mapping = {"1": "Illicit", "2": "Licit", "unknown": "Unknown"}
    df_nodes = df_nodes.with_columns(
        pl.col("class").replace(class_mapping, default="Unknown").alias("label")
    )
    
    # Rellenar nulos hipotéticos
    df_nodes = df_nodes.fill_null(0.0)
    
    nodes_out_path = os.path.join(processed_dir, "nodes.csv")
    df_nodes.write_csv(nodes_out_path)
    print(f"✅ Nodos procesados y guardados en {nodes_out_path} ({len(df_nodes)} registros).")
    
    # 2. Procesar Aristas (Relaciones)
    print("⏳ Procesando Aristas (Relaciones)...")
    df_edges = pl.read_csv(edgelist_path)
    # Las columnas son txId1, txId2. Neo4j prefiere :START_ID y :END_ID, 
    # pero podemos renombrarlo ahora o en el Cypher load. Lo renombramos para ser claros.
    df_edges = df_edges.rename({"txId1": "source", "txId2": "target"})
    
    edges_out_path = os.path.join(processed_dir, "relationships.csv")
    df_edges.write_csv(edges_out_path)
    print(f"✅ Aristas procesadas y guardadas en {edges_out_path} ({len(df_edges)} relaciones).")
    
    return True

if __name__ == "__main__":
    raw_directory = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    processed_directory = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    
    # Crear carpetas si no existen por alguna razón
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(processed_directory, exist_ok=True)
    
    success = load_and_clean_data(raw_directory, processed_directory)
    if success:
         print("🎉 ¡Ingesta de datos finalizada con éxito!")
