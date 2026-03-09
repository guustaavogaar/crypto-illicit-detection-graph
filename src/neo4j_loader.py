from neo4j import GraphDatabase
import time

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        """Creates unique constraints to speed up node creation and querying."""
        print("Creando restricciones (Constraints)...")
        query = "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.txId IS UNIQUE"
        with self.driver.session() as session:
            session.run(query)
        print("✅ Constraints creados.")

    def load_nodes(self):
        """Loads nodes into Neo4j from the nodes CSV file in the import directory."""
        print("Cargando Nodos en Neo4j... (Esto puede tardar unos minutos)")
        query = '''
        LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
        CALL {
            WITH row
            MERGE (t:Transaction {txId: toInteger(toFloat(row.txId))})
            SET t.time_step = toInteger(toFloat(row.time_step)),
                t.label = row.label
        } IN TRANSACTIONS OF 10000 ROWS;
        '''
        start = time.time()
        with self.driver.session() as session:
            session.run(query)
        end = time.time()
        print(f"✅ Nodos cargados en {end - start:.2f} segundos.")

    def load_relationships(self):
        """Loads relationships into Neo4j from the relationships CSV file."""
        print("Cargando Relaciones en Neo4j... (Esto puede tardar unos minutos)")
        query = '''
        LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
        CALL {
            WITH row
            MATCH (source:Transaction {txId: toInteger(toFloat(row.source))})
            MATCH (target:Transaction {txId: toInteger(toFloat(row.target))})
            MERGE (source)-[:SENDS_TO]->(target)
        } IN TRANSACTIONS OF 10000 ROWS;
        '''
        start = time.time()
        with self.driver.session() as session:
            session.run(query)
        end = time.time()
        print(f"✅ Relaciones cargadas en {end - start:.2f} segundos.")

if __name__ == "__main__":
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password"
    
    loader = Neo4jLoader(URI, USER, PASSWORD)
    try:
        loader.setup_constraints()
        # Ensure Neo4j has index ready before inserting
        time.sleep(2)
        loader.load_nodes()
        loader.load_relationships()
        print("🎉 ¡Carga masiva en Neo4j completada con éxito!")
    except Exception as e:
        print(f"❌ Error durante la carga: {e}")
    finally:
        loader.close()
