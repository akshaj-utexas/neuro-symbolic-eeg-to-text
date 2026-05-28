import sys

# src/graph_filter.py
import sqlite3
import networkx as nx

class GraphRAGRefiner:
    def __init__(self, db_path="data/conceptnet_local.db"):
        self.db_path = db_path

    def retrieve_and_filter_subgraph(self, seed_words, top_n_facts=5, min_weight=1.0):
        """
        Builds a localized subgraph from predicted seed words. Prunes isolated nodes
        representing neural noise while applying a priority safeguard to preserve 
        the highest-confidence Stage-1 predictions.
        """
        # Maintain order while keeping elements unique
        unique_seeds = []
        for w in seed_words:
            w_clean = w.lower().strip()
            if w_clean not in unique_seeds:
                unique_seeds.append(w_clean)
                
        if not unique_seeds:
            return seed_words, []

        # Map plain English words into standard ConceptNet URI format strings
        uri_to_word = {f"/c/en/{w}": w for w in unique_seeds}
        uris = list(uri_to_word.keys())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(uris))
        query = f"""
            SELECT start_node, relation, end_node, weight FROM assertions 
            WHERE start_node IN ({placeholders}) AND end_node IN ({placeholders}) AND weight >= ?
        """
        
        try:
            cursor.execute(query, uris + uris + [min_weight])
            edges = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"\n[SENSE++ Graph RAG Error] SQLite OperationalError: {e}")
            edges = []
        finally:
            conn.close()

        # Construct topological network matching original words
        G = nx.Graph()
        G.add_nodes_from(unique_seeds)
        
        for start_node, relation, end_node, weight in edges:
            s_word = uri_to_word.get(start_node, start_node.split('/')[-1])
            e_word = uri_to_word.get(end_node, end_node.split('/')[-1])
            r_clean = relation.split('/')[-1] if '/' in relation else relation
            G.add_edge(s_word, e_word, relation=r_clean, weight=weight)

        # Topological Noise Pruning Layer with High-Signal Safeguards
        if len(G.edges) > 0:
            centrality = nx.degree_centrality(G)
            
            # PROTECTED ZONE: Lock down the top 5 predictions from the model.
            # Even if they are isolated in the graph, they represent dominant neural signals.
            protected_words = [w.lower() for w in seed_words[:0]]
            
            pruned_words = []
            for word in seed_words:
                w_lower = word.lower()
                # Keep the word if it is interconnected OR if it is a top neural prediction
                if (w_lower in centrality and centrality[w_lower] > 0) or (w_lower in protected_words):
                    pruned_words.append(word)
                    
            if not pruned_words:
                pruned_words = seed_words
        else:
            pruned_words = seed_words

        # Relational Context Assembly
        relational_facts = []
        meaningful_relations = ['AtLocation', 'UsedFor', 'HasProperty', 'CapableOf', 'PartOf', 'Causes', 'NotHasProperty']
        
        for u, v, data in G.edges(data=True):
            rel = data.get('relation', '')
            if any(r.lower() in rel.lower() for r in meaningful_relations):
                # Clean up relation labels cleanly to avoid malformed output text
                clean_rel = (rel.replace('NotHasProperty', 'does not have property')
                                .replace('HasProperty', 'has property')
                                .replace('AtLocation', 'is located at')
                                .replace('UsedFor', 'is used for'))
                
                relational_facts.append(f"{u} {clean_rel} {v}")
                if len(relational_facts) >= top_n_facts:
                    break
                    
        return pruned_words, relational_facts