# backend/app/rag/rag_engine.py
import os
import time
import math
import re
import logging
from typing import List, Optional
import psycopg2
from psycopg2 import pool
from app.models.schemas import Source

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB Configuration
DB_HOST = os.getenv("POSTGRES_HOST", "legal-ai-db")
DB_USER = os.getenv("POSTGRES_USER", "legal_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "legal_pass_dev")
DB_NAME = os.getenv("POSTGRES_DB", "legal_ai")

DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": "5432"
}

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKING_MODEL = "BAAI/bge-reranker-base"

class RagEngine:
    _instance = None
    _embedder = None
    _reranker = None
    _openai = None
    _db_pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RagEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """One-time initialization (Singleton)"""
        logger.info("RagEngine Initialization (Singleton)...")
        self._init_db_pool()

    def _init_db_pool(self):
        if self._db_pool is None:
            try:
                self._db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1, maxconn=10, **DB_CONFIG
                )
                logger.info("PostgreSQL connection pool initialized.")
            except Exception as e:
                logger.error(f"DB pool initialization error: {e}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RagEngine()
        return cls._instance

    def get_db_connection(self):
        if self._db_pool:
            return self._db_pool.getconn()
        else:
            self._init_db_pool()
            return self._db_pool.getconn()

    def release_db_connection(self, conn):
        if self._db_pool and conn:
            self._db_pool.putconn(conn)

    # --- LAZY LOADING MODELS ---
    
    @property
    def embedder(self):
        if self._embedder is None:
            logger.info("Loading Embedder (Qwen3-Embedding)...")
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            logger.info("Loading Reranker (BGE-Reranker)...")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKING_MODEL)
        return self._reranker

    @property
    def openai_client(self):
        if self._openai is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            self._openai = OpenAI(api_key=api_key)
        return self._openai

    # --- UTILITIES ---

    def _extract_article_id(self, query: str) -> Optional[str]:
        """
        Detects and normalizes an article ID.
        Ex: "l'article L. 217-3" -> "L217-3"
        Ex: "L 221-28b" -> "L221-28b"
        """
        match = re.search(r"[l|L]\s*\.?\s*(\d+[-]\d+[a-zA-Z]?)", query)
        if match:
            clean_id = f"L{match.group(1)}"
            logger.info(f"Article detected: {clean_id}")
            return clean_id
        return None

    # --- SEARCH METHODS ---

    def _vector_search(self, query_vector, limit=10) -> List[Source]:
        """Pure semantic search (PGVector)"""
        results = []
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            sql = """
                SELECT article_number, content, metadata, 1 - (embedding <=> %s::vector) as score
                FROM legal_articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(sql, (query_vector, query_vector, limit))
            for row in cur.fetchall():
                meta = row[2] if row[2] is not None else {}
                score_val = float(row[3]) if row[3] is not None else 0.0
                results.append(Source(article_number=row[0], content=row[1], metadata=meta, score=score_val))
            cur.close()
        except Exception as e:
            logger.error(f"Vector Search Error: {e}")
        finally:
            self.release_db_connection(conn)
        return results

    def _keyword_search(self, query_text, limit=10) -> List[Source]:
        """Keyword search (Postgres TSVector + Article Number)"""
        results = []
        conn = None
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            extracted_id = self._extract_article_id(query_text)
            
            if extracted_id:
                search_query = "plainto_tsquery" 
                sql_param = extracted_id
            else:
                search_query = "websearch_to_tsquery"
                sql_param = query_text

            sql = f"""
                SELECT article_number, content, metadata, 
                       ts_rank_cd(content_search, {search_query}('french', %s)) as score
                FROM legal_articles
                WHERE content_search @@ {search_query}('french', %s)
                   OR article_number ILIKE %s
                ORDER BY score DESC
                LIMIT %s;
            """
            
            like_query = f"%{extracted_id if extracted_id else query_text.strip()}%"
            cur.execute(sql, (sql_param, sql_param, like_query, limit))
            
            for row in cur.fetchall():
                meta = row[2] if row[2] is not None else {}
                raw_score = float(row[3]) if row[3] is not None else 0.0
                
                if extracted_id and extracted_id.lower() == row[0].lower():
                     logger.info(f"Exact match boost : {row[0]}")
                     raw_score += 50.0 

                results.append(Source(
                    article_number=row[0], content=row[1], metadata=meta, score=raw_score
                ))
            cur.close()
        except Exception as e:
            logger.error(f"Keyword Search Error: {e}")
        finally:
            self.release_db_connection(conn)
        return results

    def _rerank(self, query: str, sources: List[Source], top_k=5) -> List[Source]:
        """Reorders candidates based on semantic relevance to the query."""
        if not sources: 
            return []
        
        pairs = [[query, doc.content] for doc in sources]
        raw_scores = self.reranker.predict(pairs)
        
        logger.info(f"Reranker raw scores: min={min(raw_scores):.2f}, max={max(raw_scores):.2f}, mean={sum(raw_scores)/len(raw_scores):.2f}")
        
        for i, doc in enumerate(sources):
            if doc.score > 20.0: 
                doc.score = 0.9999
            else:
                raw = raw_scores[i]
                doc.score = float(1 / (1 + math.exp(-raw)))
        
        ranked_sources = sorted(sources, key=lambda x: x.score, reverse=True)

        for i, doc in enumerate(ranked_sources[:3]):
            logger.info(f"   #{i+1}: {doc.article_number} (score: {doc.score:.2%}), raw: {raw_scores[i]:.2f}")
        
        return ranked_sources[:top_k]

    # --- MAIN ENTRY POINT ---

    def retrieve(self, query: str, mode: str = "advanced") -> List[Source]:
        logger.info(f"🔎 Search mode: {mode.upper()}")
        
        try:
            # 1. Vector Search
            query_vector = self.embedder.encode(query).tolist()
            
            if mode == "naive":
                return self._vector_search(query_vector, limit=3)
            
            elif mode == "advanced":
                # 1. Hybrid Retrieval
                vector_docs = self._vector_search(query_vector, limit=25)
                keyword_docs = self._keyword_search(query, limit=25)
                
                logger.info(f"Vector docs ({len(vector_docs)}): {[f'{d.article_number}({d.score:.2f})' for d in vector_docs[:10]]}...")
                logger.info(f"Keyword docs ({len(keyword_docs)}): {[f'{d.article_number}({d.score:.2f})' for d in keyword_docs[:10]]}...")
                
                # 2. Deduplication
                all_docs_map = {doc.article_number: doc for doc in vector_docs + keyword_docs}
                unique_docs = list(all_docs_map.values())
                
                logger.info(f"After fusion: {len(unique_docs)} uniques")
                
                # 3. Reranking
                final_docs = self._rerank(query, unique_docs, top_k=5)
                if final_docs:
                     logger.info(f"Top result: {final_docs[0].article_number} (score: {final_docs[0].score:.2%})")
                return final_docs
                
        except Exception as e:
            logger.error(f"Global retrieve error: {e}")
            return []

        return []

    def generate(self, query: str, sources: List[Source]) -> str:
        if not sources:
            return "Désolé, je n'ai trouvé aucun article juridique correspondant à votre recherche."

        context_text = "\n\n".join([f"--- ARTICLE {s.article_number} ---\n{s.content}" for s in sources])
        article_numbers = [s.article_number for s in sources]
        
        system_prompt = f"""Tu es un assistant juridique expert en droit de la consommation français.

            RÈGLES IMPORTANTES:
            1. Réponds TOUJOURS en français, de manière claire et pédagogique.
            2. Base ta réponse UNIQUEMENT sur les articles fournis ci-dessous.
            3. Cite EXPLICITEMENT les numéros d'articles pertinents dans ta réponse (ex: "Selon l'article L221-18...").
            4. Tu peux UNIQUEMENT citer ces articles: {', '.join(article_numbers)}
            5. Si les articles fournis permettent de répondre même partiellement, donne la meilleure réponse possible.
            6. Ne dis "je ne sais pas" que si AUCUN des articles n'est pertinent.
            7. Sois concis mais complet."""
            

        user_message = f"ARTICLES JURIDIQUES DISPONIBLES:\n{context_text}\n\nQUESTION DE L'UTILISATEUR:\n{query}"
        
        try:
            logger.info(f"Generating response with {len(sources)} sources: {article_numbers}")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                temperature=0.3  
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI Generation Error: {e}")
            return f"An error occurred during response generation. ({e})"