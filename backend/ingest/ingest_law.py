import os
import re
import psycopg2
from langchain_community.document_loaders import PyPDFLoader
# On utilise RecursiveCharacterTextSplitter qui est plus robuste
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DB_HOST = os.getenv("POSTGRES_HOST", "192.168.1.3") 

DB_CONFIG = {
    "dbname": "legal_ai",
    "user": "legal_user",
    "password": "legal_pass_dev",
    "host": DB_HOST,
    "port": "5432"
}

SOURCE_FILE = "backend/data/code_consommation.pdf"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

def ingest_with_langchain():
    print(f"🚀 Démarrage de l'ingestion vers {DB_HOST}...")

    # 1. Vérification du fichier
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ ERREUR: Le fichier {SOURCE_FILE} est introuvable.")
        return

    # 2. Chargement du PDF
    print(f"📂 Chargement du PDF : {SOURCE_FILE}")
    try:
        loader = PyPDFLoader(SOURCE_FILE)
        pages = loader.load()
        print(f"   📖 {len(pages)} pages lues.")
        
        # Fusion du texte pour ne pas couper un article au milieu d'une phrase entre deux pages
        full_text = "\n".join([p.page_content for p in pages])
        print(f"   📝 Taille totale : {len(full_text)} caractères.")
        
    except Exception as e:
        print(f"❌ Erreur lecture PDF : {e}")
        return

    # 3. Découpage (CORRECTION ICI)
    print("✂️ Découpage des articles...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Note le 's' à separators et c'est une liste
        separators=[r"(?=\nArticle [L|R])"], 
        chunk_size=4000,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=True # INDISPENSABLE pour que le regex fonctionne
    )
    
    # CORRECTION ICI : on passe une liste contenant le texte complet à create_documents
    split_docs = text_splitter.create_documents([full_text])
    print(f"✅ {len(split_docs)} articles identifiés.")

    if len(split_docs) < 2:
        print("⚠️ Attention : Peu d'articles trouvés. Vérifie que le PDF contient bien du texte sélectionnable.")

    # 4. Chargement du Modèle (Mac)
    print(f"🧠 Chargement du modèle {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")

    # 5. Connexion au Pi
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
    except Exception as e:
        print(f"❌ Connexion au Pi impossible ({DB_HOST}) : {e}")
        return

    print("🧹 Nettoyage de la table existante...")
    cur.execute("TRUNCATE TABLE legal_articles;")
    
    count = 0
    print("🌊 Envoi des données vers le Pi...")
    
    for doc in split_docs:
        content = doc.page_content.strip()
        
        if len(content) < 50: continue

        # Extraction du numéro d'article
        # On cherche dans les 200 premiers caractères du chunk
        header = content[:200]
        
        # Regex amélioré pour capturer "Article L. 123" ou "Article L123"
        match = re.search(r"Article\s+([L|R]\.?\s*\d+[-]\d+[a-zA-Z]?)", header, re.IGNORECASE)
        
        if match:
            # Nettoyage : "L. 123-1" -> "L123-1"
            article_number = match.group(1).replace(" ", "").replace(".", "")
        else:
            # Si on ne trouve pas de numéro au début, c'est peut-être un morceau de texte orphelin
            # On peut soit l'ignorer, soit le marquer. Ici on l'ignore pour la propreté.
            continue

        vector = model.encode(content).tolist()
        metadata = '{"source": "Code Consommation PDF", "type": "loi"}'

        sql = """
            INSERT INTO legal_articles 
            (article_number, content, metadata, embedding, content_search, code_source)
            VALUES (%s, %s, %s, %s, to_tsvector('french', %s), %s);
        """
        cur.execute(sql, (
            article_number, content, metadata, vector, content, "Code Consommation"
        ))
        count += 1
        if count % 10 == 0:
            print(f"   💾 {count} articles insérés...", end='\r')

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n🎉 SUCCÈS ! {count} articles ingérés sur le Raspberry Pi.")

if __name__ == "__main__":
    ingest_with_langchain()