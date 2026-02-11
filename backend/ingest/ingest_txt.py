import os
import re
import psycopg2
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

SOURCE_FILE = "./data/code_consommation2.txt"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

def clean_file_content(text):
    text = text.replace('\x0c', '')
    text = re.sub(
        r"Code de la consommation - Dernière modification.*", 
        "", 
        text, 
        flags=re.IGNORECASE
    )
    return text

def parse_articles_strict(full_text):
    # Regex
    pattern = re.compile(r"(?m)^\s*Article\s+(L|R|D)\.?\s*(\d+(?:-\d+)?(?:-\d+)?[a-zA-Z]*)")
    matches = list(pattern.finditer(full_text))
    articles = []
    print(f"{len(matches)} 'Article' found.")

    for i in range(len(matches)):
        start_index = matches[i].start()
        if i < len(matches) - 1:
            end_index = matches[i+1].start()
        else:
            end_index = len(full_text)
        raw_content = full_text[start_index:end_index].strip()
        
        match = matches[i]
        type_art = match.group(1).upper()
        num_art = match.group(2).replace(" ", "").replace(".", "")
        article_number = f"{type_art}{num_art}"
        articles.append((article_number, raw_content))
        
    return articles

def ingest_strict():
    print(f"Beggining of strict ingestion to {DB_HOST}...")

    if not os.path.exists(SOURCE_FILE):
        print(f"Unaible to find file : {SOURCE_FILE}")
        return

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    text = clean_file_content(text)
    print(f"File size : {len(text)} charactere.")

    print("Surgical chunking...")
    articles_data = parse_articles_strict(text)
    print(f"{len(articles_data)} unique articles extracted")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
    except Exception as e:
        print(f"Pi connection failed : {e}")
        return

    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_articles (
            id bigserial PRIMARY KEY,
            article_number text,
            content text,
            metadata jsonb,
            content_search tsvector,
            code_source text,
            embedding vector(1024)
        );
    """)
    cur.execute("SELECT article_number FROM legal_articles;")
    existing_ids = {row[0] for row in cur.fetchall()}
    articles_to_process = [a for a in articles_data if a[0] not in existing_ids]
    
    if not articles_to_process:
        print("All articles already in DB")
        return

    print(f"Loading of Qwen Model (CPU) to process {len(articles_to_process)} articles...")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")
    
    print("Inserting...")
    count = 0
    BATCH_SIZE = 25  # Commit every 25 articles to save progress
    
    for article_number, content in articles_to_process:
        try:
            task_instruction = f"Instruct: Retrieve relevant legal passages for the following query\nQuery: {content}"
            vector = model.encode(task_instruction).tolist()
            
            sql = """
                INSERT INTO legal_articles 
                (article_number, content, metadata, embedding, content_search, code_source)
                VALUES (%s, %s, %s, %s, to_tsvector('french', %s), %s);
            """
            
            cur.execute(sql, (
                article_number, 
                content, 
                '{"source": "Code Consommation TXT Strict"}', 
                vector, 
                content, 
                "Code Consommation"
            ))
            
            count += 1
            
            if count % BATCH_SIZE == 0:
                conn.commit()
                print(f"{count}/{len(articles_to_process)} articles committed...")
                
        except Exception as e:
            print(f"Error on {article_number}: {e}")
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nFINISHED! {count} articles inserted.")

if __name__ == "__main__":
    ingest_strict()