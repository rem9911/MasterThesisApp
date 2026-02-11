import psycopg2
import os

DB_CONFIG = {
    "dbname": "legal_ai",
    "user": "legal_user",
    "password": "legal_pass_dev",
    "host": "localhost",
    "port": "5432"
}

def check_connection():
    try:
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("'pgvector' extension  detected.")
        else:
            print("ERROR: 'pgvector' not installed !")

        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'legal_articles';
        """)
        columns = cur.fetchall()
        if columns:
            print(f"Table 'legal_articles' found with {len(columns)} columns.")
        else:
            print("ERROR: Table 'legal_articles' doesn't exist !")

        print("Test inserting fake data...")
        cur.execute("""
            INSERT INTO legal_articles (code_source, article_number, content, metadata)
            VALUES ('TEST_CODE', 'TEST_001', 'Ceci est un test.', '{"test": true}')
            ON CONFLICT (code_source, article_number) DO NOTHING;
        """)
        conn.commit()
        print("Success !")

        cur.close()
        conn.close()
        print("\n Infrastructure is ready")

    except Exception as e:
        print(f"\n Connection failed : {e}")

if __name__ == "__main__":
    check_connection()