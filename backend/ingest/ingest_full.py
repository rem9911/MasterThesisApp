import os
import psycopg2
import json



# Configuration
DB_HOST = os.getenv("POSTGRES_HOST", "localhost") # localhost car on lance le script depuis le mac, pas depuis docker
DB_CONFIG = {
    "dbname": "legal_ai",
    "user": "legal_user",
    "password": "legal_pass_dev",
    "host": DB_HOST,
    "port": "5432"
}
print("import de sentence transformer")
from sentence_transformers import SentenceTransformer
print("import ok")
# Les articles clés pour le E-commerce (Livre II - Code de la Consommation)
DATASET = [
    {
        "article_number": "L217-3",
        "content": "Le vendeur délivre un bien conforme au contrat ainsi qu'aux critères énoncés à l'article L. 217-5. Il répond des défauts de conformité existant au moment de la délivrance du bien au sens de l'article L. 216-1, qui apparaissent dans un délai de deux ans à compter de celle-ci.",
        "metadata": {"theme": "Garantie légale de conformité", "type": "loi"}
    },
    {
        "article_number": "L217-4",
        "content": "Le bien est conforme au contrat s'il correspond à la description, au type, à la quantité et à la qualité, notamment en ce qui concerne la fonctionnalité, la compatibilité, l'interopérabilité, ou toute autre caractéristique prévues au contrat.",
        "metadata": {"theme": "Conformité du bien", "type": "loi"}
    },
    {
        "article_number": "L217-5",
        "content": "En plus de respecter le contrat, le bien doit être propre à l'usage habituellement attendu d'un bien de même type, compte tenu de sa nature. Il doit posséder les qualités que le vendeur a présentées au consommateur sous forme d'échantillon ou de modèle.",
        "metadata": {"theme": "Critères de conformité", "type": "loi"}
    },
    {
        "article_number": "L217-7",
        "content": "Les défauts de conformité qui apparaissent dans un délai de vingt-quatre mois à compter de la délivrance du bien, y compris du bien comportant des éléments numériques, sont, sauf preuve contraire, présumés exister au moment de la délivrance.",
        "metadata": {"theme": "Présomption d'antériorité", "type": "loi"}
    },
    {
        "article_number": "L217-8",
        "content": "En cas de défaut de conformité, le consommateur a droit à la mise en conformité du bien par réparation ou remplacement ou, à défaut, à la réduction du prix ou à la résolution du contrat.",
        "metadata": {"theme": "Droits du consommateur", "type": "loi"}
    },
    {
        "article_number": "L221-18",
        "content": "Le consommateur dispose d'un délai de quatorze jours pour exercer son droit de rétractation d'un contrat conclu à distance, à la suite d'un démarchage téléphonique ou hors établissement, sans avoir à motiver sa décision ni à supporter d'autres coûts.",
        "metadata": {"theme": "Droit de rétractation", "type": "loi"}
    },
    {
        "article_number": "L221-5",
        "content": "Préalablement à la conclusion d'un contrat de vente ou de fourniture de services, le professionnel communique au consommateur, de manière lisible et compréhensible, les informations suivantes : les caractéristiques essentielles du bien ou du service, le prix, la date ou le délai de livraison.",
        "metadata": {"theme": "Information précontractuelle", "type": "loi"}
    },
    {
        "article_number": "L221-24",
        "content": "Lorsque le droit de rétractation est exercé, le professionnel rembourse le consommateur de la totalité des sommes versées, y compris les frais de livraison, sans retard injustifié et au plus tard dans les quatorze jours à compter de la date à laquelle il est informé de la décision du consommateur de se rétracter.",
        "metadata": {"theme": "Remboursement", "type": "loi"}
    },
    {
        "article_number": "L221-28",
        "content": "Le droit de rétractation ne peut être exercé pour les contrats : De fourniture de biens confectionnés selon les spécifications du consommateur ou nettement personnalisés ; De fourniture de biens susceptibles de se détériorer ou de se périmer rapidement.",
        "metadata": {"theme": "Exceptions rétractation", "type": "loi"}
    },
    {
        "article_number": "L216-1",
        "content": "Le professionnel délivre le bien ou fournit le service à la date ou dans le délai indiqué au consommateur, conformément au 3° de l'article L. 111-1, sauf si les parties en ont convenu autrement.",
        "metadata": {"theme": "Livraison", "type": "loi"}
    },
    {
        "article_number": "L216-2",
        "content": "En cas de manquement du professionnel à son obligation de livraison du bien à la date ou à l'expiration du délai prévus, le consommateur peut résoudre le contrat, par lettre recommandée avec demande d'avis de réception ou par un écrit sur un autre support durable.",
        "metadata": {"theme": "Retard livraison", "type": "loi"}
    },
    {
        "article_number": "L216-4",
        "content": "Le professionnel rembourse le consommateur de la totalité des sommes versées, au plus tard dans les quatorze jours suivant la date à laquelle le contrat a été dénoncé.",
        "metadata": {"theme": "Remboursement livraison", "type": "loi"}
    },
    {
        "article_number": "L241-1",
        "content": "Les clauses des contrats conclus entre professionnels et consommateurs, sont abusives lorsque, ayant pour objet ou pour effet de créer, au détriment du non-professionnel ou du consommateur, un déséquilibre significatif entre les droits et obligations des parties au contrat.",
        "metadata": {"theme": "Clauses abusives", "type": "loi"}
    },
    {
        "article_number": "L221-28b",
        "content": "Le droit de rétractation ne peut être exercé pour les contrats : De fourniture de biens confectionnés selon les spécifications du consommateur ou nettement personnalisés (produits sur mesure) ; De fourniture de biens susceptibles de se détériorer ou de se périmer rapidement.",
        "metadata": {"theme": "Exceptions rétractation", "type": "loi"}
    },
]

def ingest():
    print("🚀 Démarrage de l'ingestion massive...")
    
    # 1. Chargement du modèle
    print("⏳ Chargement du modèle BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 2. Nettoyage (Optionnel : on garde les anciens ou on vide tout ?)
    # Pour le test, on vide pour éviter les doublons avec tes anciens tests
    print("🧹 Nettoyage de la table...")
    cur.execute("TRUNCATE TABLE legal_articles;")
    
    print(f"📥 Insertion de {len(DATASET)} articles...")
    
    for doc in DATASET:
        # Vectorisation
        vector = model.encode(doc["content"]).tolist()
        metadata_json = json.dumps(doc["metadata"], ensure_ascii=False)
        # Insertion Hybride (Vecteur + TSVector pour mots-clés)
        sql = """
                INSERT INTO legal_articles 
                (article_number, content, metadata, embedding, content_search, code_source)
                VALUES (%s, %s, %s, %s, to_tsvector('french', %s), %s);
            """
        cur.execute(sql, (
            doc["article_number"],
            doc["content"],
            metadata_json,
            vector,
            doc["content"],
            "Code de la Consommation" # <--- La valeur qui manquait !
        ))
        print(f"   ✅ {doc['article_number']} inséré.")

    conn.commit()
    cur.close()
    conn.close()
    print("🎉 Ingestion terminée avec succès !")

if __name__ == "__main__":
    ingest()