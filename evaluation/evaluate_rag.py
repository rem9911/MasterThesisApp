"""
RAG Evaluation Script using RAGAS
Compares Naive vs Advanced mode on legal questions
"""
print('debut')
import json
import time
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
print('import de datasets')
from datasets import Dataset
print('import de ragas')
from ragas import evaluate
print('import de metrics')
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def load_test_data(filepath: str):
    """Load test questions from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_rag_pipeline(questions: list, mode: str = "advanced"):
    """Run RAG pipeline on all questions and collect results"""
    print('import de RagEngine')
    from backend.app.rag.rag_engine import RagEngine
    
    rag = RagEngine.get_instance()
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} mode on {len(questions)} questions...")
    print('='*60)
    
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:60]}...")
        
        start = time.time()
        
        # Retrieve sources
        sources = rag.retrieve(q['question'], mode=mode)
        
        # Generate answer
        answer = rag.generate(q['question'], sources)
        
        latency = time.time() - start
        
        # Extract article numbers and content from sources
        retrieved_articles = [s.article_number for s in sources]
        contexts = [s.content for s in sources]
        
        results.append({
            "question": q['question'],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q['reponse_attendue'],
            "expected_article": q['article'],
            "retrieved_articles": retrieved_articles,
            "question_type": q['type_question'],
            "latency_ms": latency * 1000
        })
        
        # Quick feedback
        expected_id = q['article'].replace("Article L. ", "L").replace("Article L.", "L").replace(" ", "")
        hit = any(expected_id in art for art in retrieved_articles)
        print(f"   → Hit: {'✅' if hit else '❌'} | Retrieved: {retrieved_articles[:3]} | {latency*1000:.0f}ms")
    
    return results

def calculate_retrieval_metrics(results: list):
    """Calculate retrieval-specific metrics"""
    hits = 0
    mrr_sum = 0
    
    for r in results:
        # Normalize expected article ID
        expected_id = r['expected_article'].replace("Article L. ", "L").replace("Article L.", "L").replace(" ", "")
        
        # Check if expected article is in retrieved
        retrieved = r['retrieved_articles']
        
        for i, art in enumerate(retrieved):
            if expected_id in art or art in expected_id:
                hits += 1
                mrr_sum += 1 / (i + 1)
                break
    
    n = len(results)
    return {
        "hit_rate": hits / n,
        "mrr": mrr_sum / n,
        "avg_latency_ms": sum(r['latency_ms'] for r in results) / n
    }

def run_ragas_evaluation(results: list):
    """Run RAGAS evaluation on the results"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Prepare dataset for RAGAS
    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    print("\n🔬 Running RAGAS evaluation (this may take a few minutes)...")
    
    # Configure LLM and embeddings explicitly
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Run evaluation with explicit config
    ragas_results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    
    return ragas_results

def main():
    # Load test data
    test_data = load_test_data("data_eval.json")
    
    # Optional: use subset for quick testing
    #test_data = test_data[:5]
    
    print(f"📋 Loaded {len(test_data)} test questions")
    
    # Run both modes
    results_naive = run_rag_pipeline(test_data, mode="naive")
    results_advanced = run_rag_pipeline(test_data, mode="advanced")
    
    # Calculate retrieval metrics
    print("\n" + "="*60)
    print("📊 RETRIEVAL METRICS")
    print("="*60)
    
    metrics_naive = calculate_retrieval_metrics(results_naive)
    metrics_advanced = calculate_retrieval_metrics(results_advanced)
    
    print(f"\n{'Metric':<20} {'Naive':>15} {'Advanced':>15} {'Delta':>15}")
    print("-"*65)
    print(f"{'Hit Rate':<20} {metrics_naive['hit_rate']:>14.1%} {metrics_advanced['hit_rate']:>14.1%} {(metrics_advanced['hit_rate'] - metrics_naive['hit_rate'])*100:>+14.1f}%")
    print(f"{'MRR':<20} {metrics_naive['mrr']:>15.3f} {metrics_advanced['mrr']:>15.3f} {metrics_advanced['mrr'] - metrics_naive['mrr']:>+15.3f}")
    print(f"{'Avg Latency (ms)':<20} {metrics_naive['avg_latency_ms']:>15.0f} {metrics_advanced['avg_latency_ms']:>15.0f} {metrics_advanced['avg_latency_ms'] - metrics_naive['avg_latency_ms']:>+15.0f}")
    
    # Run RAGAS evaluation
    print("\n" + "="*60)
    print("🔬 RAGAS GENERATION METRICS")
    print("="*60)
    
    try:
        ragas_naive = run_ragas_evaluation(results_naive)
        ragas_advanced = run_ragas_evaluation(results_advanced)
        
        print(f"\n{'Metric':<25} {'Naive':>15} {'Advanced':>15}")
        print("-"*55)
        
        # Convert to dict if needed (RAGAS >= 0.4)
        if hasattr(ragas_naive, 'to_pandas'):
            naive_df = ragas_naive.to_pandas()
            adv_df = ragas_advanced.to_pandas()
            # Get mean of each metric
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                if metric in naive_df.columns:
                    naive_val = naive_df[metric].mean()
                    adv_val = adv_df[metric].mean()
                    print(f"{metric:<25} {naive_val:>15.3f} {adv_val:>15.3f}")
        else:
            # Older RAGAS API
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                naive_val = getattr(ragas_naive, metric, 0)
                adv_val = getattr(ragas_advanced, metric, 0)
                print(f"{metric:<25} {naive_val:>15.3f} {adv_val:>15.3f}")
                
        # Save RAGAS results to file
        print("\n📁 Saving RAGAS detailed results...")
        if hasattr(ragas_naive, 'to_pandas'):
            ragas_naive.to_pandas().to_csv("ragas_naive_results.csv", index=False)
            ragas_advanced.to_pandas().to_csv("ragas_advanced_results.csv", index=False)
            print("   → Saved to ragas_naive_results.csv and ragas_advanced_results.csv")
            
    except Exception as e:
        print(f"⚠️ RAGAS evaluation failed: {e}")
        print("Skipping generation metrics...")
    
    # Save detailed results
    output = {
        "naive": {
            "retrieval_metrics": metrics_naive,
            "detailed_results": results_naive
        },
        "advanced": {
            "retrieval_metrics": metrics_advanced,
            "detailed_results": results_advanced
        }
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
