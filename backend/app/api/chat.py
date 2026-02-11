import time
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatResponseResult

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    from app.rag.rag_engine import RagEngine
    rag = RagEngine.get_instance()
    
    # --- COMPARISON LOGIC ---
    if request.mode == "compare":
        print("⚔️ COMPARISON Mode activated")
        start_global = time.time()

        # 1. Naive Pipeline
        t0 = time.time()
        docs_naive = rag.retrieve(request.query, mode="naive")
        ans_naive = rag.generate(request.query, docs_naive)
        res_naive = ChatResponseResult(
            answer=ans_naive,
            sources=docs_naive,
            processing_time=time.time() - t0
        )

        # 2. Advanced Pipeline
        t1 = time.time()
        docs_adv = rag.retrieve(request.query, mode="advanced")
        ans_adv = rag.generate(request.query, docs_adv)
        res_adv = ChatResponseResult(
            answer=ans_adv,
            sources=docs_adv,
            processing_time=time.time() - t1
        )

        return ChatResponse(
            comparison={
                "naive": res_naive,
                "advanced": res_adv
            },
            processing_time=time.time() - start_global
        )

    # --- CLASSIC LOGIC (Naive or Advanced) ---
    else:
        start = time.time()
        relevant_docs = rag.retrieve(request.query, mode=request.mode)
        ai_answer = rag.generate(request.query, relevant_docs)
        
        return ChatResponse(
            answer=ai_answer, 
            sources=relevant_docs,
            processing_time=time.time() - start
        )
