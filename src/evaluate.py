import json
import time
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.rag import retriever, answer_chain, format_docs

load_dotenv()

# 1. Pydantic Schemas for LLM-as-a-Judge
class FaihfulnessJudge(BaseModel):
    is_faithful: bool = Field(description= "True if the answer contains ONLY facts supported by context. False if it introduces external or unmentioned claims.")
    reason: str = Field(description="One-sentence explanation of judgment.")

class CorrectnessJudge(BaseModel):
    is_correct: bool = Field(description="True if the answer accurately conveys the ground truth facts. False otherwise.")
    reason: str = Field(description="One-sentence explanation of judgment.")  

# 2. Judge Prompts & Chains
judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

faithfulness_prompt = ChatPromptTemplate.from_template("""You are an impartial evaluator for an academic RAG system.
Evaluate whether the student tutor's generated answer is FAITHFUL to the provided context.
An answer is faithful if ALL statements in the answer are directly supported by the context.
If the answer makes claims not present in the context (hallucination), it is NOT faithful.
If the context says the information cannot be found and the tutor correctly says so, that is faithful.
Context:
{context}
Generated Answer:
{answer}
""")

faithfulness_chain = faithfulness_prompt | judge_llm.with_structured_output(FaihfulnessJudge)

correctness_prompt = ChatPromptTemplate.from_template("""You are an impartial academic evaluator.
Compare the generated answer to the ground truth reference answer.
Evaluate whether the generated answer captures the essential facts present in the ground truth.
Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}
""")

correctness_chain = correctness_prompt | judge_llm.with_structured_output(CorrectnessJudge)

# 3. Main Evaluation Function
def run_evaluation(dataset_path: Path, max_samples: int | None = None):
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if max_samples:
        cases = cases[:max_samples]
    
    print(f"\n Starting RAG Evaluation Bechmark across {len(cases)} test cases...\n")

    results = []

    # Runnig counters
    total_answerable = 0
    recall_hits = 0
    citation_hits = 0
    faithful_hits = 0
    correct_hits = 0

    total_unanswerable = 0
    refusal_hits = 0

    for i, case in enumerate(cases, 1):
        q_id = case["id"]
        question = case["question"]
        is_answerable = case["is_answerable"]

        print(f"[{i:02d}/{len(cases):02d}] Evaluating {q_id}: '{question}'")

        # Retrieve
        docs = retriever.invoke(question)
        context = format_docs(docs)
        retrieved_pages = [d.metadata.get("page_number") for d in docs]

        # Generate Answer
        response = answer_chain.invoke({
            "context": context,
            "chat_history": [],
            "question": question
        })

        # Evaluate based on category
        if is_answerable:
            total_answerable += 1

            # 1. Recall@K (Did retriever pull any expected page?)
            recall_hit = any(p in retrieved_pages for p in case["expected_pages"])
            if recall_hit:
                recall_hits += 1

            # 2. Citation Precision (Did model cite any expected page?)
            citation_hit = any(p in case["expected_pages"] for p in response.page_numbers)
            if citation_hit:
                citation_hits += 1

            # 3. Faithfulness (LLM Judge)
            faith_eval = faithfulness_chain.invoke({"context": context, "answer": response.answer})
            if faith_eval.is_faithful:
                faithful_hits += 1

            # 4. Correctness (LLM Judge)
            corr_eval = correctness_chain.invoke({
                "question": question,
                "ground_truth": case["ground_truth"],
                "answer": response.answer
            })

            if corr_eval.is_correct:
                correct_hits += 1

            status = f"Recall: {'✅' if recall_hit else '❌'} | Cite: {'✅' if citation_hit else '❌'} | Faithful: {'✅' if faith_eval.is_faithful else '❌'} | Correct: {'✅' if corr_eval.is_correct else '❌'}"
            print(f"     └─ {status}")

            results.append({
                "id": q_id,
                "question": question,
                "recall_hit": recall_hit,
                "citation_hit": citation_hit,
                "is_faithful": faith_eval.is_faithful,
                "is_correct": corr_eval.is_correct,
                "model_answer": response.answer,
                "model_pages": response.page_numbers
            })
        
        else:
            total_unanswerable += 1
            # Negative guardrail: Did it properly refuse?
            refused = (
                "cannot find" in response.answer.lower()
                or response.confidence.lower() == "low"
            )
            if refused:
                refusal_hits += 1

            print(f"     └─ Guardrail Refusal: {'✅ Refused' if refused else '❌ Hallucinated'}")
            results.append({
                "id": q_id,
                "question": question,
                "guardrail_refused": refused,
                "model_answer": response.answer
            })

        # Brief rate limit pause
        time.sleep(2)

    # 4. Compute Aggregate Metrics
    recall_score = (recall_hits / total_answerable) * 100 if total_answerable else 0
    citation_score = (citation_hits / total_answerable) * 100 if total_answerable else 0
    faith_score = (faithful_hits / total_answerable) * 100 if total_answerable else 0
    corr_score = (correct_hits / total_answerable) * 100 if total_answerable else 0
    guard_score = (refusal_hits / total_unanswerable) * 100 if total_unanswerable else 0

    print("\n" + "=" * 55)
    print("📊 BENCHMARK EVALUATION REPORT CARD")
    print("=" * 55)
    print(f"Total Test Cases Evaluated : {len(cases)} ({total_answerable} Q&A, {total_unanswerable} Guardrails)")
    print(f"1. Retrieval Recall@3      : {recall_score:.1f}% ({recall_hits}/{total_answerable})")
    print(f"2. Citation Accuracy       : {citation_score:.1f}% ({citation_hits}/{total_answerable})")
    print(f"3. Faithfulness (Grounded) : {faith_score:.1f}% ({faithful_hits}/{total_answerable})")
    print(f"4. Answer Correctness      : {corr_score:.1f}% ({correct_hits}/{total_answerable})")
    print(f"5. Refusal Guardrail       : {guard_score:.1f}% ({refusal_hits}/{total_unanswerable})")
    print("=" * 55 + "\n")

    # Save to disk
    output_path = Path("data/eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "recall_at_3": recall_score,
                "citation_accuracy": citation_score,
                "faithfulness": faith_score,
                "correctness": corr_score,
                "guardrail_accuracy": guard_score
            },
            "cases": results
        }, f, indent=2)
    print(f"Detailed results saved to: {output_path}")

if __name__ == "__main__":
    dataset_file = Path("data/eval_dataset.json")
    run_evaluation(dataset_file)