# eval.py
#
# Purpose: measure baseline retrieval + answer quality now, so later stages
# (query rewriting, reranking) can be compared against a real number instead
# of a gut feeling.

# EVAL_SET format (to be created manually, ~15-20 entries), e.g. as JSON:
# [
#   {
#     "question": "...",
#     "expected_answer": "...",           # reference answer, for correctness check
#     "expected_source_chunk_id": "..."    # which chunk SHOULD be retrieved (for hit-rate)
#   },
#   ...
# ]

# def load_eval_set(path: str) -> list[dict]:
#     - read the JSON eval file, return list of eval entries

# def compute_retrieval_hit_rate(eval_set: list[dict], k: int) -> float:
#     - for each eval entry:
#         - run vector_store.search(embed_query(entry["question"]), k)
#         - check if entry["expected_source_chunk_id"] is among the returned chunk_ids
#     - return fraction of entries where the expected chunk was retrieved (hit rate)

# def compute_answer_correctness(eval_set: list[dict]) -> float:
#     - for each eval entry:
#         - run rag_pipeline.rag_query(entry["question"])
#         - compare generated answer to entry["expected_answer"]:
#           either manual scoring, simple string/keyword match, or LLM-as-judge
#           (a second LLM call scoring correctness 0/1 or 0-5)
#     - return aggregate score (accuracy or average score)

# def run_full_eval(eval_set_path: str, k: int):
#     - eval_set = load_eval_set(eval_set_path)
#     - hit_rate = compute_retrieval_hit_rate(eval_set, k)
#     - correctness = compute_answer_correctness(eval_set)
#     - print/log both numbers — this is the "baseline" to beat with
#       query rewriting and reranking later

# if __name__ == "__main__":
#     - run_full_eval(config.EVAL_SET_PATH, config.EVAL_TOP_K)
