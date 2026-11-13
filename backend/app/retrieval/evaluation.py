
import sys
from app.retrieval.search import search_document


DEFAULT_DOCUMENT_ID = "sample-document-id"

TEST_CASES = [
    {
        "query": "What makes a strong scholarship application?",
        "relevant_pages": {1}
    },
    {
        "query": "What should students include in their scholarship essays?",
        "relevant_pages": {2, 3}
    },
    {
        "query": "What should a scholarship CV contain?",
        "relevant_pages": {5}
    }
]


def evaluate_retrieval(document_id: str = DEFAULT_DOCUMENT_ID, top_k: int = 5):
    total_recall = 0
    total_hits = 0

    for test in TEST_CASES:
        results = search_document(
            document_id,
            test["query"],
            top_k
        )

        retrieved_pages = {
            result["page_number"] for result in results
        }

        relevant_pages = test["relevant_pages"]

        hits = retrieved_pages.intersection(relevant_pages)

        recall = len(hits) / len(relevant_pages)
        hit_at_k = int(bool(hits))

        total_recall += recall
        total_hits += hit_at_k

        print(f"\nQuery: {test['query']}")
        print(f"Expected pages: {sorted(relevant_pages)}")
        print(f"Retrieved pages: {sorted(retrieved_pages)}")
        print(f"Recall@{top_k}: {recall:.2f}")
        print(f"Hit@{top_k}: {hit_at_k}")

    count = len(TEST_CASES)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Average Recall@{top_k}: {total_recall / count:.2f}")
    print(f"Hit Rate@{top_k}: {total_hits / count:.2f}")


if __name__ == "__main__":
    doc_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT_ID
    evaluate_retrieval(document_id=doc_id)
