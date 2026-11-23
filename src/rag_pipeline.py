
from data_preprocessing.query import (
    ece_collection,
    legal_collection,
    retrieve_documents
)
from LLM.model_loader import generate_response


def build_rag_prompt(question, retrieved_documents):
    """
    Build a prompt containing the question and retrieved evidence.
    """

    if not retrieved_documents:
        evidence = "No relevant evidence was retrieved."
    else:
        evidence_parts = []

        for i, document in enumerate(retrieved_documents, start=1):
            evidence_parts.append(
                f"""Evidence {i}
Source: {document['source']}
Page: {document['page']}
Content:
{document['text']}"""
            )

        evidence = "\n\n".join(evidence_parts)

    prompt = f"""You are a precise evidence-based question-answering system.

Your task is to answer the question using ONLY the supplied evidence.

STRICT RULES:

1. Use only information explicitly stated in the evidence.
2. Never use outside knowledge or assumptions.
3. Copy numerical values and units accurately.
4. Carefully distinguish between:
   - Total counts
   - Counts per category
   - Counts per group or orientation
5. Never calculate or invent values unless explicitly required
   and supported by the evidence.
6. Do not invent specimen names, orientation labels, or details.
7. If the evidence does not explicitly support the answer,
   respond exactly:
   "The provided evidence is insufficient to answer this question."
8. Give a short, direct answer. Avoid unnecessary explanations.

Evidence:
{evidence}

Question:
{question}

Answer:"""

    return prompt


def answer_question(question, collection, top_k=5):
    print("\nRetrieving relevant evidence...")

    retrieved_documents = retrieve_documents(
        collection,
        question,
        top_k=top_k
    )

    # Keep the remaining function unchanged, as it already contains the logic for generating the answer and returning the sources.

    for i, document in enumerate(retrieved_documents, start=1):
        print(f"\n--- Retrieved Chunk {i} ---")
        print(f"Source: {document['source']}")
        print(f"Page: {document['page']}")
        print(document["text"])

    print(f"Retrieved {len(retrieved_documents)} chunks.")

    if not retrieved_documents:
        return {
            "question": question,
            "answer": "The provided evidence is insufficient to answer this question.",
            "sources": []
        }

    prompt = build_rag_prompt(question, retrieved_documents)

    print("\nGenerating answer using Qwen...")

    answer = generate_response(prompt)

    sources = [
        {
            "source": document["source"],
            "page": document["page"],
            "chunk_index": document["chunk_index"],
            "distance": document["distance"]
        }
        for document in retrieved_documents
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


def main():

    print("=" * 70)
    print("DOCUMENT RAG QUESTION ANSWERING")
    print("=" * 70)

    print("\nChoose your dataset:")
    print("1. Legal")
    print("2. Electronics")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        collection = legal_collection
        print("\nSelected: Legal Data")

    elif choice == "2":
        collection = ece_collection
        print("\nSelected: Electronics Data")

    else:
        print("Invalid choice.")
        return

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    result = answer_question(question, collection)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    if not result["sources"]:
        print("No sources available.")
    else:
        for i, source in enumerate(result["sources"], start=1):
            print(
                f"{i}. {source['source']} | "
                f"Page: {source['page']} | "
                f"Distance: {source['distance']:.4f}"
            )


if __name__ == "__main__":
    main()