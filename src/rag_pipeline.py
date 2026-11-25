
import os
from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# 1. GROQ CONFIGURATION
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please configure it in your .env file."
    )

client = None


def configure_groq_client(shared_client):
    """Use a process-wide client supplied by the application runtime."""
    global client
    client = shared_client


def _get_groq_client():
    global client
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    return client

MODEL_NAME = "openai/gpt-oss-120b"


# --------------------------------------------------
# 2. GENERATE RESPONSE USING GROQ
# --------------------------------------------------

def generate_groq_response(system_prompt, user_prompt):

    response = _get_groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=2048
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 3. BUILD RAG PROMPT
# --------------------------------------------------

def build_rag_prompt(question, retrieved_documents):

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

    prompt = f"""Use ONLY the following retrieved evidence to answer
the user's question.

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
8. Give a short, direct answer.
9. Do not use your general knowledge to fill gaps.

RETRIEVED EVIDENCE:

{evidence}

QUESTION:

{question}

ANSWER:"""

    return prompt


# --------------------------------------------------
# 4. RAG MODE - ANSWER USING RETRIEVED DOCUMENTS
# --------------------------------------------------

def answer_question_with_rag(question, collection, top_k=5):

    # Import retrieval functionality only when RAG is selected.
    from data_preprocessing.query import retrieve_documents

    print("\nRetrieving relevant evidence from ChromaDB...")

    retrieved_documents = retrieve_documents(
        collection,
        question,
        top_k=top_k
    )

    print(f"\nRetrieved {len(retrieved_documents)} chunks.")

    if not retrieved_documents:

        return {
            "question": question,
            "answer": (
                "The provided evidence is insufficient to answer this question."
            ),
            "sources": []
        }

    # Display retrieved evidence
    for i, document in enumerate(retrieved_documents, start=1):

        print(f"\n--- Retrieved Chunk {i} ---")
        print(f"Source: {document['source']}")
        print(f"Page: {document['page']}")
        print(f"Distance: {document['distance']:.4f}")
        print(document["text"])

    # Build prompt
    prompt = build_rag_prompt(
        question,
        retrieved_documents
    )

    system_prompt = """You are a precise, evidence-based
question-answering system.

Your responsibility is to answer questions strictly using
the evidence supplied by the user.

Never introduce external knowledge or unsupported assumptions."""

    print("\nGenerating answer using Groq...")

    answer = generate_groq_response(
        system_prompt,
        prompt
    )

    # Prepare source information
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


# --------------------------------------------------
# 5. DIRECT MODE - ANSWER WITHOUT RETRIEVED DOCUMENTS
# --------------------------------------------------

def answer_question_direct(question):

    system_prompt = """You are a helpful and knowledgeable AI assistant.

Answer the user's question using your general knowledge
and reasoning abilities.

Instructions:
1. Provide an accurate and relevant answer.
2. Explain concepts clearly.
3. Use examples wherever helpful.
4. Do not refer to any retrieved documents or claim
   that your answer is based on a document.
5. If you are uncertain about something, acknowledge it.
"""

    print("\nGenerating answer directly using Groq...")
    print("No documents will be retrieved from ChromaDB.")

    answer = generate_groq_response(
        system_prompt,
        question
    )

    return {
        "question": question,
        "answer": answer,
        "sources": []
    }


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 70)
    print("DOCUMENT RAG QUESTION ANSWERING SYSTEM")
    print("=" * 70)

    print("\nChoose how you want to generate your answer:")

    print("\n1. With retrieved documents (RAG)")
    print("2. Without retrieved documents (Direct LLM)")

    mode = input("\nEnter your choice (1/2): ").strip()

    collection = None

    # Select RAG mode
    if mode == "1":

        print("\nChoose your dataset:")
        print("1. Legal")
        print("2. Electronics")

        choice = input("\nEnter your choice (1/2): ").strip()

        if choice == "1":

            from data_preprocessing.query import legal_collection

            collection = legal_collection
            print("\nSelected: Legal Data")

        elif choice == "2":

            from data_preprocessing.query import ece_collection

            collection = ece_collection
            print("\nSelected: Electronics Data")

        else:
            print("\nInvalid dataset choice.")
            return

    elif mode == "2":

        print("\nSelected: Direct LLM Mode")

    else:

        print("\nInvalid mode selection.")
        return

    # Get user question
    question = input("\nEnter your question: ").strip()

    if not question:
        print("\nPlease enter a valid question.")
        return

    # Generate answer
    if mode == "1":

        result = answer_question_with_rag(
            question,
            collection,
            top_k=5
        )

    else:

        result = answer_question_direct(question)

    # Display answer
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    # Display sources
    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    if not result["sources"]:

        if mode == "2":
            print("No documents were used. Answer generated directly by Groq.")
        else:
            print("No sources available.")

    else:

        for i, source in enumerate(result["sources"], start=1):

            print(
                f"{i}. {source['source']} | "
                f"Page: {source['page']} | "
                f"Chunk: {source['chunk_index']} | "
                f"Distance: {source['distance']:.4f}"
            )


if __name__ == "__main__":
    main()
