import sys
from retriever import retrieve_documents, retrieve_faiss_chunks, format_as_mcp
from generator import generate_response

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run local RAG agent.")
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument("--chunks", type=int, default=0, help="Show top-k retrieved chunks before generation.")
    return parser.parse_args()

def run():
    args = parse_args()
    query = args.query

    print("\n🔍 Stage 1: Coarse document filtering (BM25)...")
    doc_ids = retrieve_documents(query, method="bm25")

    print("\n🎯 Stage 2: Fine-grained chunk retrieval (FAISS)...")
    chunks = retrieve_faiss_chunks(query, doc_ids)

    if args.chunks > 0:
        print(f"\n📦 Top {args.chunks} retrieved chunks:")
        for i, (doc_id, chunk, score) in enumerate(chunks[:args.chunks]):
            print(f"\n--- Chunk {i+1} from {doc_id} (score: {score:.4f}) ---\n{chunk}\n")

    context = format_as_mcp([(doc_id, chunk) for doc_id, chunk, _ in chunks])

    print("\n🧠 Generating response using LM Studio (TinyLlama)...")
    answer = generate_response(query, context)

    print("\n💬 Answer:\n", answer)

if __name__ == "__main__":
    run()
