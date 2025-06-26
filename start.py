from indexer import index_all_documents

def start():
    print("🔧 Starting RAG system...")
    index_all_documents()
    print("🚀 RAG system ready. You may now run main.py for queries.")

if __name__ == "__main__":
    start()
