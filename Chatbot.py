def run_chatbot(qa_chain):
    print("📜 Declaration of Independence RAG Chatbot ready. Ask anything about the PDF (type 'exit' to quit).")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

        print(f"\n🤖 Answer: {answer}\n")
        # Use a set to get unique sources
        unique_sources = set()
        for src in sources:
            source = src.metadata.get('source', 'Unknown Source')
            if source not in unique_sources:
                unique_sources.add(source)
                
        if unique_sources:
            print("📄 Sources:")
            for source in sorted(unique_sources):
                print(f"- {source}")
        print("\n" + "="*50 + "\n")
