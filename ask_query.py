from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

def build_rag_graph(collection_name: str):
    class RAGState(TypedDict):
        query: str
        collection_name: str
        docs: list
        answer: str

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    llm_model = ChatOllama(model="llama3:8b")

    # Node 1: Retrieve relevant documents from Chroma
    def retrieve_node(state: RAGState, config) -> RAGState:
        vector_store = Chroma(
            collection_name=state["collection_name"],
            embedding_function=embedding_model,
            persist_directory="./chroma_langchain_db"
        )
        # Perform similarity search directly on the vector store
        docs = vector_store.similarity_search(state["query"], k=5)
        state["docs"] = docs
        return state

    # Node 2: LLM generates answer from retrieved documents
    def llm_node(state: RAGState, config) -> dict:
        context = "\n\n".join([doc.page_content for doc in state["docs"]])
        prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {state['query']}"
        response = llm_model.invoke(prompt)
        return {"answer": response.content if hasattr(response, "content") else str(response)}

    # Build the state graph
    builder = StateGraph(RAGState)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("llm_node", llm_node)
    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "llm_node")

    return builder.compile()
