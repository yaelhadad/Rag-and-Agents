import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import re
import json

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY is not set in environment or .env file!")

# Define data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# --- Tools ---
def summarize_tool(text: str, llm=None) -> str:
    if llm is None:
        raise ValueError("LLM must be provided to summarize_tool.")
    chain = load_summarize_chain(llm, chain_type="stuff")
    docs = [Document(page_content=text)]
    return chain.run(docs)

def star_tool(answer: str) -> str:
    return f"****{answer}****"

# --- Chunking ---
def split_into_chunks_with_metadata(text, source):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_title = None
    for line in lines:
        if re.match(r'^[A-Z][A-Z\s\d\-]+$', line.strip()) or re.match(r'^\d+\.', line.strip()) or line.strip().endswith(":"):
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "title": current_title or "Untitled",
                        "content": chunk_text
                    })
                current_chunk = []
            current_title = line.strip()
        else:
            current_chunk.append(line)
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append({
                "title": current_title or "Untitled",
                "content": chunk_text
            })
    return [
        Document(
            page_content=chunk["content"],
            metadata={"source": source, "title": chunk["title"]}
        )
        for chunk in chunks if chunk["content"]
    ]

# --- Main ---
def main():
    # Load and chunk documents only if vectorstore/metadata do not exist
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    vectorstore_path = "faiss_index"
    metadata_path = "vector_metadata.json"
    docs = []
    if not (os.path.exists(vectorstore_path) and os.path.exists(metadata_path)):
        files = list(DATA_DIR.glob("*.txt"))
        for file in files:
            with open(file, encoding="utf-8") as f:
                content = f.read()
                docs.extend(split_into_chunks_with_metadata(content, file.name))
        if not docs:
            print("No documents found in data directory.")
            return
        print(f"Loaded {len(docs)} chunks from {len(files)} files: {[f.name for f in files]}")
        # Build and save vectorstore
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(vectorstore_path)
        # Save metadata
        metadata_list = [doc.metadata for doc in docs]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        print("Vectorstore and metadata saved.")
    else:
        # Load vectorstore and metadata
        vectorstore = FAISS.load_local(vectorstore_path, embedding, allow_dangerous_deserialization=True)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        print("Loaded existing vectorstore and metadata.")
    # Print metadata fields
    all_keys = set()
    for meta in metadata_list:
        all_keys.update(meta.keys())
    print("Metadata fields:", all_keys)
    # Initialize LLM and memory
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Define tools
    tools = [
        Tool(
            name="Summarize",
            func=lambda text: summarize_tool(text, llm=llm),
            description="Summarize a long text. Input: string."
        ),
        Tool(
            name="Star",
            func=star_tool,
            description="Add **** at start and end of answer. Input: string."
        )
    ]
    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    # Example questions
    questions = [
        "What can customers do in the restaurant reservation system?",
        "And what about staff members?",
        "List the databases used in both systems and the unique databases of each of them?"
    ]
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        # Retrieve context with metadata clearly shown
        docs_context = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([
            f"[File: {doc.metadata.get('source', 'unknown')}, Title: {doc.metadata.get('title', 'Untitled')}]\n{doc.page_content}"
            for doc in docs_context
        ])
        # Add context and question to memory
        combined_input = f"""Use the following context to answer the question:

        {context}

        Question: {question}
        """
        answer = agent.run(combined_input)
        memory.chat_memory.add_user_message(f"[Context Retrieved]:\n{context}")
        
        memory.chat_memory.add_ai_message(answer)
        print(f"A{i}: {answer}")

if __name__ == "__main__":
    main()
