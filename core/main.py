# agent_vector_metadata.py

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

class DocAgent:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.DATA_DIR = Path(__file__).parent.parent / "data"
        self.vectorstore_path = "faiss_index"
        self.metadata_path = "vector_metadata.json"

        self.embedding = OpenAIEmbeddings(
            openai_api_key=self.OPENAI_API_KEY, model="text-embedding-ada-002"
        )
        self.llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.metadata_list = []

        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        tools = [
            Tool(
                name="Summarize",
                func=lambda text: self._summarize_tool(text),
                description="Summarize a long text. Input: string."
            ),
            Tool(
                name="Star",
                func=self._star_tool,
                description="Add **** at start and end of answer. Input: string."
            )
        ]
        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

    def _summarize_tool(self, text):
        chain = load_summarize_chain(self.llm, chain_type="stuff")
        docs = [Document(page_content=text)]
        return chain.run(docs)

    def _star_tool(self, answer):
        return f"****{answer}****"

    ##### Vectorstore and metadata #####
    def _split_into_chunks_with_metadata(self, text, source):
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        current_title = None
        for line in lines:
            if re.match(r'^[A-Z][A-Z\s\d\-]+$', line.strip()) or re.match(r'^\d+\.', line.strip()) or line.strip().endswith(":"):
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append({"title": current_title or "Untitled", "content": chunk_text})
                    current_chunk = []
                current_title = line.strip()
            else:
                current_chunk.append(line)
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append({"title": current_title or "Untitled", "content": chunk_text})
        return [
            Document(
                page_content=chunk["content"],
                metadata={"source": source, "title": chunk["title"]}
            )
            for chunk in chunks if chunk["content"]
        ]

    def prepare_context(self):
        # Load and chunk documents only if vectorstore/metadata do not exist
        docs = []
        if not (os.path.exists(self.vectorstore_path) and os.path.exists(self.metadata_path)):
            files = list(self.DATA_DIR.glob("*.txt"))
            for file in files:
                with open(file, encoding="utf-8") as f:
                    content = f.read()
                    docs.extend(self._split_into_chunks_with_metadata(content, file.name))
            if not docs:
                print("No documents found in data directory.")
                return None, None
            print(f"Loaded {len(docs)} chunks from {len(files)} files: {[f.name for f in files]}")
            # Build and save vectorstore
            self.vectorstore = FAISS.from_documents(docs, self.embedding)
            self.vectorstore.save_local(self.vectorstore_path)
            # Save metadata
            self.metadata_list = [doc.metadata for doc in docs]
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
            print("Vectorstore and metadata saved.")
        else:
            # Load vectorstore and metadata
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, self.embedding, allow_dangerous_deserialization=True
            )
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
            print("Loaded existing vectorstore and metadata.")
        # Print metadata fields
        all_keys = set()
        for meta in self.metadata_list:
            all_keys.update(meta.keys())
        print("Metadata fields:", all_keys)
        return self.vectorstore, self.metadata_list

    def answer_questions(self, questions):
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            docs_context = self.vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([
                f"[File: {doc.metadata.get('source', 'unknown')}, Title: {doc.metadata.get('title', 'Untitled')}]\n{doc.page_content}"
                for doc in docs_context
            ])
            combined_input = f"""Use the following context to answer the question, and include the source in your answer:

{context}

Question: {question}
"""
            answer = self.agent.run(combined_input)
            self.memory.chat_memory.add_user_message(f"[Context Retrieved]:\n{context}")
            self.memory.chat_memory.add_ai_message(answer)
            # Add sources explicitly to the answer
            sources_used = set(doc.metadata.get("source", "unknown") for doc in docs_context)
            source_str = ", ".join(sorted(sources_used))
            print(f"A{i}: {answer}\n(According to: {source_str})")

if __name__ == "__main__":
    agent_runner = DocAgent()
    agent_runner.prepare_context()
    agent_runner.answer_questions([
        "What can customers do in the restaurant reservation system?",
        "And what about staff members?",
        "List the databases used in both systems and the unique databases of each of them?"
    ])
