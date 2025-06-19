from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import os
import logging
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re
from pathlib import Path


# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentAgent:
    def __init__(self, agent_name: str, filepath: str, index_path: Optional[str] = None):
        """Initialize enhanced document agent with automatic metadata management"""
        self.agent_name = agent_name
        self.filepath = Path(filepath)
        self.index_path = Path(index_path or f"index_{agent_name.lower()}")
        
        # Create index directory
        self.index_path.mkdir(exist_ok=True)
        
        # File paths for different components
        self.faiss_index_path = self.index_path / "faiss_index"
        self.metadata_path = self.index_path / "metadata.json"
        self.documents_path = self.index_path / "documents.pkl"
        self.checksum_path = self.index_path / "file_checksum.txt"
        
        # Initialize OpenAI components using configuration
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        
        self.embedding = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Load or create index with automatic rebuild if file changed
        self.vectorstore, self.documents = self._load_or_create_index()
        
        # Setup tools
        self.tools = self._create_tools()
        
        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def _calculate_file_checksum(self) -> str:
        """Calculate MD5 checksum of the source file"""
        hash_md5 = hashlib.md5()
        with open(self.filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _should_rebuild_index(self) -> bool:
        """Check if index should be rebuilt based on file changes"""
        if not all([
            self.faiss_index_path.exists(),
            self.metadata_path.exists(),
            self.documents_path.exists(),
            self.checksum_path.exists()
        ]):
            return True
            
        # Check file checksum
        current_checksum = self._calculate_file_checksum()
        try:
            with open(self.checksum_path, 'r') as f:
                stored_checksum = f.read().strip()
            return current_checksum != stored_checksum
        except:
            return True

    def _load_or_create_index(self) -> Tuple[FAISS, List[Document]]:
        """Load existing index or create new one with automatic rebuild detection"""
        if self._should_rebuild_index():
            logger.info("Building new index...")
            return self._build_new_index()
        else:
            logger.info("Loading existing index...")
            return self._load_existing_index()

    def _build_new_index(self) -> Tuple[FAISS, List[Document]]:
        """Build new index from source file"""
        # Parse documents with enhanced metadata
        documents = self._parse_document_with_metadata()
        
        # Create FAISS index
        vectorstore = FAISS.from_documents(documents, self.embedding)
        
        # Save everything
        self._save_index_components(vectorstore, documents)
        
        return vectorstore, documents

    def _load_existing_index(self) -> Tuple[FAISS, List[Document]]:
        """Load existing index components"""
        try:
            # Load FAISS index
            vectorstore = FAISS.load_local(
                str(self.faiss_index_path), 
                self.embedding,
                allow_dangerous_deserialization=True
            )
            
            # Load documents
            with open(self.documents_path, 'rb') as f:
                documents = pickle.load(f)
                
            logger.info(f"Loaded existing index with {len(documents)} documents")
            return vectorstore, documents
            
        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            logger.info("Rebuilding index...")
            return self._build_new_index()

    def _save_index_components(self, vectorstore: FAISS, documents: List[Document]):
        """Save all index components"""
        try:
            # Save FAISS index
            vectorstore.save_local(str(self.faiss_index_path))
            
            # Save documents as pickle for exact reconstruction
            with open(self.documents_path, 'wb') as f:
                pickle.dump(documents, f)
            
            # Save metadata as JSON for human readability
            metadata = self._create_metadata_summary(documents)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Save file checksum
            checksum = self._calculate_file_checksum()
            with open(self.checksum_path, 'w') as f:
                f.write(checksum)
            
            logger.info(f"Saved index components to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index components: {e}")
            raise

    def _create_metadata_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Create comprehensive metadata summary"""
        # Analyze document structure
        clause_count = len([d for d in documents if d.metadata.get('type') == 'clause'])
        section_count = len([d for d in documents if d.metadata.get('type') == 'section'])
        paragraph_count = len([d for d in documents if d.metadata.get('type') == 'paragraph'])
        
        # Get unique metadata keys
        all_keys = set()
        for doc in documents:
            all_keys.update(doc.metadata.keys())
        
        # Create summary
        return {
            "index_info": {
                "agent_name": self.agent_name,
                "source_file": str(self.filepath),
                "created_at": datetime.now().isoformat(),
                "total_documents": len(documents),
                "file_checksum": self._calculate_file_checksum()
            },
            "document_structure": {
                "clauses": clause_count,
                "sections": section_count,
                "paragraphs": paragraph_count,
                "other": len(documents) - clause_count - section_count - paragraph_count
            },
            "metadata_fields": list(all_keys),
            "sample_documents": [
                {
                    "content_preview": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                }
                for doc in documents[:3]  # First 3 documents as examples
            ]
        }

    def _parse_document_with_metadata(self) -> List[Document]:
        """Parse document with comprehensive automatic metadata extraction"""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents = []
            sections = self._split_into_sections(content)
            
            for section_idx, section in enumerate(sections, 1):
                # Extract section metadata
                metadata = self._extract_section_metadata(section, section_idx)
                
                # Split section into smaller chunks if too long
                chunks = self._split_section_into_chunks(section)
                
                for chunk_idx, chunk in enumerate(chunks, 1):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'chunk_id': f"{section_idx}_{chunk_idx}",
                        'content_length': len(chunk),
                        'word_count': len(chunk.split()),
                        'created_at': datetime.now().isoformat()
                    })
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            
            logger.info(f"Created {len(documents)} documents with enhanced metadata")
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        # Look for common section patterns
        patterns = [
            r'^(Clause\s+\d+.*?)(?=^Clause\s+\d+|\Z)',  # Legal clauses
            r'^(\d+\.\s*.*?)(?=^\d+\.\s*|\Z)',          # Numbered sections
            r'^([A-Z][^.]*?:.*?)(?=^[A-Z][^.]*?:|\Z)',  # Title: content sections
            r'(.{500,2000}?[.!?])\s*(?=[A-Z]|\n\n|\Z)'  # Fallback: sentence groups
        ]
        
        for pattern in patterns:
            sections = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if len(sections) > 1:  # Found meaningful splits
                return [s.strip() for s in sections if s.strip()]
        
        # Fallback: split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [content]

    def _extract_section_metadata(self, section: str, section_idx: int) -> Dict[str, Any]:
        """Extract comprehensive metadata from a section"""
        metadata = {
            'section_index': section_idx,
            'source': str(self.filepath.name),
            'agent': self.agent_name
        }
        
        # Extract section type and number
        first_line = section.split('\n')[0].strip()
        
        if re.match(r'clause\s+\d+', first_line, re.IGNORECASE):
            metadata['type'] = 'clause'
            match = re.search(r'clause\s+(\d+)', first_line, re.IGNORECASE)
            if match:
                metadata['clause_number'] = int(match.group(1))
                metadata['display_title'] = first_line
        elif re.match(r'\d+\.', first_line):
            metadata['type'] = 'section'
            match = re.search(r'(\d+)\.', first_line)
            if match:
                metadata['section_number'] = int(match.group(1))
                metadata['display_title'] = first_line
        elif ':' in first_line:
            metadata['type'] = 'titled_section'
            metadata['title'] = first_line.split(':')[0].strip()
            metadata['display_title'] = metadata['title']
        else:
            metadata['type'] = 'paragraph'
            metadata['display_title'] = first_line[:50] + "..."
        
        # Extract keywords (simple keyword extraction)
        words = re.findall(r'\b[A-Za-z]{4,}\b', section.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top 5 keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        metadata['keywords'] = [word for word, freq in keywords]
        
        # Text statistics
        metadata['sentence_count'] = len(re.findall(r'[.!?]+', section))
        metadata['has_numbers'] = bool(re.search(r'\d', section))
        metadata['has_dates'] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', section))
        
        return metadata

    def _split_section_into_chunks(self, section: str, max_chunk_size: int = 1000) -> List[str]:
        """Split long sections into smaller chunks while preserving meaning"""
        if len(section) <= max_chunk_size:
            return [section]
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _create_tools(self) -> List[Tool]:
        """Create enhanced tools for the agent"""
        return [
            Tool(
                name="SearchDocuments",
                func=self._search_documents,
                description="Search for relevant information in documents. Use keywords or questions."
            ),
            Tool(
                name="GetDocumentStructure",
                func=self._get_document_structure,
                description="Get overview of document structure (clauses, sections, etc.)"
            ),
            Tool(
                name="FindByClause",
                func=self._find_by_clause,
                description="Find specific clause by number (e.g., 'clause 5' or '5')"
            ),
            Tool(
                name="SearchByKeywords",
                func=self._search_by_keywords,
                description="Search documents using specific keywords with metadata filtering"
            ),
            Tool(
                name="GetSimilarContent",
                func=self._get_similar_content,
                description="Find content similar to a given text or concept"
            )
        ]

    def _search_documents(self, query: str, k: int = 5) -> str:
        """Enhanced document search with metadata utilization"""
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                return "No relevant documents found."
            
            # Format results with metadata
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                meta = doc.metadata
                
                # Create rich result description
                result_info = f"Result {i} (Relevance: {1-score:.2f}):\n"
                result_info += f"Type: {meta.get('type', 'unknown')}\n"
                
                if meta.get('display_title'):
                    result_info += f"Title: {meta['display_title']}\n"
                
                if meta.get('keywords'):
                    result_info += f"Keywords: {', '.join(meta['keywords'])}\n"
                
                result_info += f"Content: {doc.page_content}\n"
                result_info += "-" * 50 + "\n"
                
                formatted_results.append(result_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def _get_document_structure(self, _: str = "") -> str:
        """Get document structure overview"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            structure = metadata['document_structure']
            info = metadata['index_info']
            
            result = f"Document Structure Overview:\n"
            result += f"Source: {info['source_file']}\n"
            result += f"Total Documents: {info['total_documents']}\n"
            result += f"Created: {info['created_at']}\n\n"
            
            result += "Content Breakdown:\n"
            for content_type, count in structure.items():
                if count > 0:
                    result += f"- {content_type.title()}: {count}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting document structure: {str(e)}"

    def _find_by_clause(self, clause_identifier: str) -> str:
        """Find specific clause by number"""
        try:
            # Extract number from identifier
            match = re.search(r'\d+', clause_identifier)
            if not match:
                return "Invalid clause identifier. Please provide a clause number."
            
            clause_num = int(match.group())
            
            # Search for documents with matching clause number
            matching_docs = [
                doc for doc in self.documents 
                if doc.metadata.get('clause_number') == clause_num
            ]
            
            if not matching_docs:
                return f"Clause {clause_num} not found."
            
            # Format results
            results = []
            for doc in matching_docs:
                result = f"Clause {clause_num}:\n"
                result += f"{doc.page_content}\n"
                if len(matching_docs) > 1:
                    result += f"(Part {doc.metadata.get('chunk_index', 1)} of {doc.metadata.get('total_chunks', 1)})\n"
                results.append(result)
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error finding clause: {str(e)}"

    def _search_by_keywords(self, keywords: str) -> str:
        """Search using keywords with metadata filtering"""
        try:
            keyword_list = [kw.strip().lower() for kw in keywords.split(',')]
            
            # Find documents containing keywords in their metadata
            matching_docs = []
            for doc in self.documents:
                doc_keywords = [kw.lower() for kw in doc.metadata.get('keywords', [])]
                if any(kw in doc_keywords for kw in keyword_list):
                    matching_docs.append(doc)
            
            if not matching_docs:
                # Fallback to content search
                return self._search_documents(keywords)
            
            # Format results
            results = []
            for i, doc in enumerate(matching_docs[:5], 1):  # Limit to 5 results
                meta = doc.metadata
                result = f"Result {i}:\n"
                result += f"Type: {meta.get('type', 'unknown')}\n"
                result += f"Keywords: {', '.join(meta.get('keywords', []))}\n"
                result += f"Content: {doc.page_content}\n"
                result += "-" * 50 + "\n"
                results.append(result)
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error searching by keywords: {str(e)}"

    def _get_similar_content(self, text: str, k: int = 3) -> str:
        """Find content similar to given text"""
        try:
            # Use the text directly for similarity search
            results = self.vectorstore.similarity_search_with_score(text, k=k)
            
            if not results:
                return "No similar content found."
            
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                similarity = 1 - score
                result = f"Similar Content {i} (Similarity: {similarity:.2f}):\n"
                result += f"Type: {doc.metadata.get('type', 'unknown')}\n"
                result += f"{doc.page_content}\n"
                result += "-" * 40 + "\n"
                formatted_results.append(result)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error finding similar content: {str(e)}"

    def _create_agent(self):
        """Create enhanced agent with better prompt"""
        system_message = SystemMessage(content=f"""You are {self.agent_name}, an advanced document analysis assistant.

You have access to enhanced tools that can:
1. Search documents with relevance scoring
2. Find specific clauses by number
3. Search by keywords using metadata
4. Get document structure overview
5. Find similar content

Always use the most appropriate tool for each query. When searching:
- Use SearchDocuments for general queries
- Use FindByClause for specific clause requests
- Use SearchByKeywords when user mentions specific terms
- Use GetDocumentStructure for overview questions
- Use GetSimilarContent to find related information

Provide clear, accurate responses with relevant context from the documents.
If information is not found, suggest alternative search approaches.""")
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

    def ask(self, question: str) -> str:
        """Ask a question to the agent"""
        try:
            response = self.agent_executor.invoke({"input": question})
            return response.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            return f"Error processing question: {str(e)}"

    def get_metadata_info(self) -> Dict[str, Any]:
        """Get comprehensive metadata information"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Could not load metadata: {str(e)}"}

def main():
    """Example usage"""
    try:
        # Get the correct file path
        current_dir = Path(__file__).parent.parent
        data_file = current_dir / "data" / "report.txt"
        
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            print("Please ensure the data file exists or update the path.")
            return
        
        # Create enhanced agent
        agent = EnhancedDocumentAgent("LegalAssistant", str(data_file))
        
        # Get metadata info
        print("Metadata Info:")
        print(json.dumps(agent.get_metadata_info(), indent=2))
        
        # Example questions
        questions = [
            "What are the main topics covered?",
            "What is the lease agreement validity period?",
            "What happens in force majeure cases?"
        ]
        
        for question in questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            print(f"{'='*50}")
            answer = agent.ask(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()