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
        
        # Top 2 keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
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
        """Create essential tools for the agent"""
        return [
            Tool(
                name="SearchDocuments",
                func=self._search_documents,
                description="Search for relevant information in documents. Use keywords or questions."
            ),
            Tool(
                name="GetContextInfo",
                func=self._get_context_info,
                description="Get information about previously mentioned entities, roles, or concepts"
            ),
            Tool(
                name="CompareEntities",
                func=self._compare_entities,
                description="Compare different entities, roles, or concepts mentioned in the conversation"
            ),
            Tool(
                name="SearchUserCapabilities",
                func=self._search_user_capabilities,
                description="Search for specific capabilities of a user group"
            )
        ]

    def _search_documents(self, query: str, k: int = 2) -> str:
        """Search for relevant information in documents"""
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                return "No relevant documents found."
            
            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                result_info = f"Result {i}:\n"
                result_info += f"Content: {doc.page_content}\n"
                result_info += "-" * 40 + "\n"
                formatted_results.append(result_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def _get_context_info(self, context_query: str) -> str:
        """Get information about previously mentioned entities or concepts"""
        try:
            # Search for the context query in the conversation history
            if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
                # Look for mentions in recent conversation
                recent_context = []
                for message in self.memory.chat_memory.messages[-6:]:  # Last 6 messages
                    if context_query.lower() in message.content.lower():
                        recent_context.append(message.content)
                
                if recent_context:
                    # Extract specific user groups from the context
                    context_text = recent_context[-1]  # Get the most recent context
                    
                    # Look for specific user groups mentioned
                    user_groups = []
                    if 'customers' in context_text.lower():
                        user_groups.append('customers')
                    if 'staff' in context_text.lower():
                        user_groups.append('staff')
                    if 'admins' in context_text.lower() or 'admin' in context_text.lower():
                        user_groups.append('admins')
                    
                    if user_groups:
                        context_info = f"Previously mentioned users: {', '.join(user_groups)}\n"
                        context_info += f"Context: {context_text[:200]}...\n"
                        return context_info
                    else:
                        return f"Context found: {context_text[:200]}..."
            
            # Fallback to document search
            return self._search_documents(context_query)
            
        except Exception as e:
            return f"Error getting context info: {str(e)}"

    def _search_user_capabilities(self, user_group: str, capability: str) -> str:
        """Search for specific capabilities of a user group"""
        try:
            # Create focused search query
            search_query = f"{user_group} {capability}"
            results = self.vectorstore.similarity_search_with_score(search_query, k=2)
            
            if not results:
                return f"No information found about {user_group} and {capability}."
            
            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                result_info = f"Result {i}:\n"
                result_info += f"Content: {doc.page_content}\n"
                result_info += "-" * 40 + "\n"
                formatted_results.append(result_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching for user capabilities: {str(e)}"

    def _compare_entities(self, comparison_query: str) -> str:
        """Compare different entities mentioned in the conversation"""
        try:
            # Extract entities from the comparison query
            entities = comparison_query.split(' and ')
            if len(entities) < 2:
                return "Please provide at least two entities to compare (e.g., 'customers and admins')"
            
            # Search for information about each entity
            results = []
            for entity in entities:
                entity = entity.strip()
                entity_info = self._search_documents(entity)
                results.append(f"Information about '{entity}':\n{entity_info}\n")
            
            # Return comparison information
            return "\n".join(results)
            
        except Exception as e:
            return f"Error comparing entities: {str(e)}"

    def _create_agent(self):
        """Create simplified agent with context memory support"""
        system_message = SystemMessage(content=f"""You are {self.agent_name}, a document analysis assistant with context memory.

You have access to essential tools:
1. SearchDocuments - Search for relevant information in documents
2. GetContextInfo - Get information about previously mentioned entities
3. CompareEntities - Compare different entities or concepts
4. SearchUserCapabilities - Search for specific capabilities of a user group

CONTEXT MEMORY CAPABILITIES:
- Always consider the conversation history when answering questions
- When a question references "mentioned earlier" or "previously mentioned", use the GetContextInfo tool
- Maintain context about entities, roles, and concepts mentioned in the conversation

CRITICAL CONTEXT RULES:
- When asked "Among the users mentioned earlier", ONLY consider users from the previous answer
- If the previous answer mentioned "customers", then "among the users mentioned earlier" refers ONLY to customers
- Do NOT include other user groups (admins, staff) unless they were specifically mentioned in the previous context
- Focus your search and answer ONLY on the user groups that were actually mentioned earlier

ANSWER GUIDELINES:
- Provide focused, specific answers that directly address the question
- If asked about a specific user group, focus only on that group
- Be precise about permissions and capabilities for each user role
- When asked about "who can do X", only mention users who actually have that specific capability
- When asked "among the users mentioned earlier", restrict your answer to only those users

Use the most appropriate tool for each query:
- Use SearchDocuments for general queries
- Use GetContextInfo when asked about previously mentioned entities
- Use CompareEntities when asked to compare different things
- Use SearchUserCapabilities when asked about specific user group capabilities

Provide clear, accurate responses with relevant context from the documents and conversation history.""")
        
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
    """Example usage with the two specific questions"""
    try:
        # Get the correct file path
        current_dir = Path(__file__).parent.parent
        data_file = current_dir / "data" / "restaurant_ordering_system.txt"
        
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            print("Please ensure the data file exists or update the path.")
            return
        
        # Create simplified agent
        agent = EnhancedDocumentAgent("RestaurantSystemAnalyzer", str(data_file))
        
        # Test the two specific questions
        print("\n" + "="*60)
        print("🧠 Testing Context Memory with Two Questions")
        print("="*60)
        
        # First question - establishes context
        print("\n1. Who sees the vegan meal promotion image in the system?")
        answer1 = agent.ask("Who sees the vegan meal promotion image in the system?")
        print(f"Answer: {answer1}")
        
        # Second question - should reference context from first question
        print("\n2. Among the users that were answered earlier, who can cancel any reservation in the system?")
        answer2 = agent.ask("Among the users that were answered earlier, who can cancel any reservation in the system?")
        print(f"Answer: {answer2}")
        
        print("\n✅ Test completed!")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()