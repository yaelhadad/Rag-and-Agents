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
    def __init__(self, agent_name: str, filepaths: list, index_path: Optional[str] = None):
        """Initialize enhanced document agent with multiple documents"""
        self.agent_name = agent_name
        self.filepaths = [Path(fp) for fp in filepaths]
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
        
        # Load or create index with automatic rebuild if files changed
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
        """Calculate MD5 checksum of all source files"""
        hash_md5 = hashlib.md5()
        for filepath in self.filepaths:
            with open(filepath, "rb") as f:
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
            
        # Check if all files exist
        for filepath in self.filepaths:
            if not filepath.exists():
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
        
        # Create sample documents from each source
        sample_documents = []
        sources_seen = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in sources_seen and len(sample_documents) < 6:  # Max 6 samples (3 per source if 2 sources)
                sample_documents.append({
                    "content_preview": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                })
                sources_seen.add(source)
        
        # If we have fewer sources than expected, add more samples
        if len(sample_documents) < 3:
            for doc in documents:
                if len(sample_documents) >= 3:
                    break
                sample_documents.append({
                    "content_preview": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                })
        
        # Create summary
        return {
            "index_info": {
                "agent_name": self.agent_name,
                "source_files": [str(fp) for fp in self.filepaths],
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
            "sample_documents": sample_documents
        }

    def _parse_document_with_metadata(self) -> List[Document]:
        """Parse multiple documents with comprehensive automatic metadata extraction"""
        try:
            all_documents = []
            
            for file_idx, filepath in enumerate(self.filepaths, 1):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                sections = self._split_into_sections(content)
                
                for section_idx, section in enumerate(sections, 1):
                    # Extract section metadata
                    metadata = self._extract_section_metadata(section, section_idx, filepath, file_idx)
                    
                    # Split section into smaller chunks if too long
                    chunks = self._split_section_into_chunks(section)
                    
                    for chunk_idx, chunk in enumerate(chunks, 1):
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'chunk_id': f"file{file_idx}_section{section_idx}_chunk{chunk_idx}",
                            'content_length': len(chunk),
                            'word_count': len(chunk.split()),
                            'created_at': datetime.now().isoformat()
                        })
                        
                        all_documents.append(Document(
                            page_content=chunk,
                            metadata=chunk_metadata
                        ))
            
            logger.info(f"Created {len(all_documents)} documents from {len(self.filepaths)} files with enhanced metadata")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error parsing documents: {e}")
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

    def _extract_section_metadata(self, section: str, section_idx: int, filepath: Path, file_idx: int) -> Dict[str, Any]:
        """Extract comprehensive metadata from a section"""
        metadata = {
            'section_index': section_idx,
            'source': str(filepath.name),
            'agent': self.agent_name,
            'file_index': file_idx
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
                description="Search for capabilities of a user group. Use: 'user_group' or 'user_group capability'"
            ),
            Tool(
                name="SearchUserGroup",
                func=self._search_user_group,
                description="Search for information about a specific user group (customers, staff, admins)"
            ),
            Tool(
                name="CompareSystems",
                func=self._compare_systems,
                description="Compare systems and their features from different documents"
            )
        ]

    def _search_documents(self, query: str, k: int = 2) -> str:
        """Search for relevant information across all documents"""
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                return "No relevant documents found."
            
            # Format results with source information
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                source_file = doc.metadata.get('source', 'Unknown')
                result_info = f"Result {i} (Source: {source_file}):\n"
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

    def _search_user_capabilities(self, user_group: str, capability: str = None) -> str:
        """Search for specific capabilities of a user group"""
        try:
            # Create focused search query
            if capability:
                search_query = f"{user_group} {capability}"
            else:
                search_query = f"{user_group} capabilities permissions"
            
            results = self.vectorstore.similarity_search_with_score(search_query, k=2)
            
            if not results:
                if capability:
                    return f"No information found about {user_group} and {capability}."
                else:
                    return f"No information found about {user_group} capabilities."
            
            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                source_file = doc.metadata.get('source', 'Unknown')
                result_info = f"Result {i} (Source: {source_file}):\n"
                result_info += f"Content: {doc.page_content}\n"
                result_info += "-" * 40 + "\n"
                formatted_results.append(result_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching for user capabilities: {str(e)}"

    def _search_user_group(self, query: str) -> str:
        """Search for user group information (wrapper for single parameter calls)"""
        try:
            # Extract user group from query
            user_group = query.strip()
            return self._search_user_capabilities(user_group)
        except Exception as e:
            return f"Error searching for user group: {str(e)}"

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

    def _compare_systems(self, comparison_query: str) -> str:
        """Compare systems and their features"""
        try:
            # Search for system features
            system_features = {}
            
            # Search for features in each document
            for doc in self.documents:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content.lower()
                
                # Check if this document contains feature information
                if any(keyword in content for keyword in ['feature', 'functionality', 'capability', 'can', 'able']):
                    if source not in system_features:
                        system_features[source] = []
                    system_features[source].append(doc.page_content)
            
            if not system_features:
                return "No system features found in the documents."
            
            # Format comparison
            comparison_result = "System Comparison:\n" + "="*50 + "\n"
            
            for system_name, features in system_features.items():
                comparison_result += f"\n{system_name}:\n"
                comparison_result += "-" * 30 + "\n"
                for i, feature in enumerate(features[:3], 1):  # Limit to 3 features per system
                    comparison_result += f"{i}. {feature[:200]}...\n"
                comparison_result += "\n"
            
            return comparison_result
            
        except Exception as e:
            return f"Error comparing systems: {str(e)}"

    def _create_agent(self):
        """Create simplified agent with multi-document context memory support"""
        system_message = SystemMessage(content=f"""You are {self.agent_name}, a document analysis assistant with context memory and multi-document comparison capabilities.

You have access to essential tools:
1. SearchDocuments - Search for relevant information across all documents
2. GetContextInfo - Get information about previously mentioned entities
3. CompareEntities - Compare different entities or concepts
4. SearchUserCapabilities - Search for capabilities of a user group. Use: 'user_group' or 'user_group capability'
5. SearchUserGroup - Search for information about a specific user group (customers, staff, admins)
6. CompareSystems - Compare systems and their features from different documents

MULTI-DOCUMENT CAPABILITIES:
- You can analyze and compare information from multiple documents
- Each document chunk includes source file information in metadata
- You can identify which document contains specific information
- You can compare similar concepts across different documents
- When comparing systems, always identify which system each feature belongs to

CONTEXT MEMORY CAPABILITIES:
- Always consider the conversation history when answering questions
- When a question references "mentioned earlier" or "previously mentioned", use the GetContextInfo tool
- Maintain context about entities, roles, and concepts mentioned in the conversation

CRITICAL CONTEXT RULES:
- When asked "Among the users mentioned earlier", ONLY consider users from the previous answer
- If the previous answer mentioned "customers", then "among the users mentioned earlier" refers ONLY to customers
- Do NOT include other user groups (admins, staff) unless they were specifically mentioned in the previous context
- Focus your search and answer ONLY on the user groups that were actually mentioned earlier

COMPARISON GUIDELINES:
- When asked to compare systems, search for features in each system separately
- Always specify which system each feature belongs to
- Provide structured comparisons: "System A has: [features], System B has: [features]"
- Highlight differences and similarities clearly
- Use source information to identify which document contains which features

ANSWER GUIDELINES:
- Provide focused, specific answers that directly address the question
- If asked about a specific user group, focus only on that group
- Be precise about permissions and capabilities for each user role
- When asked about "who can do X", only mention users who actually have that specific capability
- When asked "among the users mentioned earlier", restrict your answer to only those users
- When comparing information between documents, clearly indicate the source of each piece of information
- For system comparisons, provide detailed, structured responses

Use the most appropriate tool for each query:
- Use SearchDocuments for general queries across all documents
- Use GetContextInfo when asked about previously mentioned entities
- Use CompareEntities when asked to compare different things
- Use SearchUserCapabilities when asked about specific user group capabilities
- Use SearchUserGroup when asked about a specific user group (customers, staff, admins)
- Use CompareSystems when asked to compare systems and their features

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
    """Example usage with unified multi-document index"""
    try:
        # Get the correct file paths
        current_dir = Path(__file__).parent.parent
        restaurant_file = current_dir / "data" / "report.txt"
        task_tracker_file = current_dir / "data" / "system_design_daily_dev_team_task_tracker.txt"
        
        # Check if files exist
        files_to_use = []
        if restaurant_file.exists():
            files_to_use.append(str(restaurant_file))
            print(f"✅ Found: {restaurant_file.name}")
        if task_tracker_file.exists():
            files_to_use.append(str(task_tracker_file))
            print(f"✅ Found: {task_tracker_file.name}")
        
        if not files_to_use:
            print("❌ No data files found")
            print("Please ensure at least one data file exists.")
            return
        
        print(f"✅ Using {len(files_to_use)} document(s): {[Path(f).name for f in files_to_use]}")
        
        # Create unified agent with single index
        agent = EnhancedDocumentAgent("UnifiedSystemAnalyzer", files_to_use)
        
        # Get metadata info to verify unified index
        print("\n" + "="*60)
        print("📊 Unified Index Information")
        print("="*60)
        metadata_info = agent.get_metadata_info()
        print(f"Total documents: {metadata_info['index_info']['total_documents']}")
        print(f"Source files: {[Path(f).name for f in metadata_info['index_info']['source_files']]}")
        
        # Test the two questions
        print("\n" + "="*60)
        print("🧠 Testing Unified Multi-Document Context Memory")
        print("="*60)
        
        # First question - establishes context
        print("\n1. Who sees the vegan meal promotion image in the system?")
        answer1 = agent.ask("Who sees the vegan meal promotion image in the system?")
        print(f"Answer: {answer1}")
        
        # Second question - should reference context from first question
        print("\n2. Among the users that were answered earlier, who can cancel any reservation in the system?")
        answer2 = agent.ask("Among the users that were answered earlier, who can cancel any reservation in the system?")
        print(f"Answer: {answer2}")
        
        # Test multi-document comparison with improved prompt
        if len(files_to_use) > 1:
            print("\n3. Compare the features between the two systems")
            print("Expected: Detailed comparison of features from both systems")
            print("-" * 50)
            answer3 = agent.ask("Compare the features between the two systems. Please provide a detailed comparison showing what features each system has and how they differ.")
            print(f"Answer: {answer3}")
        
        print("\n✅ Unified multi-document test completed!")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()