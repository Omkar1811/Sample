import os
import openai
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import logging

from config import Config
from pdf_processor import LegalDocumentProcessor
from vector_store import LegalVectorStore, LegalQueryEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGSystem:
    def __init__(self, openai_api_key: str = None):
        # Set up OpenAI
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.openai_api_key
        
        # Initialize components
        self.document_processor = LegalDocumentProcessor(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=Config.PDF_CHUNK_OVERLAP
        )
        
        self.vector_store = LegalVectorStore()
        self.query_enhancer = LegalQueryEnhancer()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=self.openai_api_key
        )
        
        logger.info("Legal RAG System initialized successfully")
    
    def load_documents(self, pdf_directory: str = None) -> None:
        """Load and process all PDF documents"""
        logger.info("Starting document loading process...")
        
        # Use default data directory if not specified
        if pdf_directory is None:
            pdf_directory = Config.PDF_DATA_PATH
        
        # Process all PDFs
        documents = self.document_processor.process_all_pdfs(pdf_directory)
        
        if not documents:
            logger.warning("No documents were processed")
            return
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Log statistics
        stats = self.vector_store.get_collection_stats()
        logger.info(f"Loaded {stats['total_documents']} document chunks")
        logger.info(f"Case type distribution: {stats['case_type_distribution']}")
    
    def analyze_legal_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and provide comprehensive legal guidance"""
        logger.info(f"Analyzing query: {user_query}")
        
        # Enhance the query
        enhanced_query = self.query_enhancer.enhance_query(user_query)
        legal_entities = self.query_enhancer.extract_legal_entities(user_query)
        
        # Get relevant cases
        relevant_cases = self.vector_store.get_similar_cases(
            enhanced_query,
            similarity_threshold=Config.SIMILARITY_THRESHOLD,
            max_results=Config.MAX_RELEVANT_CASES
        )
        
        if not relevant_cases:
            return {
                "query": user_query,
                "relevant_cases": [],
                "legal_analysis": "No relevant cases found in the database for your query.",
                "legal_entities": legal_entities,
                "recommendations": ["Please try rephrasing your query or contact a legal professional for specific advice."]
            }
        
        # Prepare context for LLM
        context = self._prepare_context(relevant_cases)
        
        # Generate legal analysis
        legal_analysis = self._generate_legal_analysis(user_query, context)
        
        # Extract case information
        case_summaries = self._extract_case_summaries(relevant_cases)
        
        return {
            "query": user_query,
            "enhanced_query": enhanced_query,
            "legal_entities": legal_entities,
            "relevant_cases": case_summaries,
            "legal_analysis": legal_analysis,
            "similarity_scores": [case['similarity_score'] for case in relevant_cases],
            "recommendations": self._generate_recommendations(relevant_cases, legal_entities)
        }
    
    def _prepare_context(self, relevant_cases: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant cases for LLM"""
        context_parts = []
        
        for i, case in enumerate(relevant_cases):
            metadata = case['metadata']
            document = case['document']
            
            case_info = f"""
            CASE {i+1}: {metadata.get('case_title', 'Unknown Case')}
            Date: {metadata.get('date', 'Unknown')}
            Case Type: {metadata.get('case_type', 'Unknown')}
            Legal Sections: {', '.join(metadata.get('legal_sections', []))}
            Similarity Score: {case['similarity_score']:.3f}
            
            Content: {document[:800]}...
            
            ---
            """
            context_parts.append(case_info)
        
        return "\n".join(context_parts)
    
    def _generate_legal_analysis(self, query: str, context: str) -> str:
        """Generate comprehensive legal analysis using LLM"""
        prompt = Config.get_prompt_template().format(
            context=context,
            query=query
        )
        
        try:
            from langchain.schema import HumanMessage
            
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating legal analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"
    
    def _extract_case_summaries(self, relevant_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key information from relevant cases"""
        summaries = []
        
        for case in relevant_cases:
            metadata = case['metadata']
            summary = {
                "case_title": metadata.get('case_title', 'Unknown Case'),
                "date": metadata.get('date', 'Unknown'),
                "case_type": metadata.get('case_type', 'Unknown'),
                "legal_sections": metadata.get('legal_sections', []),
                "judges": metadata.get('judges', []),
                "similarity_score": case['similarity_score'],
                "key_content": case['document'][:300] + "..." if len(case['document']) > 300 else case['document']
            }
            summaries.append(summary)
        
        return summaries
    
    def _generate_recommendations(
        self, 
        relevant_cases: List[Dict[str, Any]], 
        legal_entities: Dict[str, List[str]]
    ) -> List[str]:
        """Generate practical recommendations based on cases and entities"""
        recommendations = []
        
        # Analyze case types
        case_types = [case['metadata'].get('case_type', 'Unknown') for case in relevant_cases]
        most_common_type = max(set(case_types), key=case_types.count) if case_types else None
        
        if most_common_type:
            recommendations.append(f"This appears to be primarily a {most_common_type} law matter.")
        
        # Analyze legal sections
        all_sections = []
        for case in relevant_cases:
            all_sections.extend(case['metadata'].get('legal_sections', []))
        
        if all_sections:
            common_sections = list(set(all_sections))
            recommendations.append(f"Key legal sections to consider: {', '.join(common_sections[:5])}")
        
        # General recommendations
        if relevant_cases:
            recommendations.extend([
                "Consult with a qualified lawyer for specific legal advice.",
                "Review the full text of the relevant case judgments.",
                "Consider the current legal status and any subsequent amendments.",
                "Document all evidence and maintain proper records."
            ])
        
        return recommendations
    
    def search_by_case_type(self, query: str, case_type: str) -> Dict[str, Any]:
        """Search for cases of a specific type"""
        enhanced_query = self.query_enhancer.enhance_query(query)
        
        relevant_cases = self.vector_store.search_by_case_type(
            enhanced_query, 
            case_type, 
            k=Config.MAX_RELEVANT_CASES
        )
        
        if not relevant_cases:
            return {
                "query": query,
                "case_type": case_type,
                "message": f"No {case_type} cases found related to your query."
            }
        
        context = self._prepare_context(relevant_cases)
        legal_analysis = self._generate_legal_analysis(query, context)
        case_summaries = self._extract_case_summaries(relevant_cases)
        
        return {
            "query": query,
            "case_type": case_type,
            "relevant_cases": case_summaries,
            "legal_analysis": legal_analysis
        }
    
    def search_by_legal_sections(self, query: str, sections: List[str]) -> Dict[str, Any]:
        """Search for cases involving specific legal sections"""
        enhanced_query = self.query_enhancer.enhance_query(query)
        
        relevant_cases = self.vector_store.search_by_legal_sections(
            enhanced_query, 
            sections, 
            k=Config.MAX_RELEVANT_CASES
        )
        
        if not relevant_cases:
            return {
                "query": query,
                "sections": sections,
                "message": f"No cases found involving sections: {', '.join(sections)}"
            }
        
        context = self._prepare_context(relevant_cases)
        legal_analysis = self._generate_legal_analysis(query, context)
        case_summaries = self._extract_case_summaries(relevant_cases)
        
        return {
            "query": query,
            "sections": sections,
            "relevant_cases": case_summaries,
            "legal_analysis": legal_analysis
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.vector_store.get_collection_stats()
    
    def reset_system(self) -> None:
        """Reset the entire system (useful for reloading documents)"""
        logger.info("Resetting Legal RAG System...")
        self.vector_store.reset_collection()
        logger.info("System reset complete")

# Convenience function for quick setup
def create_legal_rag_system(pdf_directory: str = None, openai_api_key: str = None) -> LegalRAGSystem:
    """Create and initialize a Legal RAG System"""
    rag_system = LegalRAGSystem(openai_api_key=openai_api_key)
    
    # Check if documents are already loaded
    stats = rag_system.get_system_stats()
    if stats['total_documents'] == 0:
        logger.info("No documents found in vector store. Loading documents...")
        rag_system.load_documents(pdf_directory)
    else:
        logger.info(f"Found {stats['total_documents']} documents in vector store.")
    
    return rag_system 