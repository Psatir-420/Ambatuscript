import os
import logging
import google.generativeai as genai
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAGEngine")

class RAGEngine:
    def __init__(self, vector_store, api_key):
        """Initialize the RAG engine.
        
        Args:
            vector_store (VectorStore): Vector store for document retrieval
            api_key (str): Gemini API key
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def generate_response(self, query, num_results=3):
        """Generate a response using RAG (legacy method for backward compatibility).
        
        Args:
            query (str): User query
            num_results (int): Number of documents to retrieve
            
        Returns:
            dict: Response with answer and sources
        """
        # Create a fake chat history with just this query
        fake_history = [{"role": "user", "content": query}]
        return self.generate_response_with_chat(query, fake_history, num_results)
    
    def generate_response_with_chat(self, query, chat_history, num_results=3, available_documents=None):
        """Generate a response using RAG with chat history.
        
        Args:
            query (str): User query
            chat_history (list): List of chat messages
            num_results (int): Number of documents to retrieve
            available_documents (list): List of available document names
            
        Returns:
            dict: Response with answer, sources, and possibly document request
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return {
                "answer": "Error: Gemini API model not initialized. Please check your API key.",
                "sources": []
            }
        
        try:
            # Parse any document requests from the previous messages
            recent_document_requests = self._check_recent_document_requests(chat_history)

            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, top_k=num_results)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
                # If no relevant docs but we have available documents, suggest asking for specific ones
                if available_documents:
                    return self._handle_no_relevant_docs(query, chat_history, available_documents)
                else:
                    return {
                        "answer": "Saya tidak dapat menemukan informasi yang relevan untuk menjawab pertanyaan Anda. Silakan coba dengan pertanyaan yang berbeda atau pastikan dokumen telah diproses.",
                        "sources": []
                    }
            
            # Create context from retrieved documents and chat history
            context = self._create_context(relevant_docs, chat_history)
            
            # Create prompt with context
            prompt = self._create_prompt(query, context, chat_history, available_documents, recent_document_requests)
            
            # Generate response using Gemini
            raw_response = self._generate_with_gemini(prompt)
            
            # Process the response to extract document requests if any
            processed_response = self._process_response(raw_response)
            
            logger.info(f"Generated response for query: {query[:50]}...")
            
            return {
                "answer": processed_response["answer"],
                "sources": relevant_docs,
                "document_request": processed_response.get("document_request", None)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": []
            }
    
    def _check_recent_document_requests(self, chat_history):
        """Check for recent document request approvals or rejections."""
        requests = []
        
        # Look at the last few system messages for document request info
        for msg in reversed(chat_history):
            if msg.get("role") == "system" and "permintaan dokumen" in msg.get("content", ""):
                requests.append(msg["content"])
                
                # Only check the last few messages
                if len(requests) >= 3:
                    break
                    
        return requests
    
    def _handle_no_relevant_docs(self, query, chat_history, available_documents):
        """Handle case when no relevant documents are found but we know what's available."""
        # Create a special prompt to recommend documents
        prompt = f"""Anda adalah asisten hukum yang ahli dalam hukum Indonesia. Pengguna bertanya:

        "{query}"

        Sayangnya, tidak ada dokumen yang relevan ditemukan dalam basis data kami. Namun, kami memiliki dokumen-dokumen berikut:

        {', '.join(available_documents)}

        Berdasarkan pertanyaan pengguna, dokumen mana yang mungkin berisi informasi yang relevan? 
        Format respons Anda sebagai berikut:
        1. Beri tahu pengguna bahwa Anda tidak menemukan informasi yang relevan
        2. Tanyakan apakah mereka ingin Anda mencari informasi spesifik dalam [nama dokumen yang paling relevan]
        3. Jelaskan mengapa dokumen tersebut mungkin membantu
        
        Tambahkan [REQUEST_DOCUMENT:nama_dokumen] di bagian akhir respons Anda (pengguna tidak akan melihat teks dalam tanda kurung siku ini).
        """
        
        # Generate suggestion
        suggestion = self._generate_with_gemini(prompt)
        
        # Process to extract document request tag
        processed = self._process_response(suggestion)
        
        return processed
    
    def _create_context(self, relevant_docs, chat_history=None):
        """Create context from relevant documents and chat history.
        
        Args:
            relevant_docs (list): List of relevant document chunks
            chat_history (list, optional): Chat history
            
        Returns:
            str: Context text
        """
        context = "Here are some relevant documents to help answer the question:\n\n"
        
        for i, doc in enumerate(relevant_docs):
            context += f"Document {i+1} (Source: {os.path.basename(doc['source'])}, Pages: {doc['metadata']['page_start']}-{doc['metadata']['page_end']}):\n"
            context += doc["text"] + "\n\n"
        
        # Add recent chat history if available
        if chat_history and len(chat_history) > 1:
            context += "\nRecent conversation history:\n"
            # Get last few messages (up to 5)
            recent_history = chat_history[-min(6, len(chat_history)):-1]  # Exclude current query
            for msg in recent_history:
                role = msg.get("role", "unknown")
                if role == "system":  # Skip system messages
                    continue
                context += f"{role.capitalize()}: {msg.get('content', '')}\n"
        
        return context
    
    def _create_prompt(self, query, context, chat_history=None, available_documents=None, recent_document_requests=None):
        """Membuat prompt untuk model Gemini dalam Bahasa Indonesia.
    
        Args:
            query (str): Pertanyaan pengguna
            context (str): Konteks hasil pencarian dokumen
            chat_history (list, optional): Chat history
            available_documents (list, optional): List of available document names
            recent_document_requests (list, optional): Recent document requests information
    
        Returns:
            str: Prompt lengkap dalam Bahasa Indonesia
        """
        # Base prompt
        prompt = f"""Anda adalah asisten hukum yang ahli dalam hukum Indonesia. Berdasarkan dokumen-dokumen berikut, berikan jawaban yang relevan, jelas, dan mudah dipahami.

    {context}

    Silakan jawab pertanyaan di bawah ini berdasarkan informasi yang terdapat dalam dokumen di atas. Anda boleh menyusun ulang kalimat dengan bahasa Anda sendiri selama maknanya tetap sesuai dengan dokumen. Jangan menambahkan informasi dari luar dokumen. Jika dokumen tidak memuat informasi yang cukup, sampaikan bahwa jawabannya tidak tersedia.
    """
        
        # Add instructions for document requests if available_documents is provided
        if available_documents:
            prompt += f"""
    
    Jika Anda membutuhkan informasi tambahan yang mungkin ada dalam dokumen lain, Anda dapat meminta pengguna untuk memberikan izin mengakses dokumen tertentu. Berikut daftar dokumen yang tersedia:
    
    {', '.join(available_documents)}
    
    Jika Anda ingin meminta dokumen tertentu, tambahkan tag [REQUEST_DOCUMENT:nama_dokumen] di akhir jawaban Anda (pengguna tidak akan melihat teks dalam tanda kurung siku ini).
    """
        
        # Add information about recent document requests if any
        if recent_document_requests and len(recent_document_requests) > 0:
            prompt += f"""
    
    Informasi terkait permintaan dokumen sebelumnya:
    {' '.join(recent_document_requests)}
    """
        
        # Add the current query
        prompt += f"""
    Question: {query}
    
    Answer:"""
    
        return prompt
    
    def _process_response(self, response_text):
        """Process the response to extract document requests.
        
        Args:
            response_text (str): Raw response text
            
        Returns:
            dict: Processed response with answer and optional document request
        """
        # Check for document request tags
        doc_request_pattern = r'\[REQUEST_DOCUMENT:([^\]]+)\]'
        match = re.search(doc_request_pattern, response_text)
        
        result = {"answer": response_text}
        
        if match:
            document_name = match.group(1).strip()
            # Remove the tag from the displayed answer
            clean_answer = re.sub(doc_request_pattern, '', response_text).strip()
            result["answer"] = clean_answer
            result["document_request"] = document_name
            
        return result
    
    def _generate_with_gemini(self, prompt):
        """Generate a response using the Gemini API.
        
        Args:
            prompt (str): Complete prompt
            
        Returns:
            str: Generated response
        """
        try:
            # Generate content with the model
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in Gemini generation: {str(e)}")
            return f"Error generating response: {str(e)}"
