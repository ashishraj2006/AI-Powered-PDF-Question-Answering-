import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'llm_tokenizer' not in st.session_state:
    st.session_state.llm_tokenizer = None


# Load embedding model
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")


# Load LLM model
@st.cache_resource
def load_llm_model(model_name="microsoft/Phi-3-mini-4k-instruct"):
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None, None


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def process_pdf(pdf_file, model):
    try:
        # Extract text from PDF
        reader = PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            return None, None, "No text found in PDF."

        # Create chunks with overlap for better context
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        if not chunks:
            return None, None, "Failed to create text chunks."

        # Generate embeddings
        embeddings = model.encode(chunks, show_progress_bar=False)

        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))

        return chunks, index, f"Successfully indexed {len(chunks)} chunks from PDF."

    except Exception as e:
        return None, None, f"Error processing PDF: {str(e)}"


def generate_answer_with_llm(question, context, tokenizer, model):
    try:
        # Create prompt for the model
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the given context. Provide clear, concise, and well-structured answers.
<|end|>
<|user|>
Context from PDF:
{context}

Question: {question}

Please provide a well-structured answer based solely on the information in the context. If the context doesn't contain enough information, say so.
<|end|>
<|assistant|>"""

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def get_answer(question, top_k=3, use_ai=True):
    if not st.session_state.pdf_loaded:
        return "⚠️ Please upload and process a PDF first."

    try:
        # Encode question
        q_emb = st.session_state.embedding_model.encode([question])
        
        # Search for similar chunks
        distances, indices = st.session_state.index.search(
            np.array(q_emb).astype('float32'), 
            min(top_k, len(st.session_state.chunks))
        )

        # Retrieve top chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(st.session_state.chunks):
                relevant_chunks.append(st.session_state.chunks[idx])

        if not relevant_chunks:
            return "❌ No relevant information found in the PDF."

        # Combine chunks
        context = "\n\n".join(relevant_chunks)
        
        # Generate AI answer if enabled
        if use_ai and st.session_state.llm_model is not None:
            return generate_answer_with_llm(
                question, 
                context, 
                st.session_state.llm_tokenizer,
                st.session_state.llm_model
            )
        else:
            # Return raw context
            response = "📄 **Relevant Information:**\n\n"
            response += context
            return response

    except Exception as e:
        return f"❌ Search error: {str(e)}"


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="PDF Question Answering",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 AI-Powered PDF Question Answering")
    st.markdown("Upload a PDF and ask questions - powered by open-source LLMs")

    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Choose LLM Model",
            [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "microsoft/Phi-3-mini-4k-instruct",
            ],
            help="TinyLlama is faster but less accurate. Phi-3 is slower but better quality."
        )
        
        # AI toggle
        use_ai = st.toggle(
            "Use AI-Generated Answers",
            value=True,
            help="Generate intelligent answers using open-source LLMs"
        )
        
        # Load LLM button
        if st.button("🔄 Load LLM Model"):
            with st.spinner(f"Loading {model_choice}... This may take a few minutes..."):
                tokenizer, model = load_llm_model(model_choice)
                if tokenizer and model:
                    st.session_state.llm_tokenizer = tokenizer
                    st.session_state.llm_model = model
                    st.success("✅ LLM loaded successfully!")
                else:
                    st.error("❌ Failed to load LLM")
        
        # Display LLM status
        if st.session_state.llm_model is not None:
            st.success("✅ LLM Ready")
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.info(f"Running on: {device}")
        else:
            st.warning("⚠️ LLM not loaded")
        
        st.divider()
        
        st.header("📤 Upload PDF")
        
        # Load embedding model if not already loaded
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = load_embedding_model()
                st.success("✅ Embedding model loaded")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to ask questions about"
        )

        if uploaded_file:
            st.info(f"📎 Selected: {uploaded_file.name}")
            
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Indexing PDF... This may take a moment."):
                    chunks, index, message = process_pdf(uploaded_file, st.session_state.embedding_model)
                    
                    if chunks:
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.success(message)
                    else:
                        st.error(message)

        # Display current PDF status
        if st.session_state.pdf_loaded:
            st.success(f"✅ Active PDF: {st.session_state.pdf_name}")
            st.metric("Chunks Indexed", len(st.session_state.chunks))
        else:
            st.warning("⚠️ No PDF loaded")

        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF and generating answer..."):
                answer = get_answer(prompt, use_ai=use_ai)
                st.markdown(answer)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Show helpful tips if no PDF loaded
    if not st.session_state.pdf_loaded and not st.session_state.chat_history:
        st.info("👈 Start by loading the LLM and uploading a PDF in the sidebar")
        
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Load LLM Model** - Click "Load LLM Model" in the sidebar (first time only)
            2. **Upload a PDF** using the file uploader
            3. **Click "Process PDF"** to index the document
            4. **Ask questions** in the chat input below
            5. Get intelligent, well-formatted answers powered by open-source AI
            
            **Model Options:**
            - **TinyLlama**: Faster, smaller, good for quick answers
            - **Phi-3**: Better quality, more accurate, slightly slower
            
            **AI Mode:**
            - **ON**: Get clear, structured answers generated by AI
            - **OFF**: Get raw text chunks from the PDF
            
            **System Requirements:**
            - GPU recommended for faster inference
            - 4-8GB RAM minimum
            - Models are cached after first load
            
            **Tips:**
            - Ask specific questions for better results
            - First answer may be slow (model loading)
            - Subsequent answers are faster
            """)
        
        with st.expander("🔧 Installation Requirements"):
            st.code("""
pip install streamlit numpy faiss-cpu sentence-transformers pypdf transformers torch
            """)


if __name__ == "__main__":
    main()