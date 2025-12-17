# 🇹🇳 Agentic RAG TN (Tunisia): Safe, Agentic Retrieval + Gemini Fallback

Agentic RAG tailored for Tunisian contexts that enforces a curated XLSX-based bad-word filter and falls back to a Gemini LLM when no retrieval answer exists. Designed to avoid exposing abusive or violent language to young users (ages 10–18).

## 🎯 Features

- **Multilingual Document Processing**: Language-aware chunking with RTL text support (Arabic, Hebrew, etc.)
- **Intelligent Vector Search**: ChromaDB-powered semantic search with automatic deduplication
- **Conversation Memory**: Context-aware responses using semantic similarity and temporal relevance
- **Web Scraping**: Automatically extracts and ranks URLs from documents
- **Interactive CLI**: User-friendly command-line interface

## 📋 Components

### 1. **MultilingualDocumentProcessor** (`multilingual_processor.py`)
- Language detection with `langdetect`
- RTL (Right-to-Left) text normalization
- Semantic chunking with metadata preservation

### 2. **IntelligentWebScraper** (`web_scraper.py`)
- URL extraction from documents
- Semantic ranking of URLs based on query relevance
- Context-aware web scraping

### 3. **OptimizedVectorStore** (`vector_store.py`)
- ChromaDB integration with persistent storage
- Automatic document deduplication using MD5 hashing
- Metadata indexing for efficient filtering

### 4. **ConversationMemoryEngine** (`memory_engine.py`)
- Semantic similarity matching with past conversations
- Temporal relevance weighting (exponential decay)
- Token-aware context management

### 5. **MultilingualRAGSystem** (`rag_system.py`)
- Main orchestrator integrating all components
- PDF processing with `pypdf`
- Query handling with multi-source retrieval

## 🚀 Installation & Setup

### 1. Activate Virtual Environment
```bash
source ~/python/bin/activate
```

### 2. Install Dependencies
All major dependencies are already installed in your environment. To verify:
```bash
pip list | grep -E "sentence-transformers|chromadb|langchain|tiktoken|pypdf"
```

If any are missing, install from requirements:
```bash
pip install -r requirements.txt
```

### 3. Provide Gemini API Key (optional fallback)
Set the `GEMINI_API_KEY` environment variable if you want the system to generate answers when retrieval fails:

```bash
export GEMINI_API_KEY="ya29.your_token_here"
# Optional: provide a custom XLSX path
export BAD_WORDS_XLSX="/path/to/T-HSAB.xlsx"
```

Notes:
- The system will always apply the XLSX filter (see `T-HSAB.xlsx`) to both retrieved snippets and any generated LLM answers.
- The Gemini endpoint used is configurable via `GEMINI_API_URL`.

## 💻 Usage

### Quick Start

```bash
python3 main.py
```

This will:
1. Initialize the RAG system
2. Load the PDF from `/path/to/your/pdf`
3. Process and index the document
4. Start the interactive CLI

### Available Commands

```
query <question>    - Ask a question about the document
stats              - Show system statistics
history            - Show conversation summary
clear              - Clear all data (vector store + memory)
help               - Show help message
exit/quit          - Exit the program
```

### Example Session

```
💬 You: query What is the main topic of this document?

🤖 Assistant:
Based on the document, here are the most relevant passages:
[Source 1, Page 1]: Introduction to RAG systems...
...

📚 Sources (5 found):
  • Page 1 (similarity: 87.3%)
  • Page 3 (similarity: 82.1%)
  • Page 5 (similarity: 78.9%)

💬 You: stats

📊 System Statistics:
  Vector Store: {'total_documents': 245, 'collection_name': 'wie_rag_collection'}
  Memory: Total interactions: 12
```

## 📂 File Structure

```
./
├── agentic_rag_tn/
│   ├── main.py                     # CLI entrypoint
│   └── core/
│       ├── rag_system.py           # Retrieval + fallback + filtering
│       ├── filter.py               # XLSX-based Tunisian bad-words filter
│       └── __init__.py
├── multilingual_processor.py       # Multilingual text processor
├── memory_engine.py                # Conversation memory
├── vector_store.py                 # Optimized ChromaDB interface
├── web_scraper.py                  # URL extraction + scraping
├── T-HSAB.xlsx                     # Optional curated Tunisian filter file
├── requirements.txt
├── api.py                          # FastAPI service
├── chromadb_data/                  # Persistent DB (auto-created)
└── deploy_acr_aci.sh               # Azure build & deployment helper
             # Persistent vector store (auto-created)
```

## 🔧 Configuration

### Embedding Model
Default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Change in `rag_system.py`:
```python
rag_system = MultilingualRAGSystem(
    embedding_model_name="your-model-name"
)
```

### Vector Store
Default location: `chromadb_data/`

Change in `main.py`:
```python
rag_system = MultilingualRAGSystem(
    persist_directory="your-custom-path",
    collection_name="your-collection-name"
)
```

### PDF Path
Default: `/path/to/your/pdf`

Change in `main.py`:
```python
pdf_path = "your-pdf-path.pdf"
```

## 🎨 Advanced Features

### Query with Options

```python
result = rag_system.query(
    query_text="Your question",
    n_results=10,           # Number of results to retrieve
    use_memory=True,        # Use conversation context
    scrape_urls=True        # Scrape URLs from documents
)
```

### Clear Data

```python
rag_system.clear_all()  # Clears vector store and memory
```

### Export Conversation

```python
history = rag_system.memory_engine.export_history()
```

## 🧪 Testing

Test individual components:

```python
# Test document processor
from multilingual_processor import MultilingualDocumentProcessor
processor = MultilingualDocumentProcessor()
chunks = processor.process_document({
    'content': 'Your text here',
    'metadata': {},
    'page_number': 1
})

# Test vector store
from vector_store import OptimizedVectorStore
store = OptimizedVectorStore()
stats = store.get_collection_stats()
print(stats)
```

## ⚠️ Important Safety Warning
[!WARNING] Sensitivity of Filter Data (T-HSAB.xlsx): The T-HSAB.xlsx file is a curated dataset containing highly offensive, abusive, and violent language (18+) specifically targeting Tunisian dialect and cultural contexts.

Do not open or display this file in educational or public settings.

It is intended strictly for programmatic filtering to prevent the RAG system from echoing or generating harmful content for the target audience (ages 10–18).

Handle this file with the same care as sensitive data to avoid accidental exposure.
## 🐛 Troubleshooting

### Issue: PDF not found
**Solution**: Verify the path exists:
```bash
ls -la /path/to/your/pdf
```

### Issue: ChromaDB errors
**Solution**: Clear the vector store:
```bash
rm -rf chromadb_data/
```

### Issue: Out of memory
**Solution**: Reduce chunk size in `multilingual_processor.py`:
```python
MultilingualDocumentProcessor(chunk_size=512, chunk_overlap=100)
```
