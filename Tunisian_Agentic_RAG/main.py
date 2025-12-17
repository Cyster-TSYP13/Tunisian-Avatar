#!/usr/bin/env python3
"""
Main Entry Point for Multilingual RAG System
Interactive CLI interface
"""

import sys
import os

# Disable TensorFlow to avoid Keras 3 compatibility issues
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['SENTENCE_TRANSFORMERS_NO_TF'] = '1'

from rag_system import MultilingualRAGSystem


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🌍 MULTILINGUAL RAG SYSTEM WITH AGENTIC CAPABILITIES 🤖  ║
║                                                              ║
║     Intelligent Document Q&A with Memory & Web Scraping     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print available commands"""
    help_text = """
📋 Available Commands:
  
  query <your question>  - Ask a question about the document
  stats                  - Show system statistics
  history                - Show conversation summary
  clear                  - Clear all data (vector store + memory)
  help                   - Show this help message
  exit / quit            - Exit the program
  
💡 Examples:
  query What is the main topic of this document?
  query Explain the methodology in detail
  stats
    """
    print(help_text)


def interactive_mode(rag_system: MultilingualRAGSystem):
    """Run interactive CLI mode"""
    print_help()
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            # Handle commands
            if command in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            
            elif command == 'help':
                print_help()
            
            elif command == 'stats':
                stats = rag_system.get_stats()
                print("\n📊 System Statistics:")
                print(f"  Vector Store: {stats['vector_store']}")
                print(f"  Memory: {stats['memory']}")
            
            elif command == 'history':
                summary = rag_system.memory_engine.get_summary()
                print(f"\n📜 Conversation History:\n{summary}")
            
            elif command == 'clear':
                confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ")
                if confirm.lower() == 'yes':
                    rag_system.clear_all()
                else:
                    print("❌ Cancelled")
            
            elif command == 'query':
                if len(parts) < 2:
                    print("❌ Please provide a question. Usage: query <your question>")
                    continue
                
                query_text = parts[1]
                
                # Execute query
                result = rag_system.query(query_text, n_results=5, use_memory=True)
                
                # Display answer
                print("\n🤖 Assistant:")
                print(result['answer'])
                
                # Display sources
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for source in result['sources'][:3]:  # Show top 3
                        page = source['metadata'].get('page_number', 'N/A')
                        score = source.get('similarity_score', 0)
                        print(f"  • Page {page} (similarity: {score:.2%})")
                        print(f"    {source['content'][:200]}...")
            
            else:
                # Treat unknown commands as queries
                query_text = user_input
                result = rag_system.query(query_text, n_results=5, use_memory=True)
                
                print("\n🤖 Assistant:")
                print(result['answer'])
                
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for source in result['sources'][:3]:
                        page = source['metadata'].get('page_number', 'N/A')
                        score = source.get('similarity_score', 0)
                        print(f"  • Page {page} (similarity: {score:.2%})")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def main():
    """Main function"""
    print_banner()
    
    # PDF path
    pdf_path = "/home/ahmed-bensalah/sight/wie_rag/data/wie_rag.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print(f"Please make sure the file exists at this location.")
        sys.exit(1)
    
    # Initialize RAG system
    rag_system = MultilingualRAGSystem(
        persist_directory="chromadb_data",
        collection_name="wie_rag_collection"
    )
    
    # Check if we need to load the PDF
    stats = rag_system.get_stats()
    total_docs = stats['vector_store'].get('total_documents', 0)
    
    if total_docs == 0:
        print(f"📥 No documents found in vector store. Loading PDF...\n")
        try:
            num_chunks = rag_system.load_pdf(pdf_path)
            print(f"\n✅ Successfully loaded {num_chunks} chunks from PDF!\n")
        except Exception as e:
            print(f"\n❌ Error loading PDF: {str(e)}")
            sys.exit(1)
    else:
        print(f"✅ Found {total_docs} documents in vector store (already loaded)\n")
        reload = input("Do you want to reload the PDF? (yes/no): ").strip().lower()
        if reload == 'yes':
            rag_system.clear_all()
            num_chunks = rag_system.load_pdf(pdf_path)
            print(f"\n✅ Successfully reloaded {num_chunks} chunks from PDF!\n")
    
    # Start interactive mode
    interactive_mode(rag_system)


if __name__ == "__main__":
    main()
