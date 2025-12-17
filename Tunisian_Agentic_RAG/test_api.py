#!/usr/bin/env python3
"""
Test the API endpoints locally
"""
import os
import sys
import time
import requests
import json

# Set environment variables before importing
os.environ['ENABLE_SYNTHESIS'] = 'true'
os.environ['HUGGINGFACE_API_KEY'] = 'your_api_key'
os.environ['HUGGINGFACE_MODEL'] = 'google/flan-t5-base'
os.environ['HF_MAX_NEW_TOKENS'] = '256'

# Test API endpoints
BASE_URL = "http://localhost:8080"

def test_initialize():
    """Test /initialize endpoint"""
    print("\n" + "="*60)
    print("Testing /initialize endpoint")
    print("="*60)
    try:
        resp = requests.post(f"{BASE_URL}/initialize", timeout=60)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_query(query_text):
    """Test /query endpoint"""
    print("\n" + "="*60)
    print(f"Testing /query endpoint with: {query_text}")
    print("="*60)
    try:
        payload = {"query": query_text}
        resp = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=120
        )
        print(f"Status: {resp.status_code}")
        data = resp.json()
        
        # Pretty print the answer
        print("\n📝 ANSWER:")
        print("-" * 60)
        print(data.get('answer', 'N/A'))
        print("-" * 60)
        
        # Show sources
        sources = data.get('sources', [])
        print(f"\n📚 SOURCES ({len(sources)}):")
        for i, src in enumerate(sources[:3], 1):
            print(f"\n[Source {i}]")
            print(f"  Score: {src.get('similarity_score', 'N/A'):.3f}")
            print(f"  Page: {src.get('metadata', {}).get('page_number', 'N/A')}")
            print(f"  Content: {src.get('content', '')[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health():
    """Test /health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test health first
    test_health()
    
    # Initialize
    test_initialize()
    
    # Give it a moment
    time.sleep(2)
    
    # Test queries
    queries = [
        "كيفاش نجم نعاون طفلي في الدراسة؟",
        "ما هي أفضل الطرق للتعلم؟",
    ]
    
    for q in queries:
        test_query(q)
        time.sleep(2)
