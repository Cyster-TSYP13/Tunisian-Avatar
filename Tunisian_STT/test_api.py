#!/usr/bin/env python3
"""
Test script for the STT API
Usage: python test_api.py <api_url> <audio_file.wav>
Example: python test_api.py http://localhost:8000 felfel0.wav
"""

import sys
import requests
import json

def test_health(api_url):
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def test_root(api_url):
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(api_url)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def test_transcribe(api_url, audio_file):
    """Test the transcribe endpoint"""
    print(f"Testing transcribe endpoint with {audio_file}...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response = requests.post(f"{api_url}/transcribe", files=files)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response:")
        print(f"  Transcript: {result.get('transcript', '')}")
        print(f"  Latency: {result.get('latency_seconds', 0)} seconds")
        print(f"  Filename: {result.get('filename', '')}\n")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"Error: File '{audio_file}' not found\n")
        return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def test_transcribe_streaming(api_url, audio_file):
    """Test the streaming transcribe endpoint"""
    print(f"Testing streaming transcribe endpoint with {audio_file}...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response = requests.post(f"{api_url}/transcribe/streaming", files=files)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response:")
        print(f"  Final Transcript: {result.get('final_transcript', '')}")
        print(f"  Partial Results: {result.get('partial_results', [])}")
        print(f"  Latency: {result.get('latency_seconds', 0)} seconds")
        print(f"  Filename: {result.get('filename', '')}\n")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"Error: File '{audio_file}' not found\n")
        return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <api_url> [audio_file.wav]")
        print(f"Example: python {sys.argv[0]} http://localhost:8000 felfel0.wav")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Testing API at: {api_url}\n")
    print("=" * 60)
    
    # Test basic endpoints
    results = []
    results.append(("Health Check", test_health(api_url)))
    results.append(("Root Endpoint", test_root(api_url)))
    
    # Test transcription if audio file provided
    if audio_file:
        results.append(("Transcribe", test_transcribe(api_url, audio_file)))
        results.append(("Transcribe Streaming", test_transcribe_streaming(api_url, audio_file)))
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    sys.exit(0 if total_passed == len(results) else 1)

if __name__ == "__main__":
    main()
