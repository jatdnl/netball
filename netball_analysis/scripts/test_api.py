#!/usr/bin/env python3
"""
Test script for Netball Analysis API
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

async def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Netball Analysis API")
        
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"   Status: {response.json()['status']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return
        
        # Test root endpoint
        print("\n2. Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            if response.status_code == 200:
                print("✅ Root endpoint passed")
                print(f"   Message: {response.json()['message']}")
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
        
        # Test configs endpoint
        print("\n3. Testing configs endpoint...")
        try:
            response = await client.get(f"{base_url}/configs")
            if response.status_code == 200:
                print("✅ Configs endpoint passed")
                configs = response.json()
                print(f"   Available configs: {configs}")
            else:
                print(f"❌ Configs endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Configs endpoint error: {e}")
        
        # Test jobs endpoint
        print("\n4. Testing jobs endpoint...")
        try:
            response = await client.get(f"{base_url}/jobs")
            if response.status_code == 200:
                print("✅ Jobs endpoint passed")
                jobs = response.json()
                print(f"   Current jobs: {len(jobs)}")
            else:
                print(f"❌ Jobs endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Jobs endpoint error: {e}")
        
        print("\n🎉 API test completed!")

def main():
    """Main function."""
    print("Starting API test...")
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: python api/app.py")
    
    try:
        asyncio.run(test_api())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()

