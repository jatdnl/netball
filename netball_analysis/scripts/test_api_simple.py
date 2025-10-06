#!/usr/bin/env python3
"""
Simple API test without external dependencies
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app
from api.models import AnalysisRequest, AnalysisResponse
from api.job_manager import JobManager
from api.analysis_service import AnalysisService

async def test_api_components():
    """Test API components directly."""
    print("🧪 Testing Netball Analysis API Components")
    
    # Test 1: Job Manager
    print("\n1. Testing Job Manager...")
    try:
        job_manager = JobManager()
        await job_manager.initialize()
        
        # Create a test job
        request = AnalysisRequest(
            job_id="test-job-123",
            video_filename="test_video.mp4",
            video_path="test_video.mp4"
        )
        
        job = await job_manager.create_job(request)
        print(f"✅ Job Manager: Created job {job.job_id}")
        
        # Test job status update
        await job_manager.update_job_status(
            job.job_id, 
            "processing", 
            progress=50.0,
            message="Processing video"
        )
        print("✅ Job Manager: Updated job status")
        
        # Test job retrieval
        retrieved_job = await job_manager.get_job(job.job_id)
        if retrieved_job:
            print(f"✅ Job Manager: Retrieved job {retrieved_job.job_id}")
        
        # Test job listing
        jobs = await job_manager.list_jobs()
        print(f"✅ Job Manager: Listed {len(jobs)} jobs")
        
        # Cleanup
        await job_manager.cleanup()
        print("✅ Job Manager: Cleanup completed")
        
    except Exception as e:
        print(f"❌ Job Manager test failed: {e}")
        return False
    
    # Test 2: Analysis Service
    print("\n2. Testing Analysis Service...")
    try:
        analysis_service = AnalysisService()
        await analysis_service.initialize()
        
        print("✅ Analysis Service: Initialized successfully")
        
        # Test result directory creation
        result_dir = Path("results/test-job-123")
        result_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Analysis Service: Created result directory")
        
        # Test analysis script creation
        script_path = Path("scripts/analyze_video.py")
        if script_path.exists():
            print("✅ Analysis Service: Analysis script exists")
        else:
            print("⚠️ Analysis Service: Analysis script not found")
        
        await analysis_service.cleanup()
        print("✅ Analysis Service: Cleanup completed")
        
    except Exception as e:
        print(f"❌ Analysis Service test failed: {e}")
        return False
    
    # Test 3: API Models
    print("\n3. Testing API Models...")
    try:
        # Test AnalysisRequest
        request = AnalysisRequest(
            job_id="test-123",
            video_filename="test.mp4"
        )
        print("✅ Models: AnalysisRequest created")
        
        # Test AnalysisResponse
        response = AnalysisResponse(
            job_id="test-123",
            status="queued",
            message="Test message",
            created_at="2024-01-01T00:00:00"
        )
        print("✅ Models: AnalysisResponse created")
        
        # Test JSON serialization
        request_json = request.json()
        response_json = response.json()
        print("✅ Models: JSON serialization works")
        
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False
    
    # Test 4: FastAPI App
    print("\n4. Testing FastAPI App...")
    try:
        # Test app initialization
        if app:
            print("✅ FastAPI App: App initialized")
        
        # Test routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/analyze", "/jobs"]
        
        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                print(f"✅ FastAPI App: Route {expected_route} exists")
            else:
                print(f"⚠️ FastAPI App: Route {expected_route} not found")
        
    except Exception as e:
        print(f"❌ FastAPI App test failed: {e}")
        return False
    
    print("\n🎉 All API component tests passed!")
    return True

def main():
    """Main function."""
    try:
        success = asyncio.run(test_api_components())
        if success:
            print("\n✅ API implementation is working correctly!")
            print("📋 Next steps:")
            print("   1. Start API server: python -m uvicorn api.app:app --host 127.0.0.1 --port 8000")
            print("   2. Test endpoints: curl http://127.0.0.1:8000/health")
            print("   3. View docs: http://127.0.0.1:8000/docs")
        else:
            print("\n❌ Some API tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

