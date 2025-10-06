"""Enhanced Streamlit frontend for netball analysis with Sprint 8 improvements."""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pathlib import Path
import io
import base64
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Netball Analysis System - Enhanced",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    st.markdown('<h1 class="main-header">🏐 Netball Analysis System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Computer Vision for Netball Game Analysis")
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Dashboard"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Navigation")
        page = st.selectbox(
            "Choose a page",
            ["🏠 Dashboard", "Upload & Analyze", "Results & Analytics", "Standings", "⚙️ Settings"],
            index=["🏠 Dashboard", "Upload & Analyze", "Results & Analytics", "Standings", "⚙️ Settings"].index(st.session_state.page)
        )
        
        # Update session state when selectbox changes
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        display_system_status()
    
    # Route to appropriate page
    if st.session_state.page == "🏠 Dashboard":
        dashboard_page()
    elif st.session_state.page == "Upload & Analyze":
        upload_and_analyze_page()
    elif st.session_state.page == "Results & Analytics":
        results_analytics_page()
    elif st.session_state.page == "Standings":
        standings_page()
    elif st.session_state.page == "⚙️ Settings":
        settings_page()

def display_system_status():
    """Display system status in sidebar."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("🟢 API Online")
            st.metric("Status", health_data.get("status", "unknown"))
            # Show brief calibration status if present
            calib = health_data.get("calibration_telemetry") or {}
            if calib:
                st.caption("Calibration")
                st.metric("Calibrated", "Yes" if calib.get("is_calibrated") else "No")
        else:
            st.error("🔴 API Offline")
    except:
        st.error("🔴 API Unreachable")

def dashboard_page():
    """Enhanced dashboard with system overview."""
    st.header("📊 System Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        response = requests.get(f"{API_BASE_URL}/processor/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            system = metrics.get("system", {})
            calib = metrics.get("calibration_telemetry", {})
            
            with col2:
                st.metric("CPU Usage", f"{system.get('cpu_percent', 0.0):.1f}%")
            with col3:
                mem = system.get('memory', {})
                st.metric("Memory Usage", f"{mem.get('percent', 0.0):.1f}%")
            with col4:
                disk = system.get('disk', {})
                used = disk.get('used', 0)
                total = disk.get('total', 1)
                free_gb = max(total - used, 0) / (1024**3)
                st.metric("Disk Free", f"{free_gb:.1f} GB")

            # Calibration health card
            st.subheader("🔧 Calibration Health")
            render_calibration_health(calib)
        else:
            st.warning("Unable to fetch system metrics")
    except Exception as _e:
        st.warning("Unable to fetch system metrics")
    
    # Recent jobs
    st.subheader("📋 Recent Analysis Jobs")
    display_recent_jobs()
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload New Video", use_container_width=True):
            st.session_state.page = "Upload & Analyze"
            st.rerun()
    
    with col2:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.page = "Results & Analytics"
            st.rerun()
    
    with col3:
        if st.button("📈 Check Standings", use_container_width=True):
            st.session_state.page = "Standings"
            st.rerun()

def display_recent_jobs():
    """Display recent analysis jobs."""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs?limit=5")
        if response.status_code == 200:
            jobs = response.json()
            if jobs:
                # Create DataFrame for display
                job_data = []
                for job in jobs:
                    job_data.append({
                        "Job ID": job["job_id"][:8] + "...",
                        "Status": job["status"].title(),
                        "Progress": f"{job['progress']:.1f}%",
                        "Created": job["created_at"][:19].replace("T", " "),
                        "Message": job["message"][:50] + "..." if len(job["message"]) > 50 else job["message"]
                    })
                
                df = pd.DataFrame(job_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No analysis jobs found")
        else:
            st.error("Failed to fetch jobs")
    except:
        st.error("Unable to fetch recent jobs")

def upload_and_analyze_page():
    """Enhanced upload and analyze page."""
    st.header("📤 Upload & Analyze Video")
    
    # File upload section
    st.subheader("📁 Video Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a netball game video for analysis. Supported formats: MP4, AVI, MOV, MKV, WebM"
    )
    
    if uploaded_file is not None:
        # Display video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.subheader("📋 File Information")
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
            st.metric("File Type", uploaded_file.type)
            st.metric("File Name", uploaded_file.name)
        
        # Analysis configuration
        st.subheader("⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Processing Options**")
            enable_possession = st.checkbox("Enable Possession Tracking", value=True)
            enable_shooting = st.checkbox("Enable Shooting Analysis", value=True)
            enable_zones = st.checkbox("Enable Zone Violation Detection", value=True)
            
            start_time = st.number_input(
                "Start Time (seconds)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Start analysis from this time in the video"
            )
        
        with col2:
            st.markdown("**Output Options**")
            end_time = st.number_input(
                "End Time (seconds)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="End analysis at this time (0 = end of video)"
            )
            
            config_file = st.selectbox(
                "Configuration",
                ["config_netball.json", "config_high_accuracy.json", "config_fast_processing.json"],
                help="Select analysis configuration"
            )
        
        # Start analysis
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            start_analysis(uploaded_file, {
                "enable_possession": enable_possession,
                "enable_shooting": enable_shooting,
                "enable_zones": enable_zones,
                "start_time": start_time if start_time > 0 else None,
                "end_time": end_time if end_time > 0 else None,
                "config": config_file
            })

def start_analysis(uploaded_file, options):
    """Start analysis with enhanced progress monitoring."""
    with st.spinner("🚀 Starting analysis..."):
        try:
            # Prepare file for upload
            files = {"video_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {k: v for k, v in options.items() if v is not None}
            
            # Start analysis
            response = requests.post(f"{API_BASE_URL}/analyze", files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            
            st.success("✅ Analysis started successfully!")
            st.info(f"Job ID: `{job_id}`")
            
            # Enhanced progress monitoring
            monitor_analysis_enhanced(job_id)
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to start analysis: {e}")

def monitor_analysis_enhanced(job_id: str):
    """Enhanced analysis progress monitoring."""
    st.subheader("📊 Analysis Progress")
    
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    metrics_container = st.container()
    
    # Initialize progress tracking
    progress_bar = progress_container.progress(0)
    status_text = status_container.empty()
    metrics_col1, metrics_col2, metrics_col3 = metrics_container.columns(3)
    
    # Progress tracking variables
    start_time = time.time()
    last_progress = 0
    
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/status")
            response.raise_for_status()
            
            status = response.json()
            current_progress = status["progress"]
            
            # Update progress bar
            progress_bar.progress(current_progress / 100)
            
            # Update status
            status_text.markdown(f"""
            <div class="metric-card">
                <strong>Status:</strong> {status['status'].title()}<br>
                <strong>Progress:</strong> {current_progress:.1f}%<br>
                <strong>Message:</strong> {status['message']}
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate and display metrics
            elapsed_time = time.time() - start_time
            if current_progress > last_progress:
                estimated_total = (elapsed_time / current_progress * 100) if current_progress > 0 else 0
                estimated_remaining = max(0, estimated_total - elapsed_time)
            else:
                estimated_remaining = 0
            
            with metrics_col1:
                st.metric("Elapsed Time", f"{elapsed_time:.0f}s")
            
            with metrics_col2:
                st.metric("Estimated Remaining", f"{estimated_remaining:.0f}s")
            
            with metrics_col3:
                st.metric("Progress Rate", f"{current_progress - last_progress:.1f}%/2s")
            
            last_progress = current_progress
            
            # Check completion
            if status["status"] == "completed":
                st.success("🎉 Analysis completed successfully!")
                
                # Display completion summary
                display_completion_summary(job_id)
                break
            
            elif status["status"] == "failed":
                st.error(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
                break
            
            elif status["status"] == "cancelled":
                st.warning("⚠️ Analysis was cancelled")
                break
            
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to get status: {e}")
            break

def display_completion_summary(job_id: str):
    """Display analysis completion summary."""
    st.subheader("📋 Analysis Summary")
    
    try:
        # Get results
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/result")
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                data = result["data"]
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Players Detected", data["detection_stats"]["total_players"])
                
                with col2:
                    st.metric("Possession Changes", data["possession_stats"]["total_possessions"])
                
                with col3:
                    st.metric("Shot Attempts", data["shooting_stats"]["total_shots"])
                
                with col4:
                    st.metric("Zone Violations", data["zone_stats"]["total_violations"])
                
                # Download options
                st.subheader("📥 Download Results")
                display_download_options(job_id)
                
            else:
                st.error("Failed to get analysis results")
        else:
            st.error("Failed to fetch results")
    except:
        st.error("Unable to fetch analysis summary")

def display_download_options(job_id: str):
    """Display download options for completed analysis."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📹 Download Video", use_container_width=True):
            download_file(job_id, "video")
    
    with col2:
        if st.button("📊 Download CSV", use_container_width=True):
            download_file(job_id, "csv")
    
    with col3:
        if st.button("📄 Download JSON", use_container_width=True):
            download_file(job_id, "json")
    
    with col4:
        if st.button("📦 Download Archive", use_container_width=True):
            download_file(job_id, "archive")

def download_file(job_id: str, file_type: str):
    """Download analysis file."""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/download/{file_type}")
        response.raise_for_status()
        
        # Get filename from response headers
        filename = f"{job_id}_{file_type}.{file_type}"
        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
        
        # Create download button
        st.download_button(
            label=f"📥 Download {filename}",
            data=response.content,
            file_name=filename,
            mime=response.headers.get('content-type', 'application/octet-stream')
        )
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to download file: {e}")

def results_analytics_page():
    """Enhanced results and analytics page."""
    st.header("📊 Results & Analytics")
    
    # Job selection
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        if response.status_code == 200:
            jobs = response.json()
            if jobs:
                # Filter completed jobs
                completed_jobs = [job for job in jobs if job["status"] == "completed"]
                
                if completed_jobs:
                    selected_job = st.selectbox(
                        "Select a completed analysis",
                        completed_jobs,
                        format_func=lambda x: f"{x['job_id'][:8]}... - {x['created_at'][:19].replace('T', ' ')}"
                    )
                    
                    if selected_job:
                        display_enhanced_results(selected_job["job_id"])
                else:
                    st.info("No completed analyses found")
            else:
                st.info("No analysis jobs found")
        else:
            st.error("Failed to fetch jobs")
    except:
        st.error("Unable to fetch analysis jobs")

def display_enhanced_results(job_id: str):
    """Display enhanced results with visualizations."""
    try:
        # Get results
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/result")
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                data = result["data"]
                summary = result.get("summary", {})
                
                # Results overview
                st.subheader("📋 Analysis Overview")
                display_results_overview(data, summary)
                
                # Detailed analytics
                st.subheader("📊 Detailed Analytics")
                display_detailed_analytics(data)
                
                # Interactive visualizations
                st.subheader("📈 Interactive Visualizations")
                display_interactive_visualizations(data)
                
            else:
                st.error("Failed to get analysis results")
        else:
            st.error("Failed to fetch results")
    except:
        st.error("Unable to fetch analysis results")

def display_results_overview(data, summary):
    """Display results overview with key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Detection Accuracy",
            f"{data['detection_stats']['detection_rate']*100:.1f}%",
            help="Percentage of frames with successful detections"
        )
    
    with col2:
        st.metric(
            "Possession Accuracy",
            f"{data['possession_stats']['possession_accuracy']*100:.1f}%",
            help="Accuracy of possession tracking"
        )
    
    with col3:
        st.metric(
            "Shooting Accuracy",
            f"{data['shooting_stats']['shooting_accuracy']:.1f}%",
            help="Percentage of successful shots"
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{data['performance_metrics']['total_processing_time']:.1f}s",
            help="Total time to process the video"
        )

def display_detailed_analytics(data):
    """Display detailed analytics with charts."""
    # Detection analytics
    st.subheader("🎯 Detection Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection confidence distribution
        fig = go.Figure(data=[
            go.Bar(
                x=['Players', 'Balls', 'Hoops'],
                y=[
                    data['detection_stats']['total_players'],
                    data['detection_stats']['total_balls'],
                    data['detection_stats']['total_hoops']
                ],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])
        fig.update_layout(title="Objects Detected", xaxis_title="Object Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average confidence
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number",
                value=data['detection_stats']['avg_confidence']*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Detection Confidence"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            )
        ])
        st.plotly_chart(fig, use_container_width=True)
    
    # Possession analytics
    st.subheader("🏀 Possession Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Possession duration distribution
        fig = go.Figure(data=[
            go.Histogram(
                x=[data['possession_stats']['avg_possession_duration']] * 10,
                nbinsx=10,
                marker_color='lightblue'
            )
        ])
        fig.update_layout(title="Possession Duration Distribution", xaxis_title="Duration (seconds)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 3-second violations
        fig = go.Figure(data=[
            go.Pie(
                labels=['Normal Possessions', '3-Second Violations'],
                values=[
                    data['possession_stats']['total_possessions'] - data['possession_stats']['three_second_violations'],
                    data['possession_stats']['three_second_violations']
                ],
                marker_colors=['lightgreen', 'red']
            )
        ])
        fig.update_layout(title="Possession Compliance")
        st.plotly_chart(fig, use_container_width=True)

def display_interactive_visualizations(data):
    """Display interactive visualizations."""
    # Shooting analytics
    st.subheader("🎯 Shooting Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Shot success rate
        fig = go.Figure(data=[
            go.Bar(
                x=['Goals', 'Misses'],
                y=[data['shooting_stats']['goals_scored'], data['shooting_stats']['shots_missed']],
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(title="Shot Outcomes", xaxis_title="Outcome", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average shot distance
        fig = go.Figure(data=[
            go.Indicator(
                mode="number",
                value=data['shooting_stats']['avg_shot_distance'],
                title={'text': "Avg Shot Distance (m)"},
                number={'suffix': "m"}
            )
        ])
        st.plotly_chart(fig, use_container_width=True)
    
    # Zone violation analytics
    st.subheader("🚫 Zone Violation Analytics")
    
    if data['zone_stats']['violations_by_type']:
        fig = go.Figure(data=[
            go.Bar(
                x=list(data['zone_stats']['violations_by_type'].keys()),
                y=list(data['zone_stats']['violations_by_type'].values()),
                marker_color='orange'
            )
        ])
        fig.update_layout(title="Violations by Type", xaxis_title="Violation Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No zone violations detected")

def standings_page():
    """Enhanced standings page."""
    st.header("📈 League Standings")
    
    # Category selection
    category = st.selectbox("Select Category", ["U12", "U15", "U18", "Open"])
    
    # Sample standings data (in real implementation, this would come from API)
    sample_standings = pd.DataFrame({
        'Position': [1, 2, 3, 4, 5, 6],
        'Team': ['Thunder', 'Lightning', 'Storm', 'Rain', 'Wind', 'Cloud'],
        'Games Played': [5, 5, 5, 5, 5, 5],
        'Wins': [5, 4, 3, 2, 1, 0],
        'Draws': [0, 0, 0, 0, 0, 0],
        'Losses': [0, 1, 2, 3, 4, 5],
        'Goals For': [65, 58, 52, 45, 38, 32],
        'Goals Against': [32, 38, 45, 52, 58, 65],
        'Goal Difference': [33, 20, 7, -7, -20, -33],
        'Goal Average': [2.03, 1.53, 1.16, 0.87, 0.66, 0.49],
        'Points': [10, 8, 6, 4, 2, 0]
    })
    
    # Display standings table
    st.subheader(f"{category} League Standings")
    st.dataframe(sample_standings, use_container_width=True)
    
    # Standings visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Points chart
        fig = px.bar(sample_standings, x='Team', y='Points',
                    title=f'{category} League Points',
                    color='Points',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Goal difference chart
        fig = px.bar(sample_standings, x='Team', y='Goal Difference',
                    title=f'{category} Goal Difference',
                    color='Goal Difference',
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    # League statistics
    st.subheader("📊 League Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Teams", len(sample_standings))
    
    with col2:
        st.metric("Total Games", sample_standings['Games Played'].sum() // 2)
    
    with col3:
        st.metric("Total Goals", sample_standings['Goals For'].sum())
    
    with col4:
        st.metric("Avg Goals/Game", f"{sample_standings['Goals For'].sum() / (sample_standings['Games Played'].sum() // 2):.1f}")

def settings_page():
    """Enhanced settings page."""
    st.header("⚙️ Settings")
    
    # API configuration
    st.subheader("🔧 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="Base URL for the analysis API"
        )
        
        if st.button("🔍 Test Connection"):
            test_api_connection(api_url)
    
    with col2:
        # Display current API status
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("🟢 API Connection Active")
            else:
                st.error("🔴 API Connection Failed")
        except:
            st.error("🔴 API Unreachable")
    
    # Analysis settings
    st.subheader("🎯 Analysis Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["Detection", "Tracking", "Calibration", "Features"])
    
    with tab1:
        st.markdown("**Detection Thresholds**")
        detection_config = {
            "player_confidence_threshold": st.slider("Player Confidence", 0.0, 1.0, 0.5, 0.05),
            "ball_confidence_threshold": st.slider("Ball Confidence", 0.0, 1.0, 0.3, 0.05),
            "hoop_confidence_threshold": st.slider("Hoop Confidence", 0.0, 1.0, 0.7, 0.05)
        }
    
    with tab2:
        st.markdown("**Tracking Parameters**")
        tracking_config = {
            "max_disappeared_frames": st.number_input("Max Disappeared Frames", 1, 100, 30),
            "max_distance": st.number_input("Max Distance", 10.0, 100.0, 50.0, 5.0),
            "confidence_hysteresis": st.slider("Confidence Hysteresis", 0.0, 0.2, 0.05, 0.01)
        }
    
    with tab3:
        st.markdown("**Calibration Settings**")
        colA, colB = st.columns(2)
        with colA:
            validation_threshold = st.slider("Validation Threshold", 0.5, 1.0, 0.95, 0.05)
            cache_enabled = st.checkbox("Enable Calibration Cache", value=True)
            fallback_method = st.selectbox("Fallback Method", ["manual", "automatic"])
        with colB:
            st.markdown("**Auto-Recalibration**")
            enable_autorecalibrate = st.checkbox("Enable Auto-Recalibration", value=True, help="Automatically re-calibrate when drift is detected from court markings/hoops")
            check_interval_frames = st.number_input("Check Interval (frames)", min_value=1, max_value=300, value=30, step=1)
            drift_threshold_pixels = st.number_input("Drift Threshold (pixels)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
            min_hoop_detections = st.number_input("Min Hoop Detections", min_value=1, max_value=2, value=1, step=1)

        calibration_config = {
            "validation_threshold": validation_threshold,
            "cache_enabled": cache_enabled,
            "fallback_method": fallback_method,
            "enable_autorecalibrate": enable_autorecalibrate,
            "check_interval_frames": int(check_interval_frames),
            "drift_threshold_pixels": float(drift_threshold_pixels),
            "min_hoop_detections": int(min_hoop_detections)
        }
    
    with tab4:
        st.markdown("**Feature Toggles**")
        feature_config = {
            "enable_possession": st.checkbox("Enable Possession Tracking", value=True),
            "enable_shooting": st.checkbox("Enable Shooting Analysis", value=True),
            "enable_zones": st.checkbox("Enable Zone Violation Detection", value=True),
            "enable_ocr": st.checkbox("Enable Player Number OCR", value=True)
        }
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        # Combine all configs
        full_config = {
            "detection": detection_config,
            "tracking": tracking_config,
            "calibration": calibration_config,
            "features": feature_config
        }
        
        # Save to file (in real implementation, this would be saved to API)
        config_path = Path("configs/custom_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        st.success("✅ Settings saved successfully!")
    
    # System information
    st.subheader("ℹ️ System Information")
    try:
        response = requests.get(f"{api_url}/processor/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            system = metrics.get('system', {})
            calib = metrics.get('calibration_telemetry', {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**System Resources**")
                st.metric("CPU Usage", f"{system.get('cpu_percent', 0.0):.1f}%")
                mem = system.get('memory', {})
                st.metric("Memory Usage", f"{mem.get('percent', 0.0):.1f}%")
                disk = system.get('disk', {})
                used = disk.get('used', 0)
                total = disk.get('total', 1)
                free_gb = max(total - used, 0) / (1024**3)
                st.metric("Disk Free", f"{free_gb:.1f} GB")
            with col2:
                st.markdown("**Calibration Health**")
                render_calibration_health(calib)
        else:
            st.warning("Unable to fetch system information")
    except Exception:
        st.warning("Unable to fetch system information")

def render_calibration_health(calib: dict):
    """Render calibration telemetry block."""
    if not calib:
        st.info("No calibration telemetry available")
        return
    is_cal = calib.get('is_calibrated')
    stats = calib.get('statistics', {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Calibrated", "Yes" if is_cal else "No")
    with c2:
        st.metric("Attempts", stats.get('total_attempts', 0))
    with c3:
        st.metric("Avg Accuracy", f"{stats.get('average_accuracy', 0.0)*100:.1f}%")

def test_api_connection(api_url: str):
    """Test API connection."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API connection successful!")
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except requests.exceptions.Timeout:
        st.error("❌ API connection timeout")
    except requests.exceptions.ConnectionError:
        st.error("❌ API connection failed")
    except Exception as e:
        st.error(f"❌ API connection error: {e}")

if __name__ == "__main__":
    main()
