"""Streamlit frontend for netball analysis."""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io


# Page configuration
st.set_page_config(
    page_title="Netball Analysis System",
    page_icon="🏐",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"


def main():
    """Main Streamlit application."""
    
    st.title("🏐 Netball Analysis System")
    st.markdown("Analyze netball games with AI-powered computer vision")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload & Analyze", "View Results", "Standings", "Settings"]
    )
    
    if page == "Upload & Analyze":
        upload_and_analyze_page()
    elif page == "View Results":
        view_results_page()
    elif page == "Standings":
        standings_page()
    elif page == "Settings":
        settings_page()


def upload_and_analyze_page():
    """Upload and analyze video page."""
    
    st.header("Upload & Analyze Video")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a netball game video for analysis"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.video(uploaded_file)
        
        # Analysis options
        st.subheader("Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_frames = st.number_input(
                "Max frames to process",
                min_value=1,
                max_value=10000,
                value=1000,
                help="Limit processing to reduce time"
            )
            
            save_video = st.checkbox("Save output video", value=True)
        
        with col2:
            save_overlays = st.checkbox("Save overlay frames", value=True)
            
            # Homography file upload
            homography_file = st.file_uploader(
                "Homography file (optional)",
                type=['json', 'yaml'],
                help="Upload homography calibration file"
            )
        
        # Start analysis button
        if st.button("Start Analysis", type="primary"):
            with st.spinner("Starting analysis..."):
                # Save uploaded file
                video_path = f"temp_{uploaded_file.name}"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Prepare request
                request_data = {
                    "video_path": video_path,
                    "max_frames": max_frames,
                    "save_video": save_video,
                    "save_overlays": save_overlays
                }
                
                # Start analysis
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze", json=request_data)
                    response.raise_for_status()
                    
                    result = response.json()
                    job_id = result["job_id"]
                    
                    st.success("Analysis started successfully!")
                    st.info(f"Job ID: {job_id}")
                    
                    # Monitor progress
                    monitor_analysis(job_id)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to start analysis: {e}")
                
                finally:
                    # Clean up temp file
                    if Path(video_path).exists():
                        Path(video_path).unlink()


def monitor_analysis(job_id: str):
    """Monitor analysis progress."""
    
    st.subheader("Analysis Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/status/{job_id}")
            response.raise_for_status()
            
            status = response.json()
            
            progress_bar.progress(status["progress"])
            status_text.text(f"Status: {status['status']}")
            
            if status["status"] == "completed":
                st.success("Analysis completed!")
                
                # Display outputs
                if status["outputs"]:
                    st.subheader("Analysis Outputs")
                    
                    for output_type, filename in status["outputs"].items():
                        if st.button(f"Download {output_type}"):
                            download_file(job_id, filename)
                
                break
            
            elif status["status"] == "failed":
                st.error(f"Analysis failed: {status.get('error', 'Unknown error')}")
                break
            
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to get status: {e}")
            break


def download_file(job_id: str, filename: str):
    """Download analysis output file."""
    try:
        response = requests.get(f"{API_BASE_URL}/download/{job_id}/{filename}")
        response.raise_for_status()
        
        # Save file
        output_path = Path("downloads") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        st.success(f"File downloaded: {output_path}")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download file: {e}")


def view_results_page():
    """View analysis results page."""
    
    st.header("View Analysis Results")
    
    # Job selection
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        response.raise_for_status()
        
        jobs = response.json()["jobs"]
        
        if not jobs:
            st.info("No analysis jobs found")
            return
        
        selected_job = st.selectbox("Select a job", jobs)
        
        if selected_job:
            # Get job status
            response = requests.get(f"{API_BASE_URL}/status/{selected_job}")
            response.raise_for_status()
            
            status = response.json()
            
            if status["status"] == "completed":
                display_results(selected_job, status["outputs"])
            else:
                st.info(f"Job status: {status['status']}")
                
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get jobs: {e}")


def display_results(job_id: str, outputs: dict):
    """Display analysis results."""
    
    st.subheader("Analysis Results")
    
    # Download and display summary
    if "summary" in outputs:
        try:
            response = requests.get(f"{API_BASE_URL}/download/{job_id}/summary.txt")
            response.raise_for_status()
            
            summary_text = response.text
            st.text_area("Summary", summary_text, height=200)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load summary: {e}")
    
    # Display events
    if "events" in outputs:
        try:
            response = requests.get(f"{API_BASE_URL}/download/{job_id}/events.csv")
            response.raise_for_status()
            
            # Parse CSV
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            st.subheader("Events")
            st.dataframe(df)
            
            # Create visualizations
            create_event_visualizations(df)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load events: {e}")
    
    # Display player stats
    if "player_stats" in outputs:
        try:
            response = requests.get(f"{API_BASE_URL}/download/{job_id}/player_stats.csv")
            response.raise_for_status()
            
            # Parse CSV
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            st.subheader("Player Statistics")
            st.dataframe(df)
            
            # Create visualizations
            create_player_visualizations(df)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load player stats: {e}")


def create_event_visualizations(df: pd.DataFrame):
    """Create event visualizations."""
    
    st.subheader("Event Visualizations")
    
    # Event timeline
    if 'timestamp' in df.columns:
        fig = px.scatter(df, x='timestamp', y='event_type', 
                        color='team', title='Event Timeline')
        st.plotly_chart(fig, use_container_width=True)
    
    # Event counts
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts()
        fig = px.pie(values=event_counts.values, names=event_counts.index,
                    title='Event Distribution')
        st.plotly_chart(fig, use_container_width=True)


def create_player_visualizations(df: pd.DataFrame):
    """Create player visualizations."""
    
    st.subheader("Player Visualizations")
    
    # Possession time
    if 'possession_time' in df.columns:
        fig = px.bar(df, x='player_id', y='possession_time',
                    title='Possession Time by Player')
        st.plotly_chart(fig, use_container_width=True)
    
    # Shots attempted vs made
    if 'shots_attempted' in df.columns and 'shots_made' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Shots Attempted', x=df['player_id'], y=df['shots_attempted']))
        fig.add_trace(go.Bar(name='Shots Made', x=df['player_id'], y=df['shots_made']))
        fig.update_layout(title='Shooting Statistics by Player', barmode='group')
        st.plotly_chart(fig, use_container_width=True)


def standings_page():
    """Standings page."""
    
    st.header("League Standings")
    
    # Category selection
    category = st.selectbox("Select Category", ["U12", "U15", "U18"])
    
    # Sample standings data
    sample_standings = pd.DataFrame({
        'Position': [1, 2, 3, 4],
        'Team': ['Team A', 'Team B', 'Team C', 'Team D'],
        'Games Played': [3, 3, 3, 3],
        'Wins': [3, 2, 1, 0],
        'Draws': [0, 0, 0, 0],
        'Losses': [0, 1, 2, 3],
        'Goals For': [45, 38, 32, 28],
        'Goals Against': [28, 32, 38, 45],
        'Goal Difference': [17, 6, -6, -17],
        'Goal Average': [1.61, 1.19, 0.84, 0.62],
        'Points': [6, 4, 2, 0]
    })
    
    st.dataframe(sample_standings)
    
    # Standings visualization
    fig = px.bar(sample_standings, x='Team', y='Points',
                title=f'{category} League Standings')
    st.plotly_chart(fig, use_container_width=True)


def settings_page():
    """Settings page."""
    
    st.header("Settings")
    
    # API configuration
    st.subheader("API Configuration")
    
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="Base URL for the analysis API"
    )
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            response.raise_for_status()
            st.success("API connection successful!")
        except requests.exceptions.RequestException as e:
            st.error(f"API connection failed: {e}")
    
    # Analysis settings
    st.subheader("Analysis Settings")
    
    default_config = {
        "player_confidence_threshold": 0.5,
        "ball_confidence_threshold": 0.3,
        "hoop_confidence_threshold": 0.7,
        "possession_timeout_seconds": 3.0
    }
    
    config = {}
    for key, default_value in default_config.items():
        config[key] = st.number_input(
            key.replace("_", " ").title(),
            value=default_value,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )
    
    if st.button("Save Settings"):
        st.success("Settings saved!")


if __name__ == "__main__":
    main()


