# Netball Analysis System

An AI-powered computer vision system for automated netball game analysis, designed to comply with MSSS 2025 rules and provide comprehensive insights for coaches, players, and analysts.

## 🏐 Features

- **Player Detection & Tracking**: YOLOv8-based detection with DeepSORT/BYTETrack tracking
- **Ball Tracking**: Kalman filter-based ball trajectory tracking
- **Court Calibration**: Homography-based court mapping and zone enforcement
- **Possession Analysis**: 3-second rule enforcement with finite state machine
- **Shooting Analysis**: Shot detection and goal/miss determination
- **Standings Calculation**: MSSS-compliant league standings with tie-breakers
- **Web Interface**: Streamlit-based frontend for analysis and visualization
- **API Service**: FastAPI backend for programmatic access
- **Data Export**: CSV and JSON export capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd netball_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Validate setup**
   ```bash
   python scripts/validate_setup.py
   ```

4. **Train models** (if weights are missing)
   ```bash
   # Train player detection model
   yolo detect train data=datasets/players.yaml model=yolov8s.pt epochs=80 name=netball_players
   
   # Train ball detection model
   yolo detect train data=datasets/ball.yaml model=yolov8s.pt epochs=120 name=netball_ball
   
   # Train hoop detection model
   yolo detect train data=datasets/hoop.yaml model=yolov8s.pt epochs=100 name=netball_hoop
   ```

5. **Calibrate court homography**
   ```bash
   python scripts/calibrate_homography.py --image path/to/court_image.jpg --mode manual --output configs/homography.yaml
   ```

6. **Run analysis**
   ```bash
   python scripts/run_local.py --video path/to/game.mp4 --config configs/config_netball.json --output output/
   ```

## 📁 Project Structure

```
netball_analysis/
├── app/                    # FastAPI backend
│   ├── api.py             # API endpoints
│   ├── workers.py         # Background workers
│   └── __init__.py
├── core/                   # Core analysis modules
│   ├── detection.py       # Object detection
│   ├── tracking.py        # Player tracking
│   ├── ball_tracker.py    # Ball tracking
│   ├── team_id.py         # Team identification
│   ├── court.py           # Court geometry
│   ├── zones.py           # Zone management
│   ├── homography.py      # Homography calibration
│   ├── analytics.py       # Game analytics
│   ├── shooting_analysis.py # Shooting analysis
│   ├── standings.py       # Standings calculation
│   ├── viz.py             # Visualization
│   ├── io_utils.py        # I/O utilities
│   ├── types.py           # Type definitions
│   └── __init__.py
├── configs/                # Configuration files
│   └── config_netball.json # Main configuration
├── datasets/              # Training datasets
│   ├── players.yaml       # Player dataset config
│   ├── ball.yaml          # Ball dataset config
│   ├── hoop.yaml          # Hoop dataset config
│   └── README.md
├── scripts/               # Utility scripts
│   ├── run_local.py       # Local analysis
│   ├── calibrate_homography.py # Court calibration
│   ├── evaluate_slice.py  # Pipeline evaluation
│   ├── standings.py       # Standings calculation
│   ├── export_csv.py      # CSV export
│   ├── draw_overlays.py   # Video overlay
│   └── validate_setup.py  # Setup validation
├── tests/                 # Unit tests
│   ├── test_homography.py
│   ├── test_zones.py
│   ├── test_possession.py
│   ├── test_shooting.py
│   ├── test_standings.py
│   └── __init__.py
├── web/                   # Streamlit frontend
│   ├── streamlit_app.py   # Main web app
│   ├── static/            # Static assets
│   └── templates/         # HTML templates
├── models/                # Model weights
│   ├── players_best.pt    # Player detection model
│   ├── ball_best.pt       # Ball detection model
│   └── hoop_best.pt       # Hoop detection model
├── output/                # Analysis outputs
│   └── .gitkeep
├── docs/                  # Documentation
│   └── prd.md             # Product Requirements Document
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🔧 Configuration

The system is configured through `configs/config_netball.json`:

```json
{
  "detection": {
    "player_confidence_threshold": 0.5,
    "ball_confidence_threshold": 0.3,
    "hoop_confidence_threshold": 0.7
  },
  "possession": {
    "timeout_seconds": 3.0,
    "transfer_distance": 2.0
  },
  "court": {
    "width": 30.5,
    "height": 15.25,
    "shooting_circle_radius": 4.9
  },
  "game_rules": {
    "duration_minutes": 15,
    "break_duration_minutes": 5,
    "extra_time_minutes": 5,
    "golden_goal_margin": 2
  }
}
```

## 🎯 Usage Examples

### Command Line Analysis

```bash
# Basic analysis
python scripts/run_local.py --video game.mp4 --output results/

# With homography calibration
python scripts/run_local.py --video game.mp4 --homography configs/homography.yaml --output results/

# Limited frames for testing
python scripts/run_local.py --video game.mp4 --max-frames 1000 --output results/
```

### API Usage

```python
import requests

# Start analysis
response = requests.post("http://localhost:8000/analyze", json={
    "video_path": "game.mp4",
    "max_frames": 1000,
    "save_video": True
})

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}")
print(status.json())

# Download results
results = requests.get(f"http://localhost:8000/download/{job_id}/analysis_result.json")
```

### Web Interface

```bash
# Start Streamlit app
streamlit run web/streamlit_app.py

# Start API server
python -m app.api
```

## 📊 Analysis Outputs

The system generates several output files:

- **`analysis_result.json`**: Complete analysis results
- **`events.csv`**: Possession and shot events
- **`player_stats.csv`**: Individual player statistics
- **`standings.csv`**: League standings (if applicable)
- **`summary.txt`**: Analysis summary
- **`analysis_output.mp4`**: Video with overlays (optional)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_homography.py

# Run with coverage
pytest --cov=core tests/
```

## 📈 Performance Benchmarks

The system is designed to meet the following performance criteria:

- **Player Detection**: ≥80% mAP@50
- **Ball Detection**: ≥85% recall on shot/pass frames
- **Processing Speed**: ≥2x real-time
- **ID Switches**: ≤3 per minute
- **Hoop Detection**: ≥95% precision

## 🔍 Evaluation

Evaluate the analysis pipeline:

```bash
python scripts/evaluate_slice.py --video test_clip.mp4 --config configs/config_netball.json --output evaluation/
```

## 📚 Documentation

- **[Product Requirements (Executive Summary)](docs/prd.md)**
- **[Product Requirements (Development-Ready)](docs/prd_enhanced.md)**
- **[API Documentation](http://localhost:8000/docs)**: Interactive API documentation
- **[Configuration Guide](configs/)**: Configuration options and examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 object detection framework
- **DeepSORT**: Multi-object tracking algorithm
- **MSSS**: Malaysian Schools Sports Council for netball rules
- **OpenCV**: Computer vision library
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Rapid web app development

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local regulations and obtain necessary permissions before using in production environments.
