# SEM Image Analysis Dashboard

A web-based dashboard for analyzing Scanning Electron Microscope (SEM) images with porosity analysis, region detection, and data logging capabilities.

## Features

- **Image Upload & Cropping**: Upload SEM images and crop regions of interest
- **Threshold Analysis**: Adjust threshold values for binary image processing
- **Area Distribution**: Analyze and visualize pore size distributions
- **Data Logging**: Save analysis results with export capabilities
- **Row Management**: Delete unwanted entries from analysis log

## Deployment Options

### Option 1: Heroku (Recommended)

1. Create a Heroku account at https://heroku.com
2. Install Heroku CLI
3. Deploy:
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit"

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku master
```

### Option 2: Render

1. Create account at https://render.com
2. Connect your GitHub repository
3. Create new Web Service
4. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn dashboard:server`

### Option 3: Railway

1. Create account at https://railway.app
2. Deploy from GitHub
3. No additional configuration needed

### Option 4: Local Network Access

Run locally but make accessible to your network:
```bash
python dashboard.py
```
Access via: http://your-local-ip:5050

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python dashboard.py
```

## Usage

1. Upload your SEM image
2. Optionally crop the region of interest
3. Adjust the threshold for binary image processing
4. Review the area distribution analysis
5. Save the analysis results to the log
6. Export data as CSV when needed

## Requirements

- Python 3.11+
- See `requirements.txt` for package dependencies

## License

This project is for academic and research use.
