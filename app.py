#!/usr/bin/env python3
"""
SEM Image Analysis Dashboard - Production Entry Point
"""
import os
from dashboard import app

if __name__ == '__main__':
    # Get port from environment variable or default to 8050
    port = int(os.environ.get('PORT', 8050))
    
    # Run the server
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=port
    )
