"""
Entry point for the Hazard Detection web dashboard.

Usage:
    python run_webapp.py

Then open http://localhost:8000 in your browser.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "webapp.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
