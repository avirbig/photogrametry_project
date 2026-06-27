"""
main.py  —  entry point for the Photogrammetry Point Marker tool.

Run:
    python main.py
"""
import os
import sys

# Make the src/ package importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mark_points import PointMarkerApp

IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "building_pictures")
JSON_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "correspondences.json")

if __name__ == "__main__":
    app = PointMarkerApp(IMAGE_FOLDER, JSON_PATH)
    app.run()
