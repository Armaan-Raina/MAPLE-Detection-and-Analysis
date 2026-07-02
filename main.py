import sys
import os

# Make the src/ modules importable by bare name (mea_io, detection, etc.)
sys.path.insert(0, os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "src"))

from event_visualizer import main

if __name__ == "__main__":
    main()
