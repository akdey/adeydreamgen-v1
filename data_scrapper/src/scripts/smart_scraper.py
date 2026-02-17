import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_scraper.main import main

if __name__ == "__main__":
    main()
