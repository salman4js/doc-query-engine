import logging

# Setup basic logging config
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",  # 'a' appends to the file (default); 'w' overwrites it on each run
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)