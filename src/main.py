import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger("Main")


def main():
    logger.info("AC Sim starting")

    config = load_config()
    logger.info(
        "Config loaded: FPS=%d, Grid=%dx%dx%d",
        config["fps"],
        config["grid"]["x"],
        config["grid"]["y"],
        config["grid"]["z"],
    )
    logger.debug(
        "Room=%.1fx%.1fx%.1fm, Temp=%.1f/%.1fC",
        config["room"]["width"],
        config["room"]["length"],
        config["room"]["height"],
        config["temperature"]["indoor"],
        config["temperature"]["outdoor"],
    )

    # TODO: Init UI, simulation, renderer (Tasks 2-6)

    logger.info("AC Sim ready")


if __name__ == "__main__":
    main()
