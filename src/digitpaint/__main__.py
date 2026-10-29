import argparse

import joblib
import pygame as pg

from digitpaint.manager import PaintManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="File path to model weights")
    args = parser.parse_args()
    manager = PaintManager(model=joblib.load(args.model), width=560, heading_size=40)
    while True:
        for event in pg.event.get():
            manager.handle_event(event)
        manager.update()


if __name__ == "__main__":
    main()
