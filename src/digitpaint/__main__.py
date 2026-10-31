import argparse
import pickle
import sys

import pygame as pg

from digitpaint.manager import PaintManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="File path to model weights")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    manager = PaintManager(model=model)
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(1)
            elif event.type == pg.KEYDOWN:
                manager.on_key(event.key)
        manager.update()


if __name__ == "__main__":
    main()
