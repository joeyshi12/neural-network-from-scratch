import numpy as np
import pygame as pg

from neural_network import MLPClassifier


class PaintManager:
    WIDTH = 560
    HEADING_SIZE = 40
    BRUSH_RADIUS: int = 24
    UNIT_LENGTH: int = 4

    def __init__(self, model: MLPClassifier):
        pg.init()
        pg.display.set_caption("Digit Recognition")
        self.font = pg.font.SysFont("lucidaconsole", 16, bold=True)
        self.surface = pg.display.set_mode((self.WIDTH, self.WIDTH + self.HEADING_SIZE))
        self.clock = pg.time.Clock()
        self.model = model

        self.surface.fill((255, 255, 255))  # fill surface with white pixels
        pg.draw.rect(
            self.surface, (160, 160, 160), (0, 0, self.WIDTH, self.HEADING_SIZE)
        )  # fill heading area with grey pixels
        self.canvas = np.zeros(
            (self.WIDTH // self.UNIT_LENGTH, self.WIDTH // self.UNIT_LENGTH)
        )

    def on_key(self, key: int) -> None:
        if key == pg.K_SPACE:
            self.predict()
        elif key == pg.K_c:
            self.clear()

    def update(self):
        if pg.mouse.get_pressed()[0]:
            x, y = pg.mouse.get_pos()
            row = (y - self.HEADING_SIZE) // self.UNIT_LENGTH
            col = x // self.UNIT_LENGTH
            self.draw(row, col)
        pg.display.flip()
        self.clock.tick(0)

    def draw(self, row: int, col: int):
        rows, cols = self.canvas.shape
        for i in range(-self.BRUSH_RADIUS, self.BRUSH_RADIUS + 1):
            for j in range(-self.BRUSH_RADIUS, self.BRUSH_RADIUS + 1):
                if 0 > row + i or row + i >= rows or 0 > col + j or col + j >= cols:
                    continue
                if i**2 + j**2 > self.BRUSH_RADIUS:
                    continue
                if self.canvas[row + i][col + j]:
                    continue
                self.canvas[row + i][col + j] = 1
                rect = (
                    self.UNIT_LENGTH * (col + j),
                    self.HEADING_SIZE + self.UNIT_LENGTH * (row + i),
                    self.UNIT_LENGTH,
                    self.UNIT_LENGTH,
                )
                pg.draw.rect(self.surface, (0, 0, 0), rect)

    def predict(self):
        data = np.zeros((28, 28))
        cell_length = self.canvas.shape[0] // 28
        for i in range(28):
            for j in range(28):
                data[i][j] = np.average(
                    self.canvas[
                        cell_length * i : cell_length * (i + 1),
                        cell_length * j : cell_length * (j + 1),
                    ]
                )
        digit = self.model.predict([data.flatten()])[0]
        width, _ = self.surface.get_size()
        pg.draw.rect(self.surface, (160, 160, 160), (0, 0, width, self.HEADING_SIZE))
        prediction_message = self.font.render(f"Digit: {digit}", True, (0, 0, 0))
        self.surface.blit(prediction_message, (10, 10))

    def clear(self):
        for row in self.canvas:
            for i, _ in enumerate(row):
                row[i] = 0
        width, height = self.surface.get_size()
        rect = (0, self.HEADING_SIZE, width, height - self.HEADING_SIZE)
        pg.draw.rect(self.surface, (255, 255, 255), rect)
