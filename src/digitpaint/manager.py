import numpy as np
import pygame as pg

from neural_network import MLPClassifier


class PaintManager:
    WIDTH = 560
    HEADING_SIZE = 80
    BRUSH_RADIUS: int = 24
    UNIT_LENGTH: int = 4
    QUEUE_MS: int = 32
    PROB_BAR_MARGIN: int = 10
    MAX_BAR_HEIGHT: int = 50
    DIGITS_ROW_HEIGHT: int = 30

    def __init__(self, model: MLPClassifier):
        self.surface = pg.display.set_mode((self.WIDTH, self.WIDTH + self.HEADING_SIZE))
        self.model = model
        self.font = pg.font.SysFont("lucidaconsole", 16, bold=True)
        self.predict_remaining_ms = -1
        self.predict_queued = False
        self.canvas = np.zeros(
            (self.WIDTH // self.UNIT_LENGTH, self.WIDTH // self.UNIT_LENGTH)
        )
        self._clear()

    def on_key(self, key: int) -> None:
        if key == pg.K_c:
            self._clear()

    def update(self, dt: int):
        if pg.mouse.get_pressed()[0]:
            x, y = pg.mouse.get_pos()
            row = (y - self.HEADING_SIZE) // self.UNIT_LENGTH
            col = x // self.UNIT_LENGTH
            self._draw_circle(row, col)
            if not self.predict_queued:
                self.predict_remaining_ms = self.QUEUE_MS
                self.predict_queued = True

        if self.predict_queued:
            self.predict_remaining_ms -= dt
            if self.predict_remaining_ms <= 0:
                self._render_prediction()
                self.predict_queued = False

    def _draw_circle(self, row: int, col: int):
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

    def _render_prediction(self):
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
        probabilities = self.model.predict_proba(np.array([data.flatten()]))[0]
        width, _ = self.surface.get_size()
        bar_container_width = width // probabilities.size
        self._draw_empty_header()
        for i, prob in enumerate(probabilities):
            rect = (
                bar_container_width * i + self.PROB_BAR_MARGIN,
                self.DIGITS_ROW_HEIGHT + int(self.MAX_BAR_HEIGHT * (1 - prob)),
                bar_container_width - self.PROB_BAR_MARGIN * 2,
                int(self.MAX_BAR_HEIGHT * prob)
            )
            pg.draw.rect(self.surface, (0, 0, 0), rect)
            digit_surface = self.font.render(str(i), True, (0, 0, 0))
            left = bar_container_width * i + bar_container_width // 2 - digit_surface.get_width() // 2
            self.surface.blit(digit_surface, (left, 10))

    def _draw_empty_header(self):
        width, _ = self.surface.get_size()
        pg.draw.rect(self.surface, (160, 160, 160), (0, 0, width, self.HEADING_SIZE))
        pg.draw.rect(self.surface, (0, 0, 0), (0, self.DIGITS_ROW_HEIGHT, width, 1))
        pg.draw.rect(self.surface, (0, 0, 0), (0, self.HEADING_SIZE - 2, width, 2))

    def _clear(self):
        for row in self.canvas:
            for i, _ in enumerate(row):
                row[i] = 0
        width, height = self.surface.get_size()
        rect = (0, self.HEADING_SIZE, width, height - self.HEADING_SIZE)
        pg.draw.rect(self.surface, (255, 255, 255), rect)
        self._draw_empty_header()
