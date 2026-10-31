import numpy as np
from tqdm import tqdm

from neural_network.util import (
    flatten_weights,
    label_binarizer_fit_transform,
    sigmoid,
    sigmoid_prime,
    unflatten_weights,
)


class MLPClassifier:
    def __init__(
        self, hidden_layer_sizes=None, alpha=0.0001, lammy=1, epochs=10, verbose=False
    ):
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes else [100]
        self.alpha = alpha
        self.lammy = lammy
        self.epochs = epochs
        self.verbose = verbose
        self.layer_sizes = []
        self.weights = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        Y = label_binarizer_fit_transform(y)
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [Y.shape[1]]
        size = sum(self.layer_sizes[i] * self.layer_sizes[i - 1] + self.layer_sizes[i] for i in range(1, len(self.layer_sizes)))

        weights_flat = 0.01 * np.random.randn(size)
        num_batches = 100
        for epoch in range(self.epochs):
            loss = None
            reordered_indices = np.random.choice(
                X.shape[0], size=X.shape[0], replace=False
            )
            for i in tqdm(
                range(num_batches),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=not self.verbose,
            ):
                batch = reordered_indices[
                    i * X.shape[0] // num_batches : (i + 1) * X.shape[0] // num_batches
                ]
                loss, gradient = self.fun_obj(weights_flat, X[batch], Y[batch])
                weights_flat -= self.alpha * gradient
            if self.verbose:
                print(f"Loss = {loss}")

        self.weights = unflatten_weights(weights_flat, self.layer_sizes)

    def predict(self, X: np.ndarray):
        Z = None
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        assert Z is not None
        return np.argmax(Z, axis=1)

    def fun_obj(self, weights_flat: np.ndarray, X: np.ndarray, Y: np.ndarray):
        weights = unflatten_weights(weights_flat, self.layer_sizes)
        activations = [X]
        activation_derivatives = []
        Z = None
        for W, b in weights:
            Z = activations[-1] @ W.T + b
            activations.append(sigmoid(Z))
            activation_derivatives.append(sigmoid_prime(Z))

        assert Z is not None
        f = np.sum(-Z[Y.astype(bool)] + np.log(np.sum(np.exp(Z), axis=1)))
        grad = np.exp(Z) / np.sum(np.exp(Z), axis=1)[:, None] - Y
        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)
        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            grad = grad @ W * activation_derivatives[i - 1]
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)
            g = [(grad_W, grad_b)] + g
        g = flatten_weights(g)

        f += 0.5 * self.lammy * np.sum(weights_flat**2)
        g += self.lammy * weights_flat
        return f, g
