import numpy as np

from neural_network import util
from neural_network.exceptions import HyperParameterError


class MLPClassifier:
    BATCH_SIZE = 100

    def __init__(
        self,
        hidden_layer_sizes=None,
        activation="relu",
        alpha=0.0001,
        learning_rate=0.001,
        max_iter=200,
        verbose=False,
    ):
        if activation == "relu":
            self.activation_func = util.relu
            self.activation_prime = util.relu_prime
        elif activation == "sigmoid":
            self.activation_func = util.sigmoid
            self.activation_prime = util.sigmoid_prime
        else:
            raise HyperParameterError(f"Unrecognized activation function {activation}")
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes else [100]
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.layer_sizes = []
        self.weights = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        Y = util.label_binarizer_fit_transform(y)
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [Y.shape[1]]
        size = sum(
            self.layer_sizes[i] * self.layer_sizes[i - 1] + self.layer_sizes[i]
            for i in range(1, len(self.layer_sizes))
        )

        N = X.shape[0]
        weights_flat = 0.01 * np.random.randn(size)
        for epoch in range(self.max_iter):
            total_loss = 0
            order = np.random.choice(N, size=N, replace=False)

            for i in range(0, N, self.BATCH_SIZE):
                indices = order[i : i + self.BATCH_SIZE]
                loss, gradient = self._backprop(weights_flat, X[indices], Y[indices])
                weights_flat -= self.learning_rate * gradient
                total_loss += loss * len(indices)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter}, Loss = {total_loss / N}")

        self.weights = util.unflatten_weights(weights_flat, self.layer_sizes)

    def predict(self, X: np.ndarray):
        A = X
        for W, b in self.weights[:-1]:
            A = self.activation_func(A @ W.T + b)

        W, b = self.weights[-1]
        Z = A @ W.T + b
        return np.argmax(Z, axis=1)

    def _backprop(self, weights_flat: np.ndarray, X: np.ndarray, Y: np.ndarray):
        weights = util.unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        activation_derivatives = []
        for W, b in weights[:-1]:
            Z = activations[-1] @ W.T + b
            activations.append(self.activation_func(Z))
            activation_derivatives.append(self.activation_prime(Z))

        W, b = weights[-1]
        Z = activations[-1] @ W.T + b

        # Compute log softmax loss and parameter derivatives
        N = X.shape[0]
        loss = np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[Y.astype(bool)])
        # Add regularization for weights
        loss += 0.5 * self.alpha * sum(np.sum(W**2) for W, _ in weights)
        Q = (util.softmax(Z) - Y) / N
        weight_derivatives = []
        for i in range(len(weights) - 1, -1, -1):
            W, b = weights[i]
            # Include regularization for weight derivatives
            W_grad = Q.T @ activations[i] + self.alpha * W
            b_grad = np.sum(Q, axis=0)
            weight_derivatives.append((W_grad, b_grad))
            if i > 0:
                Q = (Q @ W) * activation_derivatives[i - 1]

        weight_derivatives.reverse()
        gradient = util.flatten_weights(weight_derivatives)
        return loss, gradient
