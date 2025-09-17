import math
from typing import Literal, Optional, overload

import matplotlib.pyplot as plt
import numpy as np


def mse(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (1/2) * np.sum(np.square(y_pred - y), axis=0)


@overload
def sigmoid(t: float) -> float: ...

@overload
def sigmoid(t: np.ndarray) -> np.ndarray: ...

def sigmoid(t: float | np.ndarray) -> float | np.ndarray:
    if isinstance(t, float):
        return 1 / (1 + math.exp(-t))
    return 1 / (1 + np.exp(-t))


@overload
def sigmoid_grad(t: float) -> float: ...

@overload
def sigmoid_grad(t: np.ndarray) -> np.ndarray: ...

def sigmoid_grad(t: float | np.ndarray) -> float | np.ndarray:
    return sigmoid(t) * (1 - sigmoid(t))


class TwoLayerMLP:
    def __init__(
        self,
        *,
        activation: Literal["sigmoid"] = "sigmoid",
        learning_rate: float = 1.0,
        loss: Literal["mse"] = "mse",
        weights: Optional[dict[int, np.ndarray]] = None,
    ) -> None:
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_grad = sigmoid_grad
        else:
            raise ValueError(f"Activation {activation = } not supported.")

        self.epoch = 0
        self.learning_rate = learning_rate

        if loss == "mse":
            self.loss = mse
        else:
            raise ValueError(f"Loss {loss = } not supported.")

        if weights:
            self.weights = weights
        else:
            raise ValueError("Please provide weights.")

    def __call__(self, *, x_0: np.ndarray, y: np.ndarray, num_epochs: int = 1) -> None:
        self.loss_history = []

        for _ in range(num_epochs):
            self.epoch += 1
            print(f"epoch = {self.epoch}")
            _, avg_loss = self.forward(x_0=x_0, y=y)
            self.loss_history.append(avg_loss)
            self.backward(y=y)
            self.clear_grad()
            print()

    @overload
    def forward(self, *, x_0: np.ndarray, y: None) -> tuple[np.ndarray, None]: ...

    @overload
    def forward(self, *, x_0: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]: ...

    def forward(
        self,
        *, 
        x_0: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[float]]:
        self.x_0 = x_0
        print(f"x_0 = {self.x_0}")

        print(f"weights_1 = {self.weights[1]}")

        self.xi_1 = self.weights[1] @ x_0
        print(f"xi_1 = {self.xi_1}")

        self.x_1 = np.vstack((
            np.array([[1.0, 1.0, 1.0]]),
            self.activation(self.xi_1),
        ))
        print(f"x_1 = {self.x_1}")

        print(f"weights_2 = {self.weights[2]}")

        self.xi_2 = self.weights[2] @ self.x_1
        print(f"xi_2 = {self.xi_2}")

        self.x_2 = self.xi_2
        print(f"x_2 = {self.x_2}")

        if y is not None:
            loss = self.loss(y, self.x_2)
            P = loss.shape[0]
            avg_loss = loss.sum() / loss.shape[0]
            print(f"{loss = }, {P = }, {avg_loss = }")

            return self.x_2, avg_loss

        return self.x_2, None

    def backward(self, *, y: np.ndarray) -> None:
        for attr in ['xi_1', 'x_1', 'xi_2', 'x_2']:
            if not hasattr(self, attr):
                raise ValueError(f"Missing attribute self.{attr}. Run forward pass.")

        self.delta_2 = (self.x_2 - y) * self.activation_grad(self.xi_2)
        print(f"delta_2 = {self.delta_2}")

        self.delta_1 = np.vstack((
            self.delta_2 * self.weights[2][0][2-1],
            self.delta_2 * self.weights[2][0][3-1],
        )) * self.activation_grad(self.xi_1)
        print(f"delta_1 = {self.delta_1}")

        P = self.x_2.shape[1]
        print(f"{P = }")

        self.weight_changes_2 = - self.learning_rate * (1/P) * self.delta_2 @ self.x_1.T
        print(f"weight_changes_2 = {self.weight_changes_2}")

        self.weight_changes_1 = - self.learning_rate * (1/P) * self.delta_1 @ self.x_0.T
        print(f"weight_changes_1 = {self.weight_changes_1}")

        self.weights[2] += self.weight_changes_2
        print(f"updated weights_2 = {self.weights[2]}")

        self.weights[1] += self.weight_changes_1
        print(f"updated weights_1 = {self.weights[1]}")

    def clear_grad(self) -> None:
        del self.delta_2
        del self.weight_changes_2
        del self.delta_1
        del self.weight_changes_1


x_0 = np.array([[1.0, 1.0, 1.0],
                [1.0, 0.0, -1.0],
                [1.0, -2.0, 1.0]])
y = np.array([[-1.0, 1.0, 1.0]])

w = {
    1: np.array([[0.4, 0.2, -1.2],
                 [0.7, -0.7, -0.7]]),
    2: np.array([[0.4, -0.5, 2.3]]),
}
num_epochs=15

mlp = TwoLayerMLP(weights=w)
mlp(x_0=x_0, y=y, num_epochs=num_epochs)

plt.plot(mlp.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Learning rate = {mlp.learning_rate}")
plt.show()
