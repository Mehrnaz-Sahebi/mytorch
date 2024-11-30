from mytorch.optimizer import Optimizer
import numpy as np
"TODO: (optional) implement Adam optimizer"


class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        for layer in self.layers:
            layer.first_moment = np.zeros_like(layer.weight.data)
            layer.second_moment = np.zeros_like(layer.weight.data)
            if layer.need_bias:
                layer.bfirst_moment = np.zeros_like(layer.bias.data)
                layer.bsecond_moment = np.zeros_like(layer.bias.data)

        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.layers:
            layer.first_moment = self.beta1 * layer.first_moment + \
                                 (1 - self.beta1) * layer.weight.grad
            layer.second_moment = self.beta2 * layer.second_moment + \
                                  (1 - self.beta2) * layer.weight.grad ** 2

            m1_hat = layer.first_moment / (1 - self.beta1 ** self.t)
            m2_hat = layer.second_moment / (1 - self.beta2 ** self.t)

            layer.weight.data = layer.weight.data - self.learning_rate * \
                                (m1_hat / (m2_hat ** 0.5 + self.epsilon))

            if layer.need_bias:
                layer.bfirst_moment = self.beta1 * layer.bfirst_moment + \
                                      (1 - self.beta1) * layer.bias.grad
                layer.bsecond_moment = self.beta2 * layer.bsecond_moment + \
                                       (1 - self.beta2) * layer.bias.grad ** 2

                m1b_hat = layer.bfirst_moment / \
                          (1 - self.beta1 ** self.t)
                m2b_hat = layer.bsecond_moment / \
                          (1 - self.beta2 ** self.t)

                layer.bias.data = layer.bias.data - self.learning_rate * \
                                  (m1b_hat / (m2b_hat ** 0.5 + self.epsilon))

