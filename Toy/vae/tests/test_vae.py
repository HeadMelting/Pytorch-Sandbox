import torch
import unittest
from model.vanillaVAE import VAE
from torchinfo import summary


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = VAE(3, 10)

    def test_summary(self):
        print(summary(self.model, [55, 3, 64, 64], col_names=['input_size',
                                                              'output_size',
                                                              'kernel_size',
                                                              'num_params']))

    def test_forward(self):
        x = torch.randn(55, 3, 64, 64)
        y = self.model(x)

        print('Model Output Size:', y[0].size)

    def test_loss(self):
        x = torch.randn(55, 3, 64, 64)
        result = self.model(x)
        loss = self.model.loss_function(*result, M_N=0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()
