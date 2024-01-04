from .types_ import *

from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


'''
abstractmethod: BaseVAE를 implement해서 class 생성시, method를 구현하지 않으면 에러.
raise NotImplementedError: 클래스 생성시 구현하지 않아도 에러는 발생하지 않지만, 구현 안하면 메서드 호출 시 에러.

'''
