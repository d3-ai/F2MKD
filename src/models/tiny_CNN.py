import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from typing import List

from .net import Net

class tinyCNN(Net):
    def __init__(self,
    input_spec: List,
    conv_kernel_size: int = 5,
    conv_kernel_stride: int = 1,
    pool_kernel_size: int = 2,
    pool_kernel_stride: int = 2,
    ) -> None:
        super(tinyCNN, self).__init__()
        self.in_channels, self.width, self.height = input_spec[0], input_spec[1], input_spec[2]
        self.conv_kernel_size, self.conv_kernel_stride = conv_kernel_size, conv_kernel_stride
        self.pool_kernel_size, self.pool_kernel_stride = pool_kernel_size, pool_kernel_stride 
        self.conv1 = nn.Conv2d(self.in_channels, 6, self.conv_kernel_size)
        self.pool = nn.MaxPool2d(self.pool_kernel_size, self.pool_kernel_stride)
        self.conv2 = nn.Conv2d(6, 16, self.conv_kernel_size)
        for _ in range(2):
            self._conv_update()
            self._pool_update()
        self.fc1 = nn.Linear(16 * self.width * self.height, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.width * self.height)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _conv_update(self)->None:
        self.width = ceil((self.width-(self.conv_kernel_size-1)-1)/self.conv_kernel_stride + 1)
        self.height = ceil((self.height-(self.conv_kernel_size-1)-1)/self.conv_kernel_stride + 1)

    def _pool_update(self)->None:
        self.width = ceil((self.width-(self.pool_kernel_size-1)-1)/self.pool_kernel_stride + 1)
        self.height = ceil((self.height-(self.pool_kernel_size-1)-1)/self.pool_kernel_stride + 1)
