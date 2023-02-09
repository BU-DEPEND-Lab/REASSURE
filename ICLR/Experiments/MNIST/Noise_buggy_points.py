import torch, torchvision, time, numpy as np
from tools.models import MLPNet
from tools.build_PNN import MultiPointsPNN
from Experiments.exp_tools import *


target_model = MLPNet([28*28, 256, 256, 10])
target_model.load_state_dict(torch.load('target_model.pt'))
test_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=False,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64)
buggy_inputs, right_label, cut_point = find_buggy_inputs(test_dataloader, target_model, 100)

