import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import torch, torchvision, time, numpy as np
from tools.models import MLPNet
from tools.build_PNN import MultiPointsPNN
from Experiments.exp_tools import *
import warnings
warnings.filterwarnings("ignore")


def PNN_MNIST(repair_num, n, num_core, remove_redundant_constraint=False):
    torch.manual_seed(0)
    target_model = MLPNet([28*28, 256, 256, 10])
    target_model.load_state_dict(torch.load('Experiments/MNIST/target_model.pt'))

    test_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=False,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64)
    train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=True,
                                            download=True, transform=torchvision.transforms.ToTensor()), batch_size=64,
                                            shuffle=True)

    buggy_inputs, right_label = find_buggy_inputs(test_dataloader, target_model, 100)

    acc = test_acc(test_dataloader, target_model)
    print(f'Test Acc after before: {acc*100}%')
    buggy_inputs, right_label = buggy_inputs[:repair_num], right_label[:repair_num]
    P, ql, qu = specification_matrix_from_labels(right_label)
    start = time.time()
    PNN = MultiPointsPNN(target_model, n, bounds=[torch.zeros(28*28), torch.ones(28*28)])
    PNN.point_wise_repair(buggy_inputs, P, ql, qu, remove_redundant_constraint=remove_redundant_constraint)
    repaired_model = PNN.compute(num_core)
    cost_time = time.time()-start
    print('cost time:', cost_time)

    std = 0.01
    success_repair_rate(repaired_model, buggy_inputs, right_label, is_print=True)
    noised_buggy_inputs = buggy_inputs
    for _ in range(10):
        noised_buggy_inputs = torch.cat([noised_buggy_inputs, buggy_inputs + torch.randn(buggy_inputs.size()) * std], dim=0)

    buggy_inf_diff, buggy_2_diff = test_diff_on_dataloader(zip(noised_buggy_inputs.unsqueeze(1), range(len(noised_buggy_inputs))),
                                             repaired_model, target_model)
    print('Average inf Diff(on patch area):', buggy_inf_diff)
    print('Average L2 Diff(on patch area):', buggy_2_diff)
    train_inf_diff, train_2_diff = test_diff_on_dataloader(train_dataloader, repaired_model, target_model)
    test_inf_diff, test_2_diff = test_diff_on_dataloader(test_dataloader, repaired_model, target_model)
    print('Average inf Diff(on all):', (6 * train_inf_diff + test_inf_diff) / 7)
    print('Average L2 Diff(on all):', (6 * train_2_diff + test_2_diff) / 7)

    acc = test_acc(test_dataloader, repaired_model)
    print(f'Test Acc after repair: {acc*100}%')
    torch.save(repaired_model.state_dict(), 'PNNed_model_{}.pt'.format(repair_num))


if __name__ == '__main__':
    for num in [10]:
        print('-'*50, ';repair num =', num, '-'*50)
        PNN_MNIST(num, 0.5, num_core=10, remove_redundant_constraint=False)
