import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import numpy as np, time
import torch, torchvision
from tools.models import MLPNet
from tools.build_PNN import MultiPointsPNN
from Experiments.exp_tools import test_acc, specification_matrix_from_labels, test_diff_on_dataloader, success_repair_rate
import warnings
warnings.filterwarnings("ignore")


def WatermarkMNIST(num, n):
    target_model_weights = np.load('target_model.npz', allow_pickle=True)['arr_0']
    target_model = MLPNet([28*28, 150, 10])
    target_model.layers[0].weight.data = torch.from_numpy(np.transpose(target_model_weights[0]))
    target_model.layers[0].bias.data = torch.from_numpy(target_model_weights[1])
    target_model.layers[1].weight.data = torch.from_numpy(np.transpose(target_model_weights[2]))
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=True,
                                                                              download=True,
                                                                              transform=torchvision.transforms.ToTensor()), batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='../../data', train=False,
                                                                             download=True,
                                                                             transform=torchvision.transforms.ToTensor()), batch_size=64,shuffle=True)
    # target_model_test_correct_rate = test(test_dataloader, target_model, loss_fn, True)
    acc = test_acc(test_dataloader, target_model)
    print(f'Test Acc after before: {acc*100}%')
    wm_images = np.load('./WMdata/wm.images.npy')[:num]
    wm_labels = np.loadtxt('./WMdata/wm.labels.txt', dtype='int32')[:num]
    np.random.seed(0)
    std = 0.0005
    noised_buggy_inputs = wm_images
    for _ in range(10):
        noised_buggy_inputs = np.concatenate([noised_buggy_inputs, wm_images + np.random.randn(*wm_images.shape) * std], axis=0)
    wm_wlabels = torch.tensor([wm_labels[i] - 1 if wm_labels[i] > 0 else 9 for i in range(len(wm_labels))])
    wm_images = torch.from_numpy(wm_images).float()
    wm_images, wm_wlabels = wm_images.view(-1, 28*28), wm_wlabels
    P, ql, qu = specification_matrix_from_labels(wm_wlabels)

    start = time.time()
    PNN = MultiPointsPNN(target_model, n, bounds=[torch.zeros(28*28), torch.ones(28*28)])
    PNN.point_wise_repair(wm_images, P, ql, qu, False)
    repaired_model = PNN.compute(3)
    cost_time = time.time() - start
    print('cost time:', cost_time)

    success_repair_rate(repaired_model, wm_images, wm_wlabels, is_print=True)
    noised_buggy_inputs = torch.from_numpy(noised_buggy_inputs)
    buggy_inf_diff, buggy_2_diff = test_diff_on_dataloader(zip(noised_buggy_inputs.unsqueeze(1), range(len(noised_buggy_inputs))), repaired_model, target_model)
    print('Average inf Diff(on patch area):', buggy_inf_diff)
    # train_avg_diff_inf, _ = test_diff(train_dataloader, repaired_model, target_model, is_print=False)
    # test_avg_diff_inf, _ = test_diff(test_dataloader, repaired_model, target_model, is_print=False)
    # print('Average inf Diff(on all):', (6 * train_avg_diff_inf + test_avg_diff_inf) / 7)

    # buggy_avg_diff_2, _ = test_diff_on_dataloader(zip(noised_buggy_inputs.unsqueeze(1), range(len(noised_buggy_inputs))), repaired_model, target_model)
    print('Average L2 Diff(on patch area):', buggy_2_diff)
    train_inf_diff, train_2_diff = test_diff_on_dataloader(train_dataloader, repaired_model, target_model)
    test_inf_diff, test_2_diff = test_diff_on_dataloader(test_dataloader, repaired_model, target_model)
    print('Average inf Diff(on all):', (6 * train_inf_diff + test_inf_diff) / 7)
    print('Average L2 Diff(on all):', (6 * train_2_diff + test_2_diff) / 7)
    
    acc = test_acc(test_dataloader, repaired_model)
    print(f'Test Acc after repair: {acc*100}%')

if __name__ == '__main__':
    for num in [1, 5, 25, 50, 100]:
        for n in [0.02]:
            print('-'*20, 'Number:', num, '-'*20)
            WatermarkMNIST(num, n)
