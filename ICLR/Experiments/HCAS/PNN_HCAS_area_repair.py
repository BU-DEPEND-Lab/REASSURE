import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import torch, torchvision, time, math, numpy as np
from tools.models import MLPNet
from tools.build_PNN import MultiPointsPNN
from Experiments.exp_tools import specification_matrix_from_labels, test_acc, test_diff_on_dataloader
from Experiments.HCAS.nnet_reader import HCAS_Model
from Experiments.HCAS.h5_reader import My_H5Dataset
from tools.linear_region import verify, redundant_constraints_remover
from Experiments.HCAS.find_cex import find_cex


def PNN_HCAS_advanced(repair_num, n):
    right_label = 4
    bounds = [np.array([-65000, -65000, -math.pi]), np.array([65000, 65000, math.pi])]
    target_model = HCAS_Model('TrainedNetworks/HCAS_rect_v6_pra1_tau20_25HU_3000.nnet')
    all_linear_regions = np.load('all_linear_regions.npy', allow_pickle=True)
    buggy_linear_regions = \
        [item for item in all_linear_regions if not verify(item[0], item[1], item[2], target_model.layers, right_label=right_label)]
    if len(buggy_linear_regions) > repair_num:
        buggy_linear_regions = buggy_linear_regions[:repair_num]
    print('number of buggy linear regions:', len(buggy_linear_regions))
    
    h5_file = My_H5Dataset('TrainingData/HCAS_rect_TrainingData_v6_pra1_tau20.h5')
    loader = torch.utils.data.DataLoader(h5_file, batch_size=64)
    
    constraints_num = [len(item[0]) for item in buggy_linear_regions]
    P, ql, qu = specification_matrix_from_labels([right_label]*len(buggy_linear_regions), 5)
    before_acc = test_acc(loader, target_model)
    print(f'Acc before repair: {before_acc}')
    start = time.time()
    PNN = MultiPointsPNN(target_model, n, bounds, test_model=False)
    PNN.area_repair(buggy_linear_regions, P, ql, qu)
    repaired_model = PNN.compute(4)
    cost_time = time.time()-start
    print('Time:', cost_time)

    after_acc = test_acc(loader, repaired_model)
    print(f'Acc after repair: {after_acc}')

    train_inf_diff, train_2_diff = test_diff_on_dataloader(loader, repaired_model, target_model)
    print('Average inf Diff:', train_inf_diff)
    print('Average 2 Diff:', train_2_diff)

    torch.manual_seed(0)
    batch_size = 10000
    random_inputs_on_patch = torch.randint(100, 5000, [batch_size, 2], dtype=torch.float)
    theta = torch.rand([batch_size, 1], dtype=torch.float) * math.pi * 1 / 2 - math.pi
    X_on_patch = torch.cat([random_inputs_on_patch, theta], dim=-1).unsqueeze(1)
    diff_inf, diff_2 = test_diff_on_dataloader(zip(X_on_patch, range(len(X_on_patch))), repaired_model, target_model)
    print('Average inf Diff(on patch):', diff_inf)
    print('Average 2 Diff(on patch):', diff_2)


if __name__ == '__main__':
    for num in [10, 20, 50, 100]:
        print('-' * 50)
        for n in [1]:
            print('n = ', n, '-' * 50)
            PNN_HCAS_advanced(num, n)
