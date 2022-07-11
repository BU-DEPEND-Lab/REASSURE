import torch, torchvision, time, math, numpy as np
from REASSURE.ExperimentModels import MLP
from REASSURE.Repair import REASSURERepair
from example.nnet_reader import HCAS_Model
from example.h5_reader import My_H5Dataset
from REASSURE.ExperimentTools import constraints_from_labels


def success_rate(model, buggy_inputs, right_label, is_print=0):
    with torch.no_grad():
        pred = model(buggy_inputs)
        correct = (pred.argmax(1) == right_label).type(torch.float).sum().item()
    if is_print == 1:
        print('Original accuracy on buggy_inputs: {} %'.format(100*correct / len(buggy_inputs)))
    elif is_print == 2:
        print('Success repair rate: {} %'.format(100*correct / len(buggy_inputs)))
    return correct/len(buggy_inputs)


def Repair_HCAS(repair_num, n):
    input_dim = 3
    input_boundary = [np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
                      np.block([np.array([65000, 65000, math.pi]), np.array([65000, 65000, math.pi])])]
    target_model = HCAS_Model('TrainedNetworks/HCAS_rect_v6_pra1_tau05_25HU_3000.nnet')
    buggy_inputs = torch.load('cex_pra1_tau05.pt')
    buggy_inputs, right_labels = buggy_inputs[:repair_num], torch.ones([repair_num], dtype=torch.long)*4
    output_constraints = constraints_from_labels(right_labels, dim=5)
    success_rate(target_model, buggy_inputs, right_labels, is_print=1)
    start = time.time()
    REASSURE = REASSURERepair(target_model, input_boundary, n)
    repaired_model = REASSURE.point_wise_repair(buggy_inputs, output_constraints=output_constraints, core_num=1)
    cost_time = time.time()-start
    print('Time:', cost_time)
    success_rate(repaired_model, buggy_inputs, right_labels, is_print=2)


if __name__ == '__main__':
    for num in [50]:
        print('-'*50, 'number:', num, '-'*50)
        Repair_HCAS(num, 1)
