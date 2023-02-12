import torch, math
from Experiments.HCAS.nnet_reader import HCAS_Model
from Experiments.HCAS.h5_reader import My_H5Dataset


loss = torch.nn.CrossEntropyLoss()
alpha = -0.1


def find_cex(model, number, is_print=False, is_save=True, is_diff_pattern=True):
    cex_list = []
    batch_size = 64
    total = 0
    activation_pattern_set = set()
    while len(cex_list) < number:
        total += batch_size
        inputs = torch.randint(100, 5000, [batch_size, 2], dtype=torch.float)
        theta = torch.rand([batch_size, 1], dtype=torch.float)*math.pi*1/2 - math.pi
        X = torch.cat([inputs, theta], dim=-1)
        pred = torch.argmax(model(X), dim=1)
        new_cex = [X[i] for i in range(batch_size) if pred[i] != 4]
        if len(new_cex) > 0:
            for j, activation_pattern in enumerate(model.activation_pattern(torch.stack(new_cex))):
                if  not is_diff_pattern or (activation_pattern not in activation_pattern_set):
                    activation_pattern_set.add(activation_pattern)
                    cex_list.append(new_cex[j])
    cex = torch.stack(cex_list[:number])
    if is_print:
        print('Find {} cex in {}.'.format(number, total))
        print(cex)
        print(model(cex))
        print(torch.argmax(model(cex), dim=1))
    if is_save:
        torch.save(cex, '../../../PRDNN/HCAS/cex_pra1_tau05.pt')
        torch.save(cex, 'cex_pra1_tau05.pt')
    return cex


def find_right_points(model, number, is_print):
    right_points_list = []
    batch_size = 64
    total = 0
    activation_pattern_set = set()
    while len(activation_pattern_set) < number:
        total += batch_size
        inputs = torch.randint(0, 5000, [batch_size, 2], dtype=torch.float)
        theta = torch.round(torch.rand([batch_size, 1], dtype=torch.float)*math.pi*1/2*10**5)/10**5 - math.pi
        X = torch.cat([inputs, theta], dim=-1)
        pred = torch.argmax(model(X), dim=1)
        new_right_points = [X[i] for i in range(batch_size) if pred[i] == 4]
        if len(new_right_points) > 0:
            for j, activation_pattern in enumerate(model.activation_pattern(torch.stack(new_right_points))):
                if activation_pattern not in activation_pattern_set:
                    activation_pattern_set.add(activation_pattern)
                    right_points_list.append(new_right_points[j])
    right_points = torch.stack(right_points_list[:number])
    if is_print:
        print('Find {} right points in {}.'.format(number, total))
        print(right_points)
        print(network(right_points))
        print(torch.argmax(network(right_points), dim=1))
    torch.save(right_points, '../../../PRDNN/HCAS/right_points_pra1_tau05.pt')
    torch.save(right_points, 'right_points_pra1_tau05.pt')


if __name__ == '__main__':
    network = HCAS_Model('TrainedNetworks/HCAS_rect_v6_pra1_tau05_25HU_3000.nnet')
    find_right_points(network, 100, True)
    # find_cex(network, 50, True)
    # cex = torch.load('nom_cex_pra1_tau05.pt')
    # print(cex)
    # print(network(cex))
    # print(torch.argmax(network(cex), dim=1))
