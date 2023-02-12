import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import torch, torchvision, time, numpy as np
from tools.models import MLPNet, AlexClassifier
from tools.build_PNN import MultiPointsPNN
from Experiments.exp_tools import specification_matrix_from_labels
from tools.build_PNN import MultiPNN, NNSum
from Experiments.ImageNet.ImageNetTools import imagenet_test_acc, read_imagenet_images, imagenet_test_diff
import warnings
warnings.filterwarnings("ignore")


def main(num, n, num_core=10, data_path='/projectnb/pnn/data/ImageNet'):
    n_labels = 10
    torch.manual_seed(0)
    train_inputs, train_labels = read_imagenet_images(parent=data_path+'/train')
    MiMic_model = MLPNet([4096, 512, 256, 256, n_labels])
    MiMic_model.load_state_dict(torch.load('AlexNet_MiMic.pt'))
    train_labels = train_labels[:num*50]
    train_inputs = train_inputs[:num*50].float()
    
    def preprocess(inputs):
        alexnet = torchvision.models.alexnet(pretrained=True)
        l1 = alexnet.features(inputs).view(-1, 256 * 6 * 6)
        l2 = alexnet.classifier[:4](l1)
        l3 = torch.nn.functional.relu(MiMic_model.layers[0](l2)).clone().detach()
        return l3


    latent_vector = preprocess(train_inputs)
    target_model = MLPNet([512, 256, 256, n_labels])
    for i in range(3):
        target_model.layers[i] = MiMic_model.layers[i+1]

    target_pred = target_model(latent_vector).argmax(1)
    buggy_index = [i for i in range(len(target_pred)) if target_pred[i] != train_labels[i]]
    buggy_latent_vector = torch.stack([latent_vector[i] for i in buggy_index])[:num]
    train_labels = [train_labels[i] for i in buggy_index][:num]
    print(f'Find {len(train_labels)} buggy points.')

    P, ql, qu = specification_matrix_from_labels(train_labels, dim=n_labels)
    start = time.time()
    bounds = [torch.ones(512)*-20, torch.ones(512)*20]
    
    before_acc = imagenet_test_acc(data_path, target_model, preprocess=preprocess)
    print(f'Acc before repair: {before_acc}')
    
    PNN = MultiPointsPNN(target_model, n, bounds=bounds)
    PNN.point_wise_repair(buggy_latent_vector, P, ql, qu, remove_redundant_constraint=False)
    repaired_model = PNN.compute(num_core)
    cost_time = time.time() - start
    print('cost time:', cost_time)

    pred = repaired_model(buggy_latent_vector)
    correct = (pred.argmax(1) == torch.tensor(train_labels)).type(torch.float).sum().item()
    print('Repair rate = ', correct/len(buggy_latent_vector))

    after_acc = imagenet_test_acc(data_path, repaired_model, preprocess=preprocess)
    print(f'Acc after repair: {after_acc}, accuracy decrease {before_acc-after_acc}')

    diff_inf, diff_2 = imagenet_test_diff(target_model, repaired_model, data_path, test_num=1000, preprocess=preprocess)
    print('Ave inf diff:', diff_inf)
    print('Ave 2 diff:', diff_2)


if __name__ == '__main__':
    for num in [10, 20, 50]:
        for n in [0.0005]:
            print('-'*20, 'num', num, '-'*10, 'n', n, '-'*20)
            main(num, n)
