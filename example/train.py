import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch, argparse
from REASSURE.Loader import load_data
from REASSURE.Trainer import train, test_acc
from REASSURE.ExperimentModels import MLP
from REASSURE.LogTools import write_log


def check_point(epoch):
    model.eval()
    best_res = [0]
    ACC = test_acc(model, loaders['test'], device)
    if ACC > sum(best_res):
        print('Update Best Model: ACC = ${}%$'.format(ACC))
        best_res[:] = [ACC]
        torch.save(model.state_dict(), 'target_model.pt')
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MNIST'])
    parser.add_argument('--model', type=str, choices=['MLP, CNN'])
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    args.cuda = False
    args.dataset = 'MNIST'
    args.max_epochs = 20
    args.lr = 0.001

    device = torch.device("cuda:0" if args.cuda else "cpu")
    loaders = load_data(args.dataset, path='../../data')
    model = MLP([28*28, 32, 32, 10])
    model.to(device)
    note = [str(args)]
    print('-' * 10 + note[0] + '-' * 10)
    train(model, device, loaders, args.max_epochs, args.lr, callback_function=check_point, call_back_period=3)



