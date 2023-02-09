import torch, numpy as np, pathlib, os, random


def find_buggy_inputs(dataloader, model, repair_num):
    """Find buggy inputs and the corresponding correct labels for a neural network from a dataset, with a specified number of repair inputs."""
    model.eval()
    buggy_inputs, correct_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            buggy_indices = [i for i in range(len(X)) if predictions[i].argmax() != y[i]]
            buggy_inputs += [X[i] for i in buggy_indices]
            correct_labels += [y[i] for i in buggy_indices]
            if len(buggy_inputs) >= repair_num:
                buggy_inputs, correct_labels = buggy_inputs[:repair_num], correct_labels[:repair_num]
                break
    if len(buggy_inputs) < repair_num:
        print('Not enough buggy inputs found!')
    return torch.stack(buggy_inputs), torch.stack(correct_labels)


def constraints_from_labels(correct_labels, dim=10):
    """Translate the specification of a correct class label into a set of matrix constraints for a classification problem."""
    A_list, b_list = [], []
    for label in correct_labels:
        row = np.zeros([1, dim])
        row[0][label] = 1
        A = np.eye(dim) - np.matmul(np.ones([dim, 1]), row)
        A_list.append(np.delete(A, label, 0))
        b_list.append(np.zeros(dim-1))
    return A_list, b_list



def compare_models_on_dataloader(dataloader, model1, model2):
    """Calculate the average and maximum norm differences between two neural networks on a dataset."""
    total = 0
    avg_inf_norm_difference, avg_l2_norm_difference = torch.tensor(0.0), torch.tensor(0.0)
    with torch.no_grad():
        for X, _ in dataloader:
            predictions1, predictions2 = model1(X.float()), model2(X.float())
            softmax_difference = torch.softmax(predictions1, dim=-1) - torch.softmax(predictions2, dim=-1)
            inf_norm_difference = torch.norm(softmax_difference, dim=-1, p=np.inf)
            l2_norm_difference = torch.norm(softmax_difference, dim=-1, p=2)
            total += len(X)
            avg_inf_norm_difference += inf_norm_difference.sum()
            avg_l2_norm_difference += l2_norm_difference.sum()
    return avg_inf_norm_difference / total, avg_l2_norm_difference / total



def calculate_repair_success_rate(model, buggy_inputs, right_label, print_result=False):
    """Calculate the success rate of repairing buggy inputs using a model."""
    with torch.no_grad():
        predictions = model(buggy_inputs)
        correct_predictions = (predictions.argmax(dim=1) == right_label).type(torch.float).sum().item()
    success_rate = correct_predictions / len(buggy_inputs)
    if print_result:
        print('Success rate of repairing buggy inputs:', success_rate)
    return success_rate



def evaluate_model_accuracy(data_loader, model):
    """Evaluate the accuracy of a neural network on a given dataset"""
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            if y.dim() > 1:
                y = y.argmax(1)
            total += len(X)
            predictions = model(X.float())
            correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
    return correct / total

