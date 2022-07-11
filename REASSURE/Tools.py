import torch, numpy as np


def linearize_model(input, model):
    """
    :param input: x;
    :param model: neural network;
    :return: f1, f2 where model(x) = f1*x+f2 on the linear region.
    """
    f1, f2 = [], []
    for output in model(input).squeeze():
        model(input)
        output.backward(retain_graph=True)
        f1.append(input.grad.clone().detach().squeeze())
        f2.append((output - torch.inner(input.grad, input)).squeeze())
    return torch.stack(f1).detach().numpy(), torch.stack(f2).detach().numpy()


def linear_region_from_input(x, all_neurons, boundary):
    """
    :param x: input of neural network;
    :param all_neurons: all the hidden neurons of x;
    :param bounds: neural network bounds for all inputs.
    bounds = [lowerbound_list, upperbound_list], lowerbound is a np.array of lowerbound for every dimension;
    :return: H-rep of a linear region, {x| Ax <= b}.
    """
    A_list, b_list = [], []
    for neuron in all_neurons:
        neuron.backward(retain_graph=True)
        grad_x = x.grad
        A = grad_x.clone().detach()
        b = torch.matmul(A, x.squeeze()) - neuron
        x.grad.zero_()
        if neuron >= 0:
            A, b = -A, -b
        A_list.append(A.detach().numpy())
        b_list.append(b.detach().numpy())
    A = np.concatenate([boundary[0], np.concatenate(A_list)])
    b = np.concatenate([boundary[1], np.concatenate(b_list)])
    return np.float32(A), np.float32(b)


def dig_block(m: list):
    dim = len(m)
    return np.block([[m[i] if i == j else np.zeros_like(m[i]) for i in range(dim)] for j in range(dim)])