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
        f1.append(input.grad.squeeze().detach().numpy())
        f2.append((output - torch.inner(input.grad, input)).squeeze().detach().numpy())
    return np.stack(f1), np.stack(f2)


def get_linear_region(input_data, all_neurons, bounds):
    """
    :param input_data: the input of the neural network
    :param all_neurons: all hidden neurons in the network
    :param bounds: bounds on the inputs in the form of [lower_bounds, upper_bounds]
    :return: H-representation of the linear region, in the form of {x | Ax <= b}
    """
    A_list, b_list = [], []
    for neuron in all_neurons:
        neuron.backward(retain_graph=True)
        grad_x = input_data.grad
        A = grad_x.clone().detach()
        b = torch.matmul(A, input_data.squeeze()) - neuron
        input_data.grad.zero_()
        if neuron >= 0:
            A, b = -A, -b
        A_list.append(A.detach().numpy())
        b_list.append(b.detach().numpy())
    A = np.concatenate([bounds[0], np.concatenate(A_list)])
    b = np.concatenate([bounds[1], np.concatenate(b_list)])
    return np.float32(A), np.float32(b)


def construct_block_matrix(m: list):
    dim = len(m)
    return np.block([[m[i] if i == j else np.zeros_like(m[i]) for j in range(dim)] for i in range(dim)])