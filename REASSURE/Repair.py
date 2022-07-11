import torch, time, torch.nn as nn, numpy as np
from scipy.optimize import linprog
import multiprocessing
from REASSURE.Tools import linearize_model, linear_region_from_input, dig_block
from REASSURE.RepairModules import SupportNet, SingleRegionRepairNet, NetSum


class REASSURERepair:
    def __init__(self, model, input_boundary, n=10):
        self.model = model
        self.input_boundary = input_boundary
        self.n = n

    def point_wise_repair(self, buggy_inputs, output_constraints, core_num=1):
        print('Working on {} cores.'.format(core_num))
        pool = multiprocessing.Pool(core_num)
        arg_list = [[buggy_inputs[i], [output_constraints[0][i], output_constraints[1][i]]] for i in range(len(buggy_inputs))]
        repair_net_list = pool.starmap(self._repair_one_area, arg_list)
        pool.close()
        # repair_net_list = []
        # for i in range(len(buggy_inputs)):
        #     repair_net_list.append(self._repair_one_area(buggy_inputs[i], [output_constraints[0][i], output_constraints[1][i]]))
        return NetSum(self.model, repair_net_list)


    def _repair_one_area(self, buggy_input, output_constraint):
        buggy_input = buggy_input.view(1, -1)
        buggy_input.requires_grad = True
        patch_area = linear_region_from_input(
            buggy_input, self.model.allHiddenNeurons(buggy_input).view(-1), self.input_boundary)
        A, b = patch_area
        # temp = np.matmul(A, buggy_input.detach().squeeze().numpy()) - b
        g = SupportNet(A, b, self.n)
        linearized_model = linearize_model(buggy_input, self.model)
        c, d = self.repair_via_LP(patch_area, output_constraint, linearized_model)
        p = SingleRegionRepairNet(g, c, d, self.input_boundary)
        return p

    def repair_via_LP(self, patch_area, output_constraint, linearized_model: list[np.ndarray]):
        # A: in_cons_num*in_dim, b: in_cons_num*1
        # A_out: out_cons_num*out_dim, b_out: out_cons_num*1
        # cf: out_dim*in_dim, df: out_dim*1
        # p: in_cons_num*out_dim, p^: in_cons_num*out_dim
        # q: in_cons_num*out_cons_num
        # d: out_dim
        A, b = patch_area
        b: np.ndarray
        A_out, b_out = output_constraint
        cf, df = linearized_model
        cf: np.ndarray
        out_dim, in_dim = cf.shape
        in_cons_num, out_cons_num = b.size, b_out.size

        # temp = linprog(np.ones(in_dim), A_ub=A, b_ub=b, bounds=[None, None])
        # print(temp.message)

        A_eq = np.block([[dig_block([A.transpose() for _ in range(out_dim)]), dig_block([A.transpose() for _ in range(out_dim)]),
                np.zeros([in_dim*out_dim, in_cons_num*out_cons_num]), np.zeros([in_dim*out_dim, out_dim]), np.zeros([in_dim*out_dim, 1])],
             [-np.block([[A_out[i, j]*A.transpose() for j in range(out_dim)] for i in range(out_cons_num)]),
                np.zeros([in_dim*out_cons_num, in_cons_num*out_dim]), dig_block([A.transpose() for _ in range(out_cons_num)]),
              np.zeros([in_dim*out_cons_num, out_dim]), np.zeros([in_dim*out_cons_num, 1])]
             ])
        b_eq = np.block([np.zeros(in_dim*out_dim), np.block([np.matmul(A_out[i], cf) for i in range(out_cons_num)])])
        bounds = [None, None]
        A_ub = np.block([[dig_block([b for _ in range(out_dim)]), np.zeros([out_dim, in_cons_num*out_dim]), np.zeros([out_dim, in_cons_num*out_cons_num]),
                          np.eye(out_dim), -np.ones([out_dim, 1])],
                         [np.zeros([out_dim, in_cons_num*out_dim]), dig_block([b for _ in range(out_dim)]), np.zeros([out_dim, in_cons_num*out_cons_num]),
                          -np.eye(out_dim), -np.ones([out_dim, 1])],
                         [np.zeros([out_cons_num, in_cons_num*out_dim]), np.zeros([out_cons_num, in_cons_num*out_dim]),
                          dig_block([b for _ in range(out_cons_num)]), np.block([[A_out[i]]for i in range(out_cons_num)]), np.zeros([out_cons_num, 1])],
                         [-np.eye(in_cons_num*out_dim*2+in_cons_num*out_cons_num), np.zeros([in_cons_num*out_dim*2+in_cons_num*out_cons_num, out_dim+1])]
                         ])
        b_ub = np.block([np.zeros(out_dim), np.zeros(out_dim), np.block([b_out[i] - np.matmul(A_out[i], df) for i in range(out_cons_num)]),
                         np.zeros(in_cons_num*out_dim*2+in_cons_num*out_cons_num)])
        objective = np.block([np.zeros(in_cons_num*out_dim*2+in_cons_num*out_cons_num+out_dim), np.ones(1)])
        solution = linprog(c=objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        # if not solution.success:
        #     print('There is a problem with solving linear programming: ', solution.message)
        temp = np.block([dig_block([A.transpose() for _ in range(out_dim)]),
                        np.zeros([in_dim*out_dim, in_cons_num*out_dim+in_cons_num*out_cons_num+out_dim+1])])
        c_x = np.stack([temp[i*in_dim:(i+1)*in_dim] for i in range(out_dim)])
        d_x = np.block([np.zeros([out_dim, in_cons_num*out_dim*2+in_cons_num*out_cons_num]), np.eye(out_dim),
                        np.zeros([out_dim, 1])])
        return np.matmul(c_x, solution.x), np.matmul(d_x, solution.x)


if __name__ == '__main__':
    avg_t = []
    for i in range(100):
        R = REASSURERepair(1, 1, 1)
        in_dim, out_dim = 3, 5
        in_cons_num, out_cons_num = 125, 5
        A = np.round(np.random.random([in_cons_num, in_dim]), 4)
        A_out = np.round(np.random.random([out_cons_num, out_dim]), 4)
        x, x_out = np.round(np.random.random(in_dim), 4), np.round(np.random.random(out_dim), 4)
        x = np.absolute(x)
        b, b_out = np.matmul(A, x), np.matmul(A_out, x_out)
        b, b_out = b + np.ones_like(b)*0.01, b_out + np.ones_like(b_out)*0.01
        A = np.block([[A], [np.eye(in_dim)], [-np.eye(in_dim)]])
        b = np.block([b, np.ones(in_dim), np.zeros(in_dim)])
        cf, df = np.random.random([out_dim, in_dim]), np.random.random(out_dim)
        # print(dig_block([A.transpose() for _ in range(out_dim)]))
        start = time.time()
        solution = R.repair_via_LP([A, b], [A_out, b_out], [cf, df])
        c, d = solution
        avg_t.append(time.time()-start)
        print(i, time.time()-start)
        y = np.matmul(c+cf, x) + df + d
        print(np.matmul(A_out, y) - b_out <= 0)
    print(sum(avg_t)/len(avg_t))
