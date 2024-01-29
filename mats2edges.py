import numpy as np

def mats2edges(all_mats):
    num_node = np.shape(all_mats)[0]
    num_sub = np.shape(all_mats)[2]
    num_task = np.shape(all_mats)[3]
    num_edge = num_node * (num_node-1) // 2

    all_edges = np.zeros([num_edge, num_sub, num_task])
    iu1 = np.triu_indices(num_node, 1)
    for i in range(num_sub):
        for j in range(num_task):
            all_edges[:, i, j] = all_mats[iu1[0], iu1[1], i, j]

    all_edges = np.transpose(all_edges, (1, 2, 0))
    all_edges = np.reshape(all_edges, [num_sub, -1])
    return all_edges


def mat2edges(all_mats):
    num_node = np.shape(all_mats)[0]
    num_sub = np.shape(all_mats)[2]
    num_edge = num_node * (num_node-1) // 2

    all_edges = np.zeros([num_edge, num_sub])
    iu1 = np.triu_indices(num_node, 1)
    for i in range(num_sub):
        all_edges[:, i] = all_mats[iu1[0], iu1[1], i]

    all_edges = np.transpose(all_edges, (1, 0))
    all_edges = np.reshape(all_edges, [num_sub, -1])
    return all_edges
