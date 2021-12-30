import numpy as np
num_node = 18
link = [(i, i) for i in range(num_node)]
neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                     (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                     (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
edge = link + neighbor_link
center = 1

max_hop=1
dilation=1
valid_hop = range(0, max_hop + 1, dilation)
adjacency = np.zeros((num_node, num_node))

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
def get_hop_distance(num_node, edge, max_hop):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

        # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


hop_dis = get_hop_distance(
            num_node, edge, max_hop=max_hop)

for hop in valid_hop:
    adjacency[hop_dis == hop] = 1
normalize_adjacency = normalize_digraph(adjacency)


A = []
for hop in valid_hop:
    a_root = np.zeros((num_node, num_node))
    a_close = np.zeros((num_node, num_node))
    a_further = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if hop_dis[j, i] == hop:
                if hop_dis[j, center] == hop_dis[
                    i, center]:
                    a_root[j, i] = normalize_adjacency[j, i]
                elif hop_dis[j,
                        center] > hop_dis[i,
                        center]:
                    a_close[j, i] = normalize_adjacency[j, i]
                else:
                    a_further[j, i] = normalize_adjacency[j, i]
    if hop == 0:
        A.append(a_root)
    else:
        A.append(a_root + a_close)
        A.append(a_further)
A = np.stack(A)

