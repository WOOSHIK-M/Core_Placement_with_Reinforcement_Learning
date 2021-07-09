import os
import shutil
import numpy as np
import time
import networkx as nx
import torch
import random
import tqdm
import pandas as pd
import imageio

from Code.DataInfo import Node


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} s".format(original_fn.__name__, (end_time - start_time)), "\n")
        return result

    return wrapper_fn


def print_title(title):
    print('%s' % '=' * 40)
    print('{0:^40}'.format(title))
    print('%s' % '=' * 40 + '\n')


def make_gif(folderpath, file_type='png', gif_name='animation.gif', speed_spec=None):
    if speed_spec is None:
        speed_spec = {'duration': 2.}

    images = []
    for file_name in os.listdir(folderpath):
        if file_name.endswith('.{}'.format(file_type)):
            temp_path = os.path.join(folderpath, file_name)
            images.append(imageio.imread(temp_path))

    imageio.mimsave("{}/{}".format(folderpath, gif_name), images, **speed_spec)


def parse_lut(filepath, allnodenum, is_mul, lut_normal):
    if '.csv' in filepath:
        lut = np.array(pd.read_csv(filepath, header=None))
    else:
        lut = np.loadtxt(filepath)

    if allnodenum < lut.shape[1]:
        lut = lut[:, :allnodenum]

    new_lut = np.zeros([lut.shape[0], allnodenum])
    if not is_mul:
        lut[lut_normal:] = 0
    new_lut[:lut.shape[0], :lut.shape[1]] = lut
    return np.transpose(new_lut) - 1, new_lut


def node_to_adjmat(nodes):
    adjmat = np.zeros([len(nodes), len(nodes)])

    for src, node in enumerate(nodes):
        temp_src = src
        for packet_idx, dst in enumerate(node.connected_to_cores):
            temp_dst = dst
            packet_num = node.connected_to_packets[packet_idx]

            adjmat[temp_src, temp_dst] += packet_num

            while nodes[temp_dst].is_mul:
                temp_src = temp_dst
                temp_dst = nodes[temp_dst].multi_to_cores[0]

                adjmat[temp_src, temp_dst] += packet_num
    return adjmat


def adjmat_to_G(adjmat):
    G = nx.DiGraph()

    for i in range(adjmat.shape[0]):
        for j in range(adjmat.shape[1]):
            if adjmat[i][j] != 0:
                G.add_edge(i+1, j+1, weight=adjmat[i][j])
    return G


def lut_to_adjmat(lut, lut_normal, allnodenum):
    adjmat = np.zeros([allnodenum, allnodenum], dtype=np.int)

    # Normal Packets
    lut_n = np.array(lut.T[:, :lut_normal], dtype=np.int) - 1
    for src, lut_line in enumerate(lut_n):
        connected_to_cores = np.unique(lut_line)
        for dst in connected_to_cores:
            if dst != -1 and src != dst:
                adjmat[src, dst] += np.sum(lut_line == dst)

    # Multicast Packets
    lut_m = np.array(lut.T[:, lut_normal:], dtype=np.int) - 1
    is_mul = []
    for src, lut_line in enumerate(lut_m):
        if (lut == src + 1).sum() > 0:
            multi_to_cores = np.unique(lut_line)
            if np.all(multi_to_cores == -1):
                is_mul.append(0)
            else:
                for i in multi_to_cores:
                    if i != -1:
                        is_mul.append(i)
                        break
        else:
            is_mul.append(0)

    adjmat_m = np.zeros(adjmat.shape, dtype=np.int)
    for src, lut_line in enumerate(adjmat):
        for dst, weight in enumerate(lut_line):
            if weight != 0 and is_mul[dst] != 0:
                sc = dst
                dc = is_mul[dst]
                adjmat_m[sc, dc] += weight

                sc = dc
                while is_mul[sc] != 0:
                    dc = is_mul[sc]
                    adjmat_m[sc, dc] += weight

                    sc = dc
    adjmat += adjmat_m

    adjmat_to = adjmat
    adjmat_from = np.transpose(adjmat)

    adjmat = adjmat_to + adjmat_from

    return adjmat, adjmat_to, adjmat_from


def make_placement(config, x_vec, y_vec, is_print=False):
    X_DIM, Y_DIM, x_dim, y_dim = config.params_dict['X_DIM'], config.params_dict['Y_DIM'], config.params_dict['x_dim'], config.params_dict['y_dim'],
    x_lines = X_DIM * x_dim
    y_lines = Y_DIM * y_dim

    new_placement = np.zeros([y_lines, x_lines], dtype=np.int)

    for i in range(len(x_vec)):
        new_placement[y_vec[i], x_vec[i]] = i

    if is_print:
        print(" => Placement:")
        print(new_placement)
        print('\n')

    return new_placement + 1


def placement_to_lut(chip_placement, origin_lut, allnodenum, line_num):
    new_lut = np.zeros([allnodenum, line_num], dtype=np.int) - 1
    new_index = np.zeros(allnodenum, dtype=np.int) - 1
    new_pos = np.array(chip_placement.reshape(-1), dtype=np.int)

    for idx, node in enumerate(new_pos):
        if node != 0:
            new_lut[idx, :] = origin_lut[int(node) - 1, :]
            new_index[node - 1] = idx

    for detail_i in enumerate(new_lut):
        for detail_j in enumerate(detail_i[1]):
            if new_lut[detail_i[0], detail_j[0]] != -1:
                new_lut[detail_i[0], detail_j[0]] = new_index[int(new_lut[detail_i[0], detail_j[0]])]
    return np.transpose(new_lut + 1)


def make_result_folder(ConfigData):
    # Make result directory
    ResultFolderPath = ConfigData.params_dict["ProjectFolderPath"] + "/result"
    if not os.path.isdir(ResultFolderPath):
        os.mkdir(ResultFolderPath)

    folder_name = ConfigData.params_dict['ProjectFolderPath'] + '/result/{}'.format(ConfigData.params_dict['lut_filename_e'])
    subfoldernum = 0
    if os.path.isdir(folder_name):
        dirlst = os.listdir(folder_name)
        dirlstnp = np.array(dirlst, dtype=np.int)
        for i in dirlstnp:
            check_path = folder_name + '/' + str(i) + '/subs'
            if len(os.walk(check_path).__next__()[2]) == 0:
                shutil.rmtree(folder_name + '/' + str(i))
        dirlst = os.listdir(folder_name)
        dirlstnp = np.array(dirlst, dtype=np.int)
        if dirlstnp.shape[0] != 0:
            subfoldernum = np.max(dirlstnp) + 1
    else:
        os.mkdir(folder_name)

    folder_name = folder_name + '/' + str(subfoldernum)
    if os.path.isdir(folder_name):
        print('Already existed folder')
        exit()
    else:
        os.mkdir(folder_name)
    ConfigData.params_dict['rst_path'] = folder_name

    shutil.copy(ConfigData.params_dict["ConfigFilePath"], folder_name)
    # shutil.copy("model.py", folder_name)
    # shutil.copy("env.py", folder_name)

    # Make mode index
    mode = ConfigData.params_dict['mode']
    linename = '%s' % '=' * 20
    if mode == 0:
        os.mkdir(folder_name + '/{0:^30}'.format(" MODE - Exact Mapping "))
    elif mode == 1:
        os.mkdir(folder_name + '/{0:^30}'.format(" MODE - Random Search "))
    elif mode == 2:
        os.mkdir(folder_name + '/{0:^30}'.format(" MODE - Simulated Annealing "))
    elif mode == 3:
        os.mkdir(folder_name + '/{0:^30}'.format(" MODE - Reinforcement Learning "))
    os.mkdir(folder_name + "/" + linename)

    # Make save directory
    new_lut_folder = folder_name + '/new_lut'
    if not os.path.isdir(new_lut_folder):
        os.mkdir(new_lut_folder)
    ConfigData.params_dict['new_lut_path'] = new_lut_folder

    sub_folder = folder_name + '/subs'
    if not os.path.isdir(sub_folder):
        os.mkdir(sub_folder)
    ConfigData.params_dict['sub_folder_path'] = sub_folder

    plot_folder = folder_name + '/plots'
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    ConfigData.params_dict['plot_folder_path'] = plot_folder


def laplacian_normalize(adjmat):
    if np.max(adjmat) != 0:
        adjmat /= np.max(adjmat) + 1e-12

    # Laplacian D^(-1/2) * A * D&(-1/2)
    D = np.zeros(adjmat.shape)
    for i in range(D.shape[0]):
        D[i, i] = (adjmat[i] != 0).sum()

    with np.errstate(divide='ignore'):
        D_sqrt = 1.0 / np.sqrt(D)
    D_sqrt = np.where(D_sqrt == np.inf, 0.0, D_sqrt)

    L = D - adjmat
    L = np.matmul(D_sqrt, L)
    adjmat = np.matmul(L, D_sqrt)

    return adjmat


def RandomGraphGeneration(num_placed, connection_number, connection_weight):
    choice_number = list(connection_number)
    choice_weight = list(connection_weight)
    choice_dst = list(range(1, num_placed + 1))

    G = nx.DiGraph()
    src = 1
    while G.number_of_nodes() < num_placed:
        random.shuffle(choice_number)

        temp_number = choice_number[0]

        for _ in range(temp_number):
            random.shuffle(choice_weight)
            temp_weight = choice_weight[0]

            random.shuffle(choice_dst)
            temp_dst = choice_dst[0]
            while temp_dst == src:
                random.shuffle(choice_dst)
                temp_dst = choice_dst[0]

            if src < temp_dst:
                G.add_edge(src, temp_dst, weight=temp_weight)
            else:
                G.add_edge(temp_dst, src, weight=temp_weight)

        src += 1
    return G


def GetLaplacianAdj(G, MaxNum):
    temp_laplacian = nx.directed_laplacian_matrix(G, weight="weight")

    LaplacianAdj = np.zeros([MaxNum, MaxNum])
    LaplacianAdj[:temp_laplacian.shape[0], :temp_laplacian.shape[1]] = temp_laplacian
    return LaplacianAdj


def GetFeatures(G, MaxNum):
    numberofnodes = G.number_of_nodes()

    # 7 features - is_multicast, in_degree, out_degree, all_degree, in_weight, out_weight, all_weight
    features = []

    for i in range(1, numberofnodes + 1):
        if i in G.nodes:
            temp_features = [0,
                             G.in_degree[i],
                             G.out_degree[i],
                             G.in_degree[i] + G.out_degree[i],
                             G.in_degree(weight="weight")[i],
                             G.out_degree(weight="weight")[i],
                             G.in_degree(weight="weight")[i] + G.out_degree(weight="weight")[i]]
        else:
            temp_features = [0 for _ in range(7)]
        features.append(temp_features)

    features = np.array(features, dtype=np.float)
    for i in range(1, 7):
        features[:, i] = features[:, i] / np.max(features[:, i])

    Features = np.zeros([MaxNum, 7])
    Features[:features.shape[0], :features.shape[1]] = features

    return Features


def GetLabel(G):
    # return G.size(weight="weight") / nx.number_of_nodes(G)
    return torch.norm(torch.tensor(nx.directed_laplacian_matrix(G, weight="weight"))).item()


def GetNode(G, MaxNum):
    nodes = [Node() for _ in range(MaxNum)]

    adj = nx.adj_matrix(G, weight="weight").toarray()
    for rowidx, row in enumerate(adj):
        for colidx, value in enumerate(row):
            if value != 0:
                nodes[rowidx].connected_to_cores.append(colidx)
                nodes[rowidx].connected_to_packets.append(value)
                nodes[rowidx].connected_to_cores_num += 1

                nodes[colidx].connected_from_cores.append(rowidx)
                nodes[colidx].connected_from_packets.append(value)
                nodes[colidx].connected_from_cores_num += 1
    return nodes


def MakeDataset(MaxNum, MinNum, DataNum, BaseNum, connection_number, connection_weight):
    # The number of nodes
    Numchoice = list(range(MinNum, MaxNum + 1))

    # Dataset Generation
    DataSet = []
    Adjacencies = []
    Features = []
    Label = []
    Nodes = []
    for _ in tqdm.tqdm(range(DataNum)):
        random.shuffle(Numchoice)
        G = RandomGraphGeneration(Numchoice[0], connection_number, connection_weight)

        DataSet.append(G)
        Adjacencies.append(GetLaplacianAdj(G, BaseNum))
        Features.append(GetFeatures(G, BaseNum))
        Label.append(GetLabel(G))
        Nodes.append(GetNode(G, MaxNum))

    return DataSet, Adjacencies, Features, Label, Nodes

