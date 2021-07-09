import numpy as np
import csv
import time
import math
import os
import torch
import networkx as nx

from Code.DataInfo import ConfigClass, Node
from Code.Model import *
import Code.Utils as utils
import Code.CommDetect as cm


def make_config():
    # utils.print_title("CONFIGURATION ... ")

    config = ConfigClass()

    if config.use_cuda:
        print("Available devices: {}".format(torch.cuda.device_count()))

    # GET PROJECT ABSTRACT PATH
    CurrentFilePath = os.path.dirname(os.path.abspath(__file__))
    ProjectFolderPath = os.path.abspath(os.path.join(CurrentFilePath, os.pardir))
    config.params_dict["ProjectFolderPath"] = ProjectFolderPath

    # PARSE DATA FROM CONFIGURATION FILE
    ConfigFilePath = ProjectFolderPath + "/Config_File/config.ini"
    config.params_dict["ConfigFilePath"] = ConfigFilePath
    file1 = open(ConfigFilePath, 'r', encoding='utf=8')  # r - read, w - write, a - add
    info = file1.readlines()

    for line in info:
        # print(line[:-1])

        if '#' == line[0]:
            continue
        elif '=' in line:
            equal_index = line.index('=')
            param_name = line[:equal_index].strip()
            param_value = line[equal_index + 1:].strip()

            if '#' in line:
                comment_index = line.index('#')
                param_value = line[equal_index + 1:comment_index].strip()

            if param_name in config.params_dict.keys():
                if ',' in param_value:
                    config.params_dict[param_name] = param_value.split(', ')
                else:
                    if 'True' in param_value:
                        config.params_dict[param_name] = True
                    elif 'False' in param_value:
                        config.params_dict[param_name] = False
                    elif type(config.params_dict[param_name]) is type(True):
                        if int(param_value) == 1:
                            config.params_dict[param_name] = True
                        else:
                            config.params_dict[param_name] = False
                    else:
                        config.params_dict[param_name] = type(config.params_dict[param_name])(param_value)
    file1.close()
    print("\n")

    # Basic topology info
    allchipnum = config.params_dict['X_DIM'] * config.params_dict['Y_DIM']
    nodeperchip = config.params_dict['x_dim'] * config.params_dict['y_dim']
    allnodenum = allchipnum * nodeperchip

    config.params_dict['allnodenum'] = allnodenum
    config.params_dict['nodeperchip'] = nodeperchip
    config.params_dict['allchipnum'] = allchipnum

    X_DIM = config.params_dict['X_DIM']
    x_dim = config.params_dict['x_dim']
    y_dim = config.params_dict['y_dim']

    for iii in range(allnodenum):
        corenum = iii
        chipnum = 0
        while corenum >= nodeperchip:
            corenum -= nodeperchip
            chipnum += 1
        chipX = chipnum % X_DIM
        chipY = math.floor(chipnum / X_DIM)
        coreX = corenum % x_dim
        coreY = math.floor(corenum / x_dim)

        config.params_dict['chipidx'].append(int(chipnum))
        config.params_dict['phy_x'].append(int(coreX + chipX * x_dim))
        config.params_dict['phy_y'].append(int(coreY + chipY * y_dim))
        config.params_dict['OnePhyX'].append(int(iii % (x_dim * X_DIM)))
        config.params_dict['OnePhyY'].append(int(iii / (x_dim * X_DIM)))

    # READ LUT FILE
    lut_filename = config.params_dict['lut_filename']

    if type(lut_filename) is type(''):
        lut_filename = [lut_filename]
        config.params_dict['lut_filename'] = lut_filename

    lutnum = len(lut_filename)
    config.params_dict['lutnum'] = lutnum

    filename_extracted = [os.path.splitext(os.path.basename(i))[0] for i in lut_filename]
    config.params_dict['lut_filename_e'] = filename_extracted

    lut, lut_original = [], []
    for i in lut_filename:
        temp_lut, temp_lut_original = utils.parse_lut(ProjectFolderPath + "/Dataset/" + i,
                                                      config.params_dict['allnodenum'],
                                                      config.params_dict['is_mul'],
                                                      config.params_dict['lut_normal'])
        lut.append(temp_lut)
        lut_original.append(temp_lut_original)
    config.params_dict['lut_original'] = lut_original
    config.params_dict['lut'] = lut

    # nodes
    nodes = [generate_node(lut[i],
                           config.params_dict['lut_normal'],
                           allnodenum,
                           print_node_info=config.params_dict['print_node_info']) for i in range(lutnum)]
    config.params_dict['nodes'] = nodes

    # Make multicast vector
    mul_vec = np.zeros([lutnum, allnodenum])
    for idx1 in range(lutnum):
        for idx2, node in enumerate(nodes[idx1]):
            if node.is_mul == 1:
                mul_vec[idx1, idx2] = 1

    config.params_dict['mulvec'] = mul_vec

    # Make adjacency graph, node degree, weight
    adj_matrix = [utils.node_to_adjmat(nodes[i]) for i in range(lutnum)]
    G = [utils.adjmat_to_G(adj_matrix[i]) for i in range(lutnum)]

    config.params_dict['G'] = G

    config.params_dict['num_placed'] = [temp_g.number_of_nodes() for temp_g in G]
    vec_placed = [(np.array(temp_g.nodes, dtype=np.int) - 1).tolist() for temp_g in G]
    config.params_dict['vec_placed'] = np.sort(vec_placed).tolist()

    # Domain Adaptation
    if config.params_dict['DA_mode']:
        DA_process(config)

    # # Pre-processing when RL-agent is employed
    # if allchipnum > 1 and config.params_dict['mode'] == 3:
    #     utils.print_title("Reinforcement Learning ... ")
    #
    #     time.sleep(0.5)
    #
    #     cm.CCD(config)

    # MAKE RESULT FILE
    utils.make_result_folder(config)

    return config


def generate_node(lut, lut_normal, allnodenum, print_node_info=False):
    lut = lut
    lut_normal = lut_normal

    lut = np.array(lut, dtype=np.int)
    NodeSize = allnodenum

    num_placed = 0
    vec_placed = []

    noderange = range(NodeSize)
    lut = lut.tolist()
    line_normal = lut_normal
    not_isolated = np.zeros(NodeSize)

    nodes = []
    nodenum = 0

    for i in noderange:
        node = Node()

        if i < len(lut):
            # find normal connections
            normal_line = lut[i][:line_normal]
            while -1 in normal_line:
                normal_line.remove(-1)

            connected_to_cores = []
            connected_to_packets = []
            if normal_line:
                not_isolated[nodenum] = 1
                for v in normal_line:
                    if v not in connected_to_cores:
                        connected_to_cores.append(v)
                        not_isolated[v] = 1
                for v in connected_to_cores:
                    connected_to_packets.append(normal_line.count(v))

            node.connected_to_cores = connected_to_cores
            node.connected_to_cores_num = len(connected_to_cores)
            node.connected_to_packets = connected_to_packets

            # find multicast connections
            mul_line = lut[i][line_normal:]
            while -1 in mul_line:
                mul_line.remove(-1)

            multi_to_cores = []
            if mul_line and (np.array(lut) == i).sum() > 0:
                not_isolated[nodenum] = 1
                for v in mul_line:
                    if v not in multi_to_cores:
                        multi_to_cores.append(v)
                        not_isolated[v] = 1
            if len(multi_to_cores) > 0:
                node.is_mul = 1
            node.multi_to_cores = multi_to_cores

        nodes.append(node)
        nodenum += 1

    for idx, val in enumerate(nodes):
        if not_isolated[idx] == 0:
            nodes[idx].is_isolate = 1
        else:
            num_placed += 1
            vec_placed.append(idx + 1)

    for i in range(len(nodes)):
        if nodes[i].connected_to_cores:
            for j in range(len(nodes[i].connected_to_cores)):
                nodes[nodes[i].connected_to_cores[j]].connected_from_cores.append(i)
                nodes[nodes[i].connected_to_cores[j]].connected_from_packets.append(nodes[i].connected_to_packets[j])

    for i in range(len(nodes)):
        if nodes[i].multi_to_cores:
            if i not in nodes[nodes[i].multi_to_cores[0]].connected_from_cores:
                nodes[nodes[i].multi_to_cores[0]].connected_from_cores.append(i)
                nodes[nodes[i].multi_to_cores[0]].connected_from_packets.append(sum(nodes[nodes[i].multi_to_cores[0]].connected_from_packets))

    for i in range(len(nodes)):
        if nodes[i].connected_from_cores:
            nodes[i].connected_from_cores_num = len(nodes[i].connected_from_cores)

    mul_num = 0
    for i in range(len(nodes)):
        if nodes[i].is_mul == 1:
            mul_num += 1

    if print_node_info:
        for i in range(len(nodes)):
            node = nodes[i]
            print('%-10s' % ('Node' + str(i) + ': '), end=' ')
            print('%-41s' % ('con_to: ' + str(node.connected_to_cores)), end=' ')
            print('%-45s' % ('packet_num: ' + str(node.connected_to_packets)), end=' ')
            print('%-43s' % ('con_from: ' + str(node.connected_from_cores)), end=' ')
            print('is_mul: {: < 5}'.format(node.is_mul), end=' ')
            print('%-15s' % ('mul_to: ' + str(node.multi_to_cores)), end=' ')
            print('from_mul: {: < 5}'.format(node.from_mul), end=' ')
            print('to_mul: {: < 5}'.format(node.to_mul), end=' ')
            print('is_iso: {: < 8}'.format(node.is_isolate))
        print("==>  Node class is generated...")
        print('==>  Canvas Size: {} \n==>  Will_be_placed: {} \n==>  The number of multicast nodes: {}'.format(NodeSize, num_placed, mul_num))

    return nodes


def DA_process(config):
    TrainData, TrainAdj, TrainFeatures, TrainLabel, TrainNode \
        = utils.MakeDataset(config.params_dict['max_nodenum'],
                            config.params_dict['min_nodenum'],
                            config.params_dict['random_lutnum'],
                            config.params_dict['nodeperchip'],
                            np.array(config.params_dict['degrees'], dtype=np.int).tolist(),
                            np.array(config.params_dict['weights'], dtype=np.int).tolist())

    config.params_dict['random_nodes'] = TrainNode

    model = GCN(nfeat=7,
                dropout=config.params_dict['dropout'],
                MaxNum=config.params_dict['nodeperchip'])
    model.load_state_dict(torch.load(config.params_dict['ProjectFolderPath'] + '/gcn_model.pt')['gcn_model'], strict=False)

    model.eval()
    random_features = torch.empty([config.params_dict['random_lutnum'], config.params_dict['nodeperchip']])
    for lutidx in range(config.params_dict['random_lutnum']):
        random_features[lutidx] \
            = model(torch.FloatTensor(TrainFeatures[lutidx]),
                    torch.FloatTensor(TrainAdj[lutidx]),
                    is_features=True)
    config.params_dict['random_features'] = random_features

    config.params_dict['test_features'] = [model(
        torch.FloatTensor(utils.GetFeatures(tempg, config.params_dict['nodeperchip'])),
        torch.FloatTensor(utils.GetLaplacianAdj(tempg, config.params_dict['nodeperchip'])),
        is_features=True
    ).tolist() for tempg in config.params_dict['G']]
