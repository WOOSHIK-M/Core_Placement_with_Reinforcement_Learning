import networkx as nx
import numpy as np
import copy
import community as lvcm
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.cm as cmm

import Code.Utils as utils


def buildG(adjmat):
    G = nx.Graph()

    # Construct the weighted network graph
    for src, connected_to_cores in enumerate(adjmat):
        for dst, weight in enumerate(connected_to_cores):
            if weight != 0:
                G.add_edge(int(src), int(dst), weight=float(weight))

    return G


def computeLoadBalance(NewA, comm_num, chip_iter, chip_x):
    LoadBalance = np.zeros([comm_num, comm_num])
    for sc, connected_to_cores in enumerate(NewA):
        for dc, weight in enumerate(connected_to_cores):
            if weight != 0:
                schip = chip_iter[sc]
                schipx = schip % chip_x
                schipy = int(schip / chip_x)

                dchip = chip_iter[dc]
                dchipx = dchip % chip_x
                dchipy = int(dchip / chip_x)

                CurChip = schip
                path = [CurChip]
                while schipx != dchipx:
                    if schipx < dchipx:
                        CurChip += 1
                        schipx += 1
                    else:
                        CurChip -= 1
                        schipx -= 1
                    path.append(CurChip)

                while schipy != dchipy:
                    if schipy < dchipy:
                        CurChip += chip_x
                        schipy += 1
                    else:
                        CurChip -= chip_x
                        schipy -= 1
                    path.append(CurChip)

                for idx in range(len(path) - 1):
                    LoadBalance[path[idx], path[idx + 1]] += weight
    return LoadBalance


def FindBestChipPlacement(OnChipData, NewA, comm_num, chip_x):
    chip_iter = list(range(comm_num))
    chip_iters = list(permutations(chip_iter))
    chip_iters = [list(i) for i in chip_iters]

    MinIndex = -1
    MinCongestion = 1e12
    for ChipIter, chip_iter in enumerate(chip_iters):
        LoadBalance = computeLoadBalance(NewA, comm_num, chip_iter, chip_x)
        # print("# of Placement: {}, MaxCongestion: {}".format(ChipIter, np.max(LoadBalance)))
        if np.max(LoadBalance) < MinCongestion:
            MinIndex = ChipIter
            MinCongestion = np.max(LoadBalance)

            if MinCongestion == 0:
                break

    chip_iter = chip_iters[MinIndex]

    print('FINAL COMMUNITIES ... ')
    for i, n in enumerate(chip_iter):
        print("=> COMM {}: {} member(s)".format(i, len(OnChipData[n])))
    print("\nMINIMUM CONGESTION of Inter-Chip: {}".format(MinCongestion))

    NewOnChipData = []
    for i in chip_iter:
        NewOnChipData.append(OnChipData[i])

    return NewOnChipData


def PlotG(G, partition):
    pos = nx.spring_layout(G)
    cmap = cmm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def IsAbleToBePlaced(GroupSize, AvailableSize, StartPosition):
    CommNum = len(AvailableSize)
    for i in range(CommNum):
        CheckIdx = StartPosition + i
        if CheckIdx >= CommNum:
            CheckIdx -= CommNum

        if AvailableSize[CheckIdx] >= GroupSize:
            return CheckIdx
    return -1


def SortGroups(DendoDict, DendoLevel, G):
    NodeIdx = np.array(list(DendoDict[DendoLevel].keys()))  # Logic Core Index
    CommIdx = np.array(list(DendoDict[DendoLevel].values()))  # Community Index

    NodeData = [NodeIdx[np.where(CommIdx == i)].tolist() for i in range(np.max(CommIdx) + 1)]  # Groups Info (Includes all index of Logic cores)
    GroupSize = [len(i) for i in NodeData]

    SeG = lvcm.induced_graph(DendoDict[DendoLevel], G, weight='weight')
    LinkData = nx.node_link_data(SeG)['links']

    GroupS = []  # Source groups
    GroupT = []  # Target groups
    GroupW = []  # Each weight of links
    for data in LinkData:
        if data['source'] != data['target']:
            GroupS.append(data['source'])
            GroupT.append(data['target'])
            GroupW.append(data['weight'])

    Assigned = [False for _ in range(len(GroupSize))]  # Check which community is assigned their priority
    WeightSorted = np.argsort(GroupW)[::-1]  # The weight of connection has the highest priority
    SortedTriplet = [[GroupS[i], GroupT[i], GroupW[i]] for i in WeightSorted]  # [Group1, Group2, Weight]
    MappingPriority = []

    SortIdx = 0
    while not np.all(Assigned):
        Group1 = SortedTriplet[SortIdx][0]
        Group2 = SortedTriplet[SortIdx][1]

        BigGroup = Group1
        SmallGroup = Group2
        if GroupSize[Group1] < GroupSize[Group2]:
            BigGroup = Group2
            SmallGroup = Group1

        if not Assigned[BigGroup]:
            MappingPriority.append(BigGroup)
            Assigned[BigGroup] = True

        if not Assigned[SmallGroup]:
            MappingPriority.append(SmallGroup)
            Assigned[SmallGroup] = True

        SortIdx += 1
        if SortIdx == len(SortedTriplet):
            for i in Assigned:
                if not Assigned[i]:
                    MappingPriority.append(i)
                    Assigned[i] = True
    return MappingPriority, NodeData, GroupSize


def CCD(config):
    print("Customized Community Detection ... ")

    allnodenum = config.params_dict['allnodenum']
    comm_num = config.params_dict['allchipnum']
    max_member_num = config.params_dict['nodeperchip']
    chip_x = config.params_dict['X_DIM']
    lut = config.params_dict['lut_original']
    adjmat = config.params_dict['adjmat']

    print("==>  Start DETECTING COMMUNITIES ...\n")

    new_lut = np.zeros([config.params_dict['lut_mul'], comm_num * max_member_num])
    new_lut[:, :lut.shape[1]] = lut

    # Remove directions
    adjmat_origin = copy.deepcopy(adjmat)

    # Vacant Space
    VacantSpace = [max_member_num for _ in range(comm_num)]

    # Post processing
    G = buildG(adjmat)          # G - original network graph / G_backup - store removed edges

    # Fast Unfolding Algorithm
    # partition = lvcm.best_partition(graph=G, partition=None, weight='weight', resolution=1., randomize=True)

    DendoG = lvcm.generate_dendrogram(graph=G, weight='weight', resolution=1., randomize=True)
    DendoLevel = len(DendoG)
    DendoDict = []
    DendoDetail = [[] for _ in range(allnodenum)]
    for i in range(DendoLevel):
        TempG = lvcm.partition_at_level(DendoG, DendoLevel - i - 1)
        DendoDict.append(TempG)
        for TempKey in list(TempG.keys()):
            DendoDetail[TempKey].append(TempG[TempKey])

    # Groups are sorted by their priority
    MappingPriority, NodeData, GroupSize = SortGroups(DendoDict, 0, G)

    # Start to map Communities
    PlaceOrder = [NodeData[i] for i in MappingPriority]

    GlobalIdx = 0

    GroupIdx = 0
    ChipIdx = 0
    TempLevel = 1
    CurrentSplitNum = 0

    AfterPlaced = [[] for _ in range(comm_num)]
    while True:
        CurrentGroup = GroupIdx

        CurrentGroupSize = len(PlaceOrder[CurrentGroup])

        # print("\nPlaceOrder: {}, \nVacantSpace: {} \nCurrentGroupSize (Idx: {}): {}\nGlobalIdx: {}\n".format(PlaceOrder, VacantSpace,
        #                                                                         GroupIdx, CurrentGroupSize, GlobalIdx))

        # Check which chip has enough vacant space
        PlacedIdx = IsAbleToBePlaced(CurrentGroupSize, VacantSpace, ChipIdx)

        if PlacedIdx == -1:
            TempGroups = dict()
            TempIdx = MappingPriority[GlobalIdx]

            flag1 = True
            TempMappingPriority = []
            while TempLevel < DendoLevel:
                for DendoIdx, PartitionDetail in enumerate(DendoDetail):
                    if len(PartitionDetail) > 0:
                        if PartitionDetail[0] == TempIdx:
                            if PartitionDetail[TempLevel] in TempGroups:
                                TempGroups[PartitionDetail[TempLevel]].append(DendoIdx)
                            else:
                                TempGroups[PartitionDetail[TempLevel]] = [DendoIdx]
                if len(TempGroups.keys()) == 1 or CurrentSplitNum != 0:
                    TempLevel += 1
                    TempIdx = list(TempGroups.keys())[0]
                    TempGroups = dict()
                    continue

                TempMappingPriority = SortGroups(DendoDict, TempLevel, G)
                flag1 = False
                break

            if flag1:
                TempOrder = PlaceOrder[0][::-1]
                CurrentSplitNum = len(TempOrder)

                del PlaceOrder[0]
                for SplitNodes in TempOrder:
                    PlaceOrder.insert(0, [SplitNodes])
            else:
                CurrentSplitNum = 0
                del PlaceOrder[0]
                for ReGrouped in TempMappingPriority[0][::-1]:
                    if ReGrouped in TempGroups:
                        PlaceOrder.insert(0, list(TempGroups[ReGrouped]))

                        CurrentSplitNum += 1
        else:
            ChipIdx = PlacedIdx

            VacantSpace[ChipIdx] -= CurrentGroupSize
            AfterPlaced[ChipIdx] += PlaceOrder[0]
            del PlaceOrder[0]

            CurrentSplitNum -= 1
            if CurrentSplitNum == 0:
                GlobalIdx += 1
                TempLevel = 1

        if GroupIdx == len(PlaceOrder):
            break

    NewIdx = np.zeros(allnodenum, dtype=np.int) - 1
    for idx, cm in enumerate(AfterPlaced):
        for mem in cm:
            NewIdx[mem] = idx

    NewA = np.zeros([comm_num, comm_num])

    for idx1 in range(allnodenum):
        for idx2 in range(allnodenum):
            if adjmat_origin[idx1, idx2] != 0:
                if NewIdx[idx1] != NewIdx[idx2]:
                    NewA[NewIdx[idx1], NewIdx[idx2]] += adjmat_origin[idx1, idx2]
    PlacedChipData = FindBestChipPlacement(AfterPlaced, NewA, comm_num, chip_x)

    NewChipPos = np.zeros(allnodenum)
    for chipidx, cm in enumerate(PlacedChipData):
        for mem in cm:
            NewChipPos[mem] = chipidx + 1

    config.params_dict['assignedchipidx'] = NewChipPos

    # # Make New Index of Members
    # NewPos = np.zeros(lut.shape[1], dtype=np.int) - 1
    # for ChipIdx, Comm in enumerate(PlacedChipData):
    #     for core_idx, mem in enumerate(Comm):
    #         NewPos[mem] = int(ChipIdx * max_member_num + core_idx)
    #
    # # Save New LUT
    # new_lut = np.zeros(lut.shape, dtype=np.int)
    # for pre, post in enumerate(NewPos):
    #     if post != -1:
    #         new_lut[:, post] = lut[:, pre]
    #
    # for idx1, line in enumerate(new_lut):
    #     for idx2, pre in enumerate(line):
    #         if pre != 0:
    #             new_lut[idx1, idx2] = NewPos[pre - 1] + 1
    #
    # NewA = utils.lut_to_adjmat(config, new_lut)[0]
    # ModA = np.zeros([comm_num, comm_num], dtype=np.int)
    # for i in range(comm_num):
    #     for j in range(comm_num):
    #         if i != j:
    #             p = NewA[i * max_member_num:(i + 1) * max_member_num, j * max_member_num:(j + 1) * max_member_num]
    #             ModA[i, j] = np.sum(p)
    #
    # config.params_dict['lut'] = new_lut.T - 1
    #
    # print("\n==>  End DETECTING COMMUNITIES ...\n")
    #
    # return np.transpose(new_lut) - 1, new_lut


# # draw the partitioned graph
'''
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = colormp.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
'''

# # Plot Modularity Curve
# plt.subplots_adjust(hspace=0.5, wspace=0.3)
# plt.plot(list(range(1, len(log_m) + 1)), log_m)
# plt.xlabel('step')
# plt.ylabel('modularity')
# plt.title("unweighted")
# plt.savefig("Q_curve.pdf")
#
# """ Graph G Plot """
# pos = nx.kamada_kawai_layout(G)
# fig = plt.figure(figsize=(7, 6))
# node_size = 100
# node_color_list = []
# # max_k, 즉 Modularity 가 가장 높은 지점의 Community k 별로 색깔 설정
# # node_color_list.append(~~~)
# im = nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=node_size)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
# plt.xticks([])
# plt.yticks([])
# plt.show()

# FinalComp = runNsizeCom(BestComp, adjmat)
# # Make New Index of Members
# new_pos = np.zeros(lut.shape[1], dtype=np.int)
# for chip_idx, cm in enumerate(FinalComp):
#     for core_idx, mem in enumerate(cm):
#         new_pos[mem] = int(chip_idx * max_member_num + core_idx)
#
# # Save New LUT
# new_lut = np.zeros(lut.shape, dtype=np.int)
# for pre, post in enumerate(new_pos):
#     new_lut[:, post] = lut[:, pre]
# for idx1, line in enumerate(new_lut):
#     for idx2, pre in enumerate(line):
#         if pre != 0:
#             new_lut[idx1, idx2] = new_pos[pre] + 1

# TEST LUT
# lut = np.zeros([512, 10])
# lut[:64, 0] = 5
# lut[:64, 1] = 5
# lut[:64, 2] = 5
# lut[:64, 3] = 5
#
# lut[:16, 4] = 6
# # lut[:32, 5] = 5
#
# lut[:64, 5] = 7
# lut[64:128, 5] = 8
# lut[128:192, 5] = 9
# lut[192:256, 5] = 10

# partition2 = lvcm.induced_graph(partition, G, weight='weight')          # partition으로 나눠진 graph를 return
"""
    Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights of the links between their elements is w
"""

# partition3 = lvcm.generate_dendrogram(G, part_init=None, weight='weight', resolution=1.0, randomize=None, random_state=None)
"""
    Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger are the communities
"""

# partition4 = lvcm.partition_at_level(partition3, 1)
"""
    Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger are the communities
"""

# MOD = lvcm.modularity(partition, G, weight='weight')
"""
    Compute the modularity of a partition of a graph
"""
