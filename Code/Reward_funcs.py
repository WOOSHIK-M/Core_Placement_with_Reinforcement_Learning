import math
import numpy as np


def Deadlock_Constraint(reward, node, x_vec, y_vec, deadlock_coef=0.0, deadlockCons=0):
    deadlock_reward = 0.0
    for i in range(len(node)):
        if node[i].is_mul:
            mulx, muly = x_vec[i], y_vec[i]

            fromc = node[i].connected_from_cores
            toc = node[i].multi_to_cores[0]

            to_flag = False
            tox, toy = x_vec[toc], y_vec[toc]
            if tox != mulx:
                to_flag = True

            from_flag = False
            if to_flag:
                for sc in fromc:
                    fromx, fromy = x_vec[sc], y_vec[sc]
                    if fromy > muly:
                        from_flag = True
                    if from_flag:
                        deadlock_reward += 1
                        from_flag = False
    deadlock_reward /= deadlockCons + 1e-7
    reward *= math.exp(deadlock_reward * deadlock_coef)

    return reward


def Communication_Cost(node, x_vec, y_vec, deadlock_control=False, deadlock_coef=0.0, deadlockCons=0):
    reward = 0.0
    for sc in range(len(node)):
        connected_to_cores = node[sc].connected_to_cores
        packets = node[sc].connected_to_packets
        sx, sy = x_vec[sc], y_vec[sc]

        for idx, dc in enumerate(connected_to_cores):
            dx, dy = x_vec[dc], y_vec[dc]

            reward += (math.fabs(sx - dx) + math.fabs(sy - dy)) * packets[idx]
            while node[dc].is_mul == 1:
                sx, sy = dx, dy
                dc = node[dc].multi_to_cores[0]
                dx, dy = x_vec[dc], y_vec[dc]
                reward += (math.fabs(sx - dx) + math.fabs(sy - dy)) * packets[idx]

    # DEADLOCK PENALTY
    if deadlock_control:
        reward = Deadlock_Constraint(reward, node, x_vec, y_vec, deadlock_coef, deadlockCons)

    return reward


def Split_Directions_Cost(node, x_vec, y_vec, deadlock_control=False, deadlock_coef=0.0, deadlockCons=0):
    directions = np.zeros(4)  # left, right, up, down
    for sc in range(len(node)):
        connected_to_cores = node[sc].connected_to_cores
        packets = node[sc].connected_to_packets
        sx, sy = x_vec[sc], y_vec[sc]

        for idx, dc in enumerate(connected_to_cores):
            dx, dy = x_vec[dc], y_vec[dc]

            if sx > dx:
                directions[0] += math.fabs(sx - dx) * packets[idx]
                if sy > dy:
                    directions[2] += math.fabs(sy - dy) * packets[idx]
                elif sy < dy:
                    directions[3] += math.fabs(sy - dy) * packets[idx]
            elif sx < dx:
                directions[1] += math.fabs(sx - dx) * packets[idx]
                if sy > dy:
                    directions[2] += math.fabs(sy - dy) * packets[idx]
                elif sy < dy:
                    directions[3] += math.fabs(sy - dy) * packets[idx]
            else:
                if sy > dy:
                    directions[2] += math.fabs(sy - dy) * packets[idx]
                else:
                    directions[3] += math.fabs(sy - dy) * packets[idx]
            while node[dc].is_mul == 1:
                sx, sy = dx, dy
                dc = node[dc].multi_to_cores[0]

                dx, dy = x_vec[dc], y_vec[dc]
                if sx > dx:
                    directions[0] += math.fabs(sx - dx) * packets[idx]
                    if sy > dy:
                        directions[2] += math.fabs(sy - dy) * packets[idx]
                    elif sy < dy:
                        directions[3] += math.fabs(sy - dy) * packets[idx]
                elif sx < dx:
                    directions[1] += math.fabs(sx - dx) * packets[idx]
                    if sy > dy:
                        directions[2] += math.fabs(sy - dy) * packets[idx]
                    elif sy < dy:
                        directions[3] += math.fabs(sy - dy) * packets[idx]
                else:
                    if sy > dy:
                        directions[2] += math.fabs(sy - dy) * packets[idx]
                    else:
                        directions[3] += math.fabs(sy - dy) * packets[idx]

    reward = np.max(directions[:2]) + np.max(directions[2:]) + np.sum(directions)

    # DEADLOCK PENALTY
    if deadlock_control:
        reward = Deadlock_Constraint(reward, node, x_vec, y_vec, deadlock_coef, deadlockCons)

    return reward
