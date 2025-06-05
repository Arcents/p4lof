
import os
import re
import copy
import time
import random
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import TopoParserConfig


class TopoParser:
    def __init__(self, config=TopoParserConfig()):

        self.config = config

        self.num_nodes = 0
        self.num_edges = 0
        self.topo_fd = f"{config.datadir}{config.topo_fn}/"
        self.topo_fp = f"{self.topo_fd}{config.topo_fn}"
        self.demand_fns = None

        self.graph = TopoGraph()
        self.parseTopology()

        self.demand_mtxs = {}
        self.parseDemand()

        # 设置当前的TM
        self.cur_demand_mark = None
        self.cur_demand_mtx = None
        self.setRandomCurDMtx()

        self.printTopoInfo()

        self.num_cdtpaths = 0
        self.cdtpaths = self.calShortestPaths()

        self.total_start = time.time()
        self.update_deps = self.getDeps()
        self.deps_end = time.time()
        self.topo_time = self.deps_end - self.total_start
        # self.figureUpdateTimes()
        self.figureNumCycles()
        self.figureTime()
        # self.figureTopoNumCycle()

    def printTopoInfo(self):
        degrees = dict(self.graph.degree())
        leaf_nodes = [node for node, degree in degrees.items() if degree <= 2]
        avg_degree = 2 * self.num_edges / self.num_nodes
        print(len(leaf_nodes), avg_degree)


    def figureTime(self):

        all_update_deps = [sublist for sublist in self.update_deps if len(sublist) <= self.config.num_updates]
        num_sample = self.config.num_updates \
                     if self.config.num_updates <= len(all_update_deps) \
                     else len(all_update_deps)
        if num_sample == len(all_update_deps):
            print("Warning: Full sampling! ")
        total_time_mtx = np.zeros((self.config.rep_times, self.config.deps_len))
        core_time_mtx = np.zeros((self.config.rep_times, self.config.deps_len))
        for tep_time_i in range(self.config.rep_times):
            update_deps = random.sample(all_update_deps, num_sample)
            print(update_deps)
            avg_update_deps = 0
            for dep in update_deps:
                avg_update_deps += len(dep)
            avg_update_deps /= len(update_deps)
            greedy_start_time = time.time()
            _ = self.greedyTimes(update_deps) / len(update_deps)
            core_time_mtx[tep_time_i, 0] = time.time() - greedy_start_time
            total_time_mtx[tep_time_i, 0] = core_time_mtx[tep_time_i, 0] + self.topo_time
            for i in range(1, self.config.deps_len):
                winlen_time = time.time()
                _, _ = self.windowTimes(update_deps, win_len=i+1)
                core_time_mtx[tep_time_i, i] = time.time() - winlen_time
                total_time_mtx[tep_time_i, i] = core_time_mtx[tep_time_i, i] + self.topo_time
        np.savetxt(f"rst/total_time_{self.config.topo_fn}.txt", total_time_mtx)
        np.savetxt(f"rst/core_time_{self.config.topo_fn}.txt", core_time_mtx)
        print(total_time_mtx)


    def figureNumCycles(self):
        all_update_deps = [sublist for sublist in self.update_deps if len(sublist) <= self.config.num_updates]
        num_sample = self.config.num_updates \
                     if self.config.num_updates <= len(all_update_deps) \
                     else len(all_update_deps)
        if num_sample == len(all_update_deps):
            print("Warning: Full sampling! ")


        num_cycles_mtx = np.zeros((self.config.rep_times, self.config.deps_len))
        for tep_time in range(self.config.rep_times):
            update_deps = random.sample(all_update_deps, num_sample)
            # print(update_deps)
            avg_update_deps = 0
            for dep in update_deps:
                avg_update_deps += len(dep)
            avg_update_deps /= len(update_deps)
            num_cycles_mtx[tep_time, 0] = 0
            for i in range(1, self.config.deps_len):
                num_cycles_mtx[tep_time, i] = self.windowTimes(update_deps, win_len=i+1)[1] #\
                                                # / len(update_deps)
        np.savetxt(f"rst/num_cycles_{self.config.topo_fn}.txt", num_cycles_mtx)
        print(num_cycles_mtx)


    def figureTopoNumCycle(self):
        counts = Counter(len(sublist) for sublist in self.update_deps)
        count_np = np.zeros((max(counts.keys())+1, 2), dtype=int)
        for i in range(max(counts.keys())+1):
            count_np[i, 0] = i
        for key in counts.keys():
            count_np[key, 1] = counts[key]
        np.savetxt(f"rst/topocount_{self.config.topo_fn}.txt", count_np)

    def figureUpdateTimes(self):
        all_update_deps = [sublist for sublist in self.update_deps if len(sublist) <= self.config.num_updates]
        num_sample = self.config.num_updates \
                     if self.config.num_updates <= len(all_update_deps) \
                     else len(all_update_deps)
        if num_sample == len(all_update_deps):
            print("Warning: Full sampling! ")

        update_times_mtx = np.zeros((self.config.rep_times, self.config.deps_len))
        for tep_time in range(self.config.rep_times):
            update_deps = random.sample(all_update_deps, num_sample)
            # print(update_deps)
            avg_update_deps = 0
            for dep in update_deps:
                avg_update_deps += len(dep)
            avg_update_deps /= len(update_deps)
            update_times_mtx[tep_time, 0] = self.greedyTimes(update_deps) \
                                            # / len(update_deps)# / avg_update_deps
            for i in range(1, self.config.deps_len):
                update_times_mtx[tep_time, i] = self.windowTimes(update_deps, win_len=i+1)[0] \
                                                # / len(update_deps)# / avg_update_deps
        np.savetxt(f"rst/update_times_{self.config.topo_fn}.txt", update_times_mtx)
        print(update_times_mtx)


    def parseTopology(self):
        print("\nloading topology file...", self.topo_fp)
        with open(self.topo_fp, 'r') as f:
            # 读取文件首行，解析节点数与边数
            first_line = f.readline()
            nodes_ptn = r"Node_num:\s*(\d+)"
            nodes_mch = re.search(nodes_ptn, first_line)
            self.num_nodes = int(nodes_mch.group(1)) if nodes_mch else None
            edges_ptn = r"Edge_num:\s*(\d+)"
            edges_mch = re.search(edges_ptn, first_line)
            self.num_edges = int(edges_mch.group(1)) if edges_mch else None

            # 添加节点
            self.graph.add_nodes_from(list(range(self.num_nodes)))
            # 添加边
            f.readline()
            for line in f:
                eid, source, target, weight, capability, _ = re.split(r"[\t\n]", line)
                self.graph.add_edge(int(source), int(target),
                                    weight=int(weight), capability=int(capability))

    def parseDemand(self):
        self.demand_fns = os.listdir(self.topo_fd)
        self.demand_fns.remove(self.config.topo_fn)
        print("loading demand file...", self.demand_fns)
        for demand_fn in self.demand_fns:
            demand_fp = f"{self.topo_fd}{demand_fn}"
            demand_mtx = np.loadtxt(demand_fp)
            demand_mtx = demand_mtx.reshape((-1, self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                demand_mtx[:, i, i] = np.zeros((demand_mtx.shape[0]))
            self.demand_mtxs[demand_fn] = demand_mtx

    def plotTopology(self, node_size=1000):
        """绘制超图
        使用不同颜色标识超边连接的节点
        """

        # 设置更大的画布
        plt.figure(figsize=(10, 8))

        # 使用带参数的弹簧布局
        pos = nx.spring_layout(
            self.graph,
            k=0.3,  # 增加节点间斥力
            iterations=50,  # 增加迭代次数
            scale=2  # 增加布局scale
        )

        # 绘制节点和边，增加间距
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=node_size,
            node_color="blue",
            alpha=0.6
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            alpha=0.4,
            width=2
        )

        # 添加节点标签，调整位置偏移
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.1)
        )

        # 绘制基本图结构
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size)
        nx.draw_networkx_edges(self.graph, pos)
        # nx.draw_networkx_labels(self, pos)

        # 为每个超边使用不同颜色标记节点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.graph.edges)))
        for (edge_id, edge_data), color in zip(self.graph.edges.items(), colors):
            nodes = self.graph.nodes
            nx.draw_networkx_nodes(self.graph, pos,
                                   nodelist=nodes,
                                   node_color=[color],
                                   node_size=node_size)
        plt.axis("off")
        plt.show()

    def setRandomCurDMtx(self):
        # 随机选择一个流量
        randomkey = random.choices(list(self.demand_mtxs.keys()), k=1)[0]
        demand_mtx = self.demand_mtxs[randomkey]
        num_demands = demand_mtx.shape[0]
        ri_demand = random.randint(0, num_demands - 1)
        self.cur_demand_mtx = demand_mtx[ri_demand, :, :].copy()
        self.cur_demand_mark = np.zeros(shape=self.cur_demand_mtx.shape, dtype=np.int32)

    def calShortestPaths(self):
        self.num_cdtpaths = 1
        pathgenerator = nx.all_pairs_all_shortest_paths(self.graph, weight="weight", method="dijkstra")
        self.cdtpaths = dict(pathgenerator)
        self._reshapeCdtPaths()
        return self.cdtpaths

    def _reshapeCdtPaths(self):
        for source in self.cdtpaths:
            for target in self.cdtpaths[source]:
                if source  ==  target:
                    continue
                n = self.num_cdtpaths if len(self.cdtpaths[source][target]) >= self.num_cdtpaths \
                    else len(self.cdtpaths[source][target])
                self.cdtpaths[source][target] = self.cdtpaths[source][target][:n]

    def getCdtPaths(self, source, target):
        return self.cdtpaths[source][target]

    def getCustoizedTL(self, spt_mtx, demand_mtx=None):
        if demand_mtx is None:
            demand_mtx = self.cur_demand_mtx
        nodetload = np.zeros((self.num_nodes,), dtype=float)
        edgetload = {sd: 0.0 for sd in dict(self.graph.edges).keys()}
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                if source == target:
                    continue
                # 计算ECMP路径，其中路径cost为1而非weight
                ecmpaths = self.getCdtPaths(source, target)
                load_spt = spt_mtx[source, target, :len(ecmpaths)]
                load_spt = load_spt / load_spt.sum()
                for pi in range(len(ecmpaths)):
                    load = demand_mtx[source, target] * load_spt[pi]
                    for ni in range(len(ecmpaths[pi])):
                        # 添加节点负载
                        node = ecmpaths[pi][ni]
                        nodetload[node] += load
                        if ni != 0:
                            # 添加链路负载
                            lastnode = ecmpaths[pi][ni - 1]
                            edgetload[(lastnode, node)] += load
        return nodetload, edgetload

    def getTopkEdges(self, k=3):
        edge_pass_rcd = {}
        for source in range(self.num_nodes):
            for target in range(self.num_nodes):
                if source == target:
                    continue
                cdtpath = self.getCdtPaths(source, target)[0]
                for idx in range(len(cdtpath)-1):
                    node1 = cdtpath[idx]
                    node2 = cdtpath[idx+1]
                    # 由于在统计过程中，图转换为无环图考虑，因而统一key
                    if node2 < node1:
                        node1, node2 = node2, node1
                    if (node1, node2) not in edge_pass_rcd:
                        edge_pass_rcd[(node1, node2)] = 0
                    edge_pass_rcd[(node1, node2)] += 1
        sorted_items = sorted(edge_pass_rcd.items(), key=lambda x: x[1], reverse=True)
        k = min(k, len(sorted_items))
        topk_edges = [key for key, value in sorted_items[:k]]
        return topk_edges

    @staticmethod
    def findLongestCycles(graph):
        cycles = list(nx.simple_cycles(graph))
        if not cycles:
            return []
        max_len = max(len(c) for c in cycles)
        longest_cycles = [c for c in cycles if len(c) == max_len]
        return longest_cycles

    def getDeps(self):
        # 计算经过数量最大的几条边
        topk_edges = self.getTopkEdges()
        self.graph.remove_edges_from(topk_edges)
        oldcdtpaths = self.cdtpaths
        self.cdtpaths = self.calShortestPaths()

        breakflag = False
        i_updates = 0
        num_updates = self.config.num_updates
        update_deps = []
        for source in range(self.num_nodes):
            for target in range(self.num_nodes):
                if source == target:
                    continue
                oldcdtpath = oldcdtpaths[source][target][0]
                for newsource in range(self.num_nodes):
                    for newtarget in range(self.num_nodes):
                        if newsource == newtarget:
                            continue
                        if not newtarget in self.cdtpaths[newsource]:
                            continue
                        newcdtpath = self.cdtpaths[newsource][newtarget][0]
                        update_graph = nx.DiGraph()
                        update_graph.add_nodes_from(oldcdtpath)
                        update_graph.add_nodes_from(newcdtpath)
                        update_edges = [(oldcdtpath[idx], oldcdtpath[idx+1])
                                        for idx in range(len(oldcdtpath)-1)]
                        update_edges += [(newcdtpath[idx], newcdtpath[idx+1])
                                        for idx in range(len(newcdtpath)-1)]
                        update_graph.add_edges_from(update_edges)
                        longest_cycles = self.findLongestCycles(update_graph)
                        if len(longest_cycles) == 0:
                            continue
                        if len(longest_cycles[0]) >= 2:
                            if longest_cycles[0] in update_deps:
                                continue
                            update_deps.append(longest_cycles[0])
                            i_updates += 1
                            if i_updates == num_updates * 160:
                                return update_deps
                        breakflag = True
                        break
                    if breakflag:
                        breakflag = False
                        break
                # if breakflag:
                #     breakflag = False
                #     break
        return update_deps

    @staticmethod
    def greedyTimes(update_deps):
        update_times = 0
        update_deps = copy.deepcopy(update_deps)
        while len(update_deps) > 0:
            # 获取第一列字段
            nodeps_list = []
            for i in range(len(update_deps)):
                nodeps_list.append(update_deps[i][0])
            counter = Counter(nodeps_list)
            max_count = max(counter.values())
            most_common = [k for k, v in counter.items() if v == max_count]

            for i in range(len(update_deps)):
                if update_deps[i][0] == most_common[0]:
                    update_deps[i].pop(0)
            for i in range(len(update_deps)-1, -1, -1):
                if len(update_deps[i]) == 0:
                    update_deps.pop(i)
            update_times += 1
        return update_times

    @staticmethod
    def windowTimes(update_deps, win_len=2):
        update_times = 0
        cycles = set()
        update_deps = copy.deepcopy(update_deps)
        while len(update_deps) > 0:
            nodeps_list = []
            # 生成依赖图
            graph = nx.DiGraph()
            for i in range(len(update_deps)):
                deps = update_deps[i][:win_len] \
                    if len(update_deps[i]) > win_len else update_deps[i]
                nodeps_list += deps

                for j in range(len(deps)-1):
                    node1 = deps[j]
                    node2 = deps[j+1]
                    graph.add_edge(node1, node2)
            #
            for cycle in nx.simple_cycles(graph):
                cycles.add(tuple(TopoParser.standardize_cycle(cycle)))

            counter = Counter(nodeps_list)
            max_count = max(counter.values())
            most_common = [k for k, v in counter.items() if v == max_count]

            stack = [most_common[0]]
            TopoParser.getBeforeDeps(stack, most_common[0], update_deps, win_len)

            for _ in range(len(stack)):
                del_deps = stack.pop()
                for i in range(len(update_deps)):
                    if update_deps[i][0] == del_deps:
                        update_deps[i].pop(0)
                for i in range(len(update_deps) - 1, -1, -1):
                    if len(update_deps[i]) == 0:
                        update_deps.pop(i)
                update_times += 1
        return update_times, len(cycles)

    @staticmethod
    def getBeforeDeps(stack, dep, update_deps, win_len):
        for i in range(len(update_deps)):
            deps = update_deps[i][:win_len] \
                if len(update_deps[i]) > win_len else update_deps[i]
            indices = [i for i, v in enumerate(deps) if v == dep]
            if len(indices) > 0:
                idx = indices[0]
                if idx > 0:
                    stack += update_deps[i][idx-1::-1]

    @staticmethod
    def standardize_cycle(cycle):
        # 找到最小节点的位置
        min_node = min(cycle)
        min_index = cycle.index(min_node)
        # 从最小节点开始旋转环
        return list(cycle[min_index:] + cycle[:min_index])


class TopoGraph(nx.Graph):
    def __init__(self):
        super(TopoGraph, self).__init__()


if __name__ == "__main__":
    topo_parser = TopoParser()
    topo_parser.plotTopology()
