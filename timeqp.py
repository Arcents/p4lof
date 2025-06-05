import matplotlib.pyplot as plt
import networkx as nx

class Hypergraph(nx.Graph):
    def __init__(self):
        super().__init__()
        self.hyperedges = []

    def add_hyperedge(self, hyperedge):
        """
        添加超边到超图中。
        :param hyperedge: 超边，表示为一个节点列表。
        """
        for i in range(len(hyperedge) - 1):
            self.add_edge(hyperedge[i], hyperedge[i + 1])
        self.hyperedges.append(hyperedge)

    def degree(self, node):
        """
        获取节点的度数。
        :param node: 节点。
        :return: 节点的度数。
        """
        return super().degree(node)

    def max_degree_node(self):
        """
        获取度数最大的节点。
        :return: 度数最大的节点。
        """
        if not self.nodes:
            return None
        return max(self.nodes, key=self.degree)

    def load_from_file(self, file_path):
        """
        从文件中加载超图数据。
        :param file_path: 文件路径。
        """
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    node1, node2, weight = parts
                    self.add_hyperedge([node1, node2])

    def plot_hypergraph(self):
        """
        绘制超图。
        """
        pos = nx.spring_layout(self)
        nx.draw(self, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Hypergraph Visualization")
        plt.show()

    def plot_filtered_hypergraph(self):
        """
        绘制过滤后的超图，只包含度数大于等于2的节点及其对应的边。
        """
        filtered_nodes = [node for node in self.nodes if self.degree(node) >= 2]
        filtered_edges = [(u, v) for u, v in self.edges if u in filtered_nodes and v in filtered_nodes]

        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(filtered_nodes)
        G_filtered.add_edges_from(filtered_edges)

        pos = nx.spring_layout(G_filtered)
        nx.draw(G_filtered, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("Filtered Hypergraph Visualization (Nodes with Degree >= 2)")
        plt.show()

if __name__ == "__main__":
    hg = Hypergraph()
    hg.load_from_file('dataset/weights-dist/1239/weights.intra')
    print("度数最大的节点:", hg.max_degree_node())
    hg.plot_hypergraph()
    hg.plot_filtered_hypergraph()
