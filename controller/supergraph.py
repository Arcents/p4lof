
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SuperGraph(nx.Graph):
    def __init__(self, datapath=None):
        """初始化超图"""
        super().__init__()
        self.hyperedges = {}  # 存储超边 {edge_id: set(nodes)}

        if datapath is not None:
            self.read_dataset(datapath)
        else:
            print(f"警告： SuperGraph 未利用数据集初始化超图")


    def read_dataset(self, datapath):
        """读取网络拓扑数据文件
        Args:
            datapath: 数据文件路径
        格式:
            node1 node2 weight
        """
        try:
            with open(datapath, 'r') as f:
                for line in f:
                    # 跳过注释行和空行
                    if line.startswith('#') or not line.strip():
                        continue
                        
                    # 分割每行数据
                    items = line.strip().split()
                    if len(items) >= 3:
                        node1, node2, weight = items[0], items[1], float(items[2])
                        
                        # 添加节点
                        self.add_node(node1)
                        self.add_node(node2)
                        
                        # 添加带权重的边
                        self.add_edge(node1, node2, weight=weight)
        except FileNotFoundError:
            print(f"错误: read_dataset 找不到文件 {datapath}")
        except ValueError as e:
            print(f"错误: read_dataset 数据格式不正确 - {e}")
        except Exception as e:
            print(f"错误: read_dataset {e}")


    def add_hyperedge(self, edge_id, nodes, **attr):
        """添加超边
        Args:
            edge_id: 超边ID
            nodes: 节点集合
            attr: 超边属性
        """
        if not all(node in self.nodes for node in nodes):
            raise ValueError("所有节点必须先存在于图中")
        self.hyperedges[edge_id] = {'nodes': set(nodes), 'attr': attr}


    def get_hyperedges(self):
        """获取所有超边"""
        return self.hyperedges


    def draw_hypergraph(self, node_size=500):
        """绘制超图
        使用不同颜色标识超边连接的节点
        """

        # 设置更大的画布
        plt.figure(figsize=(15, 12))
        
        # 使用带参数的弹簧布局
        pos = nx.spring_layout(
            self,
            k=0.3,         # 增加节点间斥力
            iterations=50, # 增加迭代次数
            scale=2        # 增加布局scale
        )
        
        # 绘制节点和边，增加间距
        nx.draw_networkx_nodes(
            self, 
            pos, 
            node_size=node_size,
            node_color='lightblue',
            alpha=0.6
        )
        
        nx.draw_networkx_edges(
            self, 
            pos,
            alpha=0.4,
            width=1
        )
        
        # 添加节点标签，调整位置偏移
        nx.draw_networkx_labels(
            self, 
            pos,
            font_size=6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.1)
        )
        
        # 绘制基本图结构
        nx.draw_networkx_nodes(self, pos, node_size=node_size)
        nx.draw_networkx_edges(self, pos)
        # nx.draw_networkx_labels(self, pos)
        
        # 为每个超边使用不同颜色标记节点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.hyperedges)))
        for (edge_id, edge_data), color in zip(self.hyperedges.items(), colors):
            nodes = edge_data['nodes']
            nx.draw_networkx_nodes(self, pos, 
                                 nodelist=nodes,
                                 node_color=[color], 
                                 node_size=node_size)

        plt.axis('off')
        plt.show()

if __name__ == "__main__":

    # 使用示例
    datapath = "./dataset/weights-dist/1221/weights.intra"
    
    g = SuperGraph(datapath)
    g.draw_hypergraph()

