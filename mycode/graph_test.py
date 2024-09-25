import networkx as nx
import matplotlib.pyplot as plt



def plot_relation_graph(data):
    G = nx.DiGraph()
    edge_labels = {}
    # 遍历dict，创建图中的节点和边
    for node, subnodes in data.items():
        for subnode, relations in subnodes.items():
            G.add_node(node)
            rel_node, rel_subnode = relations
            edge_labels[(node, rel_node)] = f"{subnode} -> {rel_subnode}" if subnode != rel_subnode else f"{rel_subnode}"
            G.add_edge(node, rel_node)

    # 绘制图
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G,  k=0.5, iterations=50)  # 布局
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold',
            arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8)
    plt.title("2-level Node Relation Graph")
    plt.tight_layout()
    plt.show()


# 示例 dict
# data = {
#     "A": {
#         "A1": [("B", "B1"), ("C", "C1")],
#         "A2": [("B", "B2")]
#     },
#     "B": {
#         "B1": [("A", "A1")],
#         "B2": [("C", "C2")]
#     }
# }
from test_20_tables import relation_ships as data
plot_relation_graph(data)
