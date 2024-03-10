import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname
from math import floor
import matplotlib as mpl
import numpy as np
import itertools

def main():
    script_path = dirname(realpath(__file__))
    with open('data/USA/city_locations.json', 'r') as city_file:
        city_locations = json.load(city_file)
    routes_df = pd.read_csv('data/USA/routes.csv')
    tickets_df = pd.read_csv('data/USA/tickets.csv')

    graph = nx.Graph()
    for ind, city_A, city_B, dist, col in routes_df.itertuples():
        attr = {
            'dist': dist,
            'col': col,
            'usage': 0
        }
        graph.add_edge(city_A, city_B, **attr)

    shortest_paths = []
    for ind, city_A, city_B, point in tickets_df.itertuples():
        # shortest_paths.extend(list(nx.all_shortest_paths(graph, city_A, city_B, 'dist')))
        shortest_paths.extend(itertools.islice(nx.shortest_simple_paths(graph, city_A, city_B, 'dist'), 6))
        break

    for path in shortest_paths:
        print(nx.path_weight(graph, path, 'dist'))
        for idx in range(len(path)-1):
            graph.edges[path[idx], path[idx+1]]['usage'] += 1

    fig, ax = plt.subplots(1,1)
    img = plt.imread(join(script_path, '..', 'data', 'USA', 'USA_map.jpg'))
    ax.imshow(img, alpha=0.5)
    for key, value in city_locations.items():
        x = floor(value[0]*img.shape[1])
        y = floor((1-value[1])*img.shape[0])
        city_locations.update({key: [x , y]})
    # options = {
    #     'font_weight': 1,
    #     'node_size': 80
    # }
    # nx.draw(graph, city_locations, ax=ax, **options)

    edgewidth = [graph.get_edge_data(u, v)['dist'] for u, v in graph.edges()]
    edgecolor = [graph.get_edge_data(u, v)['usage'] for u, v in graph.edges()]
    min_visited = min(edgecolor)
    max_visited = max(edgecolor)
    cmap = plt.cm.get_cmap('cool', 50)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    true_tcks = list(set(edgecolor))
    tcks =  [t/max(edgecolor) for t in true_tcks]
    print(tcks)

    cbar = plt.colorbar(sm, ticks=tcks,
                        boundaries=np.linspace(0, 1, 100))
    # cbar.ax.set_yticklabels([str(min_visited), str(max_visited)],
    #                         rotation='vertical')
    cbar.ax.set_yticklabels(true_tcks,
                            rotation='vertical')
    cbar.ax.set_ylabel(r'Route usage')

    attr = {
        'node_size': 80
    }
    nx.draw_networkx_nodes(graph, city_locations, **attr)
    attr = {
        'width': 5,
        "edge_color": edgecolor,
        "edge_vmin": min(edgecolor),
        "edge_vmax": max(edgecolor),
        "edge_cmap": cmap
    }
    nx.draw_networkx_edges(graph, city_locations, **attr)
    plt.show()

if __name__ == "__main__":
    main()