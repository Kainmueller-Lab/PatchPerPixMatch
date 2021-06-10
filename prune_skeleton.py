import networkx as nx
import os
from glob import glob
import math
import argparse
import sys


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-folder", type=str, dest="in_folder",
        required=True
    )
    parser.add_argument(
        "--prune-iter", type=int, dest="prune_iter"
    )
    parser.add_argument(
        "--prune-length", type=int, dest="prune_length"
    )
    parser.add_argument(
        "--scale-factor", type=float, dest="scale_factor"
    )
    parser.add_argument(
        "--out-folder", type=str, dest="out_folder"
    )
    parser.add_argument(
        "--num-worker", type=int, dest="num_worker",
        default=1
    )

    args = parser.parse_args()
    return args


def create_graph_from_swc(fn, scale_factor=1):
    # create graph
    graph = nx.Graph()
    # open swc file
    f = open(fn, "r")
    # create graph node for each swc line
    for line in f:
        line = line.replace("\n", "")
        if line == "":
            continue
        if line[0] == "#":
            continue
        if line[0] == " ":
            continue
        split = line.split(" ")
        node_id = int(split[0])
        node_type = int(float(split[1]))
        x = float(split[2]) * scale_factor
        y = float(split[3]) * scale_factor
        z = float(split[4]) * scale_factor
        diameter = float(split[5]) * scale_factor
        parent_id = int(split[6])

        graph.add_node(
            node_id,
            x=x, y=y, z=z,
            parent_id=parent_id,
            node_type=node_type,
            diameter=diameter
        )
        if parent_id != -1:
            graph.add_edge(node_id, parent_id)

    f.close()

    return graph


def write_swc_line(graph, cnode, cparent, f, cnode_relabeled=-1):
    # output skeleton to swc
    # node_id, node_type, x, y, z, diameter, parent_id
    if cnode_relabeled != -1:
        cnode_name = cnode_relabeled
    else:
        cnode_name = cnode
    line = "%i %i %f %f %f %f %i\n" % (
        cnode_name,
        graph.nodes[cnode]['node_type'],
        graph.nodes[cnode]['x'],
        graph.nodes[cnode]['y'],
        graph.nodes[cnode]['z'],
        graph.nodes[cnode]['diameter'],
        cparent
    )
    f.write(line)


def traverse_graph(graph, cnode, cparent, visited, f, cnode_relabeled=-1):
    write_swc_line(graph, cnode, cparent, f, cnode_relabeled)
    visited.append(cnode)
    if cnode_relabeled != -1:
        node_cnt = cnode_relabeled
    else:
        node_cnt = cnode
    for neighbor in graph.neighbors(cnode):
        if neighbor not in visited:
            if cnode_relabeled != -1:
                visited, cnt = traverse_graph(
                    graph, neighbor, cnode_relabeled, visited, f,
                    node_cnt + 1
                )
                node_cnt = cnt
            else:
                visited, cnt = traverse_graph(
                    graph, neighbor, cnode, visited, f)
                node_cnt = neighbor

    return visited, node_cnt


def write_swc(graph, outfn):
    print(outfn)
    sys.setrecursionlimit(5000)
    f = open(outfn, "w")
    node_id_cnt = 1
    visited = []

    # iterate through connected components
    for cc in nx.connected_components(graph):
        # find start point
        nodes = sorted(list(cc))
        start_node = nodes[0]
        for node in nodes:
            if nx.degree(graph, node) == 1:
                start_node = node
                break
        visited, cnt = traverse_graph(
            graph, start_node, -1, visited, f, node_id_cnt
        )
        node_id_cnt = len(visited) + 1

    f.close()


def get_n_degree_nodes(graph, degree):
    nodes = []
    for node_id in graph.nodes():
        if nx.degree(graph, node_id) == degree:
            nodes.append(node_id)

    return nodes


def get_ge_n_degree_nodes(graph, degree):
    nodes = []
    for node_id in graph.nodes():
        if nx.degree(graph, node_id) >= degree:
            nodes.append(node_id)

    return nodes


def get_branch_length(graph, node):
    distance = 0
    path = []
    # check if node is end point
    assert nx.degree(graph, node) == 1, \
        "Node % i is no endpoint. Please check!" % node
    path.append(node)
    tmp = list(graph.neighbors(node))[0]

    while len(list(graph.neighbors(tmp))) == 2:
        path.append(tmp)
        for nn in graph.neighbors(tmp):
            if nn not in path:
                tmp = nn
                break

    path.append(tmp)
    # get path length
    for idx in range(1, len(path)):
        u = path[idx - 1]
        v = path[idx]
        distance += math.sqrt(
            (graph.nodes[u]['x'] - graph.nodes[v]['x']) ** 2 +
            (graph.nodes[u]['y'] - graph.nodes[v]['y']) ** 2 +
            (graph.nodes[u]['z'] - graph.nodes[v]['z']) ** 2
        )
    if len(list(graph.neighbors(tmp))) > 2:
        path.remove(tmp)

    return distance, len(path), path


def prune_graph(graph, prune_iter, prune_length):
    for i in range(prune_iter):
        to_remove = {}
        # iterate through end nodes
        end_nodes = get_n_degree_nodes(graph, 1)
        print("end nodes: ", len(end_nodes))
        branching_nodes = get_ge_n_degree_nodes(graph, 3)
        print("branching nodes: ", len(branching_nodes))
        for end_node in end_nodes:
            if not graph.has_node(end_node):
                continue
            length, num_segments, path = get_branch_length(graph, end_node)
            if length <= prune_length:
                to_remove[end_node] = path

        # remove short branches
        for end_node in to_remove.keys():
            graph.remove_nodes_from(to_remove[end_node])

    return graph


def main():
    args = get_arguments()
    print(args)

    swcs = glob(args.in_folder + "/*.swc")
    if args.out_folder is None:
        out_folder = os.path.join(
            args.in_folder,
            "iter_%i_len_%i" % (
                args.prune_iter, args.prune_length)
        )
    else:
        out_folder = args.out_folder
    os.makedirs(out_folder, exist_ok=True)

    for swc in swcs:
        skeleton = create_graph_from_swc(swc, args.scale_factor)
        skeleton = prune_graph(
            skeleton,
            args.prune_iter,
            args.prune_length
        )
        outfn = os.path.join(
            out_folder,
            os.path.basename(swc)
        )
        write_swc(skeleton, outfn)


if __name__ == "__main__":
    main()
