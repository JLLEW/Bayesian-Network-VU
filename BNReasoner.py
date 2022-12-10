from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
from networkx.utils import UnionFind
import pandas as pd
import networkx as nx


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # METHODS FOR ASSIGNMENT -------------------------------------------------------------------------------------

    # network pruning
    def prune(self, queries: list, evidence: dict) -> BayesNet:

        output_net = deepcopy(self.bn)

        # edge prunning
        for var in evidence.keys():
            # update conditional probability table based on evidence
            cpt = output_net.get_cpt(var)
            output_net.update_cpt(var, cpt[cpt[var] == evidence[var]])

            children = output_net.get_children(var)
            for child in children:
                # remove the edge
                output_net.del_edge((var, child))
                # update the conditional probability table
                cpt = output_net.get_cpt(child)
                output_net.update_cpt(child, cpt[cpt[var] == evidence[var]])

        # leaf nodes prunning
        while True:
            leaf_vars = [var for var in output_net.get_all_variables() if not output_net.get_children(var)]
            
            # if there are no more nodes to prune
            if not leaf_vars - set(queries + list(evidence.keys())):
                break
            
            for leaf in leaf_vars:
                if leaf not in queries and leaf not in evidence:
                    output_net.del_var(leaf)
        
        return output_net

    # d-separation
    def are_d_seperated(self, X: list, Y: list, Z: list) -> bool:
        # make sure that X, Y and Z are sets and conists of unique variables
        X = set(X)
        Y = set(Y)
        Z = set(Z)

        graph = deepcopy(self.bn)
        XYZ = X.union(Y).union(Z)

        # remove leaf nodes that are not in union of X, Y, Z
        while True:
            leaf_vars = [var for var in graph.get_all_variables() if not graph.get_children(var)]
            # if there are no more nodes to prune
            if not set(leaf_vars) - XYZ:
                break
            
            for leaf in leaf_vars:
                if leaf not in XYZ:
                    graph.del_var(leaf)
        
        # remove outgoing edges for each node from Z
        edges = graph.structure.out_edges(Z)
        for edge in edges:
            graph.del_edge(edge)

        # check if X and Y are disconnected in the graph
        # create empty UnionFind instance
        disjoint_set = UnionFind(graph.structure.nodes())
        # get weakly connected components
        for wcc in nx.weakly_connected_components(graph.structure):
            disjoint_set.union(*wcc)

        disjoint_set.union(*X)
        disjoint_set.union(*Y)

        # early condition if any of sets is empty
        if not X or not Y:
            return False

        # to check if X is not d-seperated from Y, we just have to check if any of vars from X is in Y
        for x in X:
            for y in Y:
                if disjoint_set[y] == disjoint_set[x]:
                    return False

        # X and Y are disconnected in pruned graph
        return True

    def are_independent(self, X: list, Y: list, Z: list) -> bool:
        # d-separation implies an independence
        if self.are_d_seperated(X, Y, Z):
            return True

        # if there is a direct edge then X and Y are not independent
        edges = self.bn.structure.edges()
        for x in X:
            for y in Y:
                if (x, y) in edges or (y,x) in edges:
                    return False
        
        return True

    # marginalize by summing-out
    def marginalize(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:
        vars = [col_name for col_name in factor.columns if col_name not in [var, "p", " "]]
        marginalized = factor.groupby(vars).sum().reset_index()
        if 'p' in marginalized.columns:
            marginalized = marginalized.drop(var, axis=1)
            marginalized = marginalized.rename(columns={"p" : f"\u03A3 {var}"})

        return marginalized
