import vigra
import numpy as np

from EdgeRF import RandomForest

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.cgp as ncgp
    import nifty.graph.rag as nrag
    import nifty.segmentation as nseg
except ImportError:
    try:
        import nifty_with_cplex.cgp as ncgp
        import nifty_with_cplex.graph.rag as nrag
        import nifty_with_cplex.segmentation as nseg
    except ImportError:
        try:
            import nifty_with_gurobi.cgp as ncgp
            import nifty_with_gurobi.graph.rag as nrag
            import nifty_with_gurobi.segmentation as nseg
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# accumulate the higher order features
def higher_order_feature_accumulator(
    ds,
    higher_order_feat_list
):
    pass


# map edge features to junctions
def junction_feats_from_edge_feats(ds, feature_list):
    pass


# TODO
# dedicated features for junctions
# TODO


# junction labels from groundtruth
def junction_groundtruth(ds):
    pass


# learn random forests from higher order feats
def learn_higher_order_rf(ds):
    pass


# TODO learn and predict
