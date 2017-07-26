from higher_order_feats import learn_and_predict_higher_order_rf
from higher_order_feats import junction_predictions_to_costs, project_junction_probs_to_edges
from EdgeRF import learn_and_predict_rf_from_gt
from MCSolverImpl import probs_to_energies
from MCSolver import _get_feat_str
from ExperimentSettings import ExperimentSettings


# complete higher order workflow
# TODO -> need higher order mc
def higher_order_workflow(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    edge_feat_list,
    higher_order_weight=1.,
    with_defects=False
):
    edge_costs, junction_costs = higher_order_problem(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        edge_feat_list,
        higher_order_weight,
        with_defects
    )
    # TODO solve higher order multicut


def higher_order_problem(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    edge_feat_list,
    higher_order_weight=1.,
    with_defects=False
):
    junction_probs = learn_and_predict_higher_order_rf(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        edge_feat_list,
        with_defects=with_defects
    )

    edge_probs = learn_and_predict_rf_from_gt(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        edge_feat_list,
        with_defects=with_defects,
        use_2rfs=ExperimentSettings().use_2rfs
    )

    # TODO beta junctions
    junction_costs = junction_predictions_to_costs(junction_probs)
    junction_costs *= higher_order_weight

    # TODO proper weighting
    edge_costs = probs_to_energies(
        ds_test,
        edge_probs,
        seg_id_train,
        ExperimentSettings().weighting_scheme,
        ExperimentSettings().weight,
        ExperimentSettings().beta_local,
        _get_feat_str(edge_feat_list)
    )

    return edge_costs, junction_costs
