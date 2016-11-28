import numpy as np
import scipy.sparse as sparse

# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py

def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]
    n = segA.size  # number of nonzero pixels in original segA

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64)

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    B_nonzero = p_ij[:, 1:]
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = B_zero.sum()

    # This is the old code, with conversion to probabilities:
    #
    #  # sum of the joint distribution
    #  #   separate sum of B>0 and B=0 parts
    #  sum_p_ij = ((B_nonzero.astype(np.float32) / n).power(2).sum() +
    #              (float(num_B_zero) / (n ** 2)))
    #
    #  # these are marginal probabilities
    #  a_i = p_ij.sum(1).astype(np.float32) / n
    #  b_i = B_nonzero.sum(0).astype(np.float32) / n
    #
    #  sum_a = np.power(a_i, 2).sum()
    #  sum_b = np.power(b_i, 2).sum() + (float(num_B_zero) / (n ** 2))

    # This is the new code, removing the divides by n because they cancel.

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero


    # these are marginal probabilities
    a_i = p_ij.sum(1)
    b_i = B_nonzero.sum(0)

    sum_a = np.power(a_i, 2).sum()
    sum_b = np.power(b_i, 2).sum() + num_B_zero

    precision = float(sum_p_ij) / sum_b
    recall = float(sum_p_ij) / sum_a

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are

def adapted_VI(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted VI error;
    prec : float, optional
        The adapted VI precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted VI recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]
    n = segA.size  # number of nonzero pixels in original segA

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64).astype(np.float64)

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    B_nonzero = p_ij[:, 1:]
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = float(B_zero.sum())

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    eps = 1e-15
    plogp_ij = (B_nonzero / n) * (np.log(B_nonzero + eps) - np.log(n))
    sum_plogp_ij = plogp_ij.sum() - (num_B_zero / n) * np.log(n)

    # these are marginal probabilities
    a_i = p_ij.sum(1)
    b_i = B_nonzero.sum(0)
    sum_aloga_i = ((a_i / n) * (np.log(a_i + eps) - np.log(n))).sum()
    #   separate sum of B>0 and B=0 parts
    sum_blogb_i = ((b_i / n) * (np.log(b_i + eps) - np.log(n))).sum() - (num_B_zero / n) * np.log(n)

    precision = (sum_plogp_ij - sum_aloga_i - sum_blogb_i) / sum_blogb_i
    recall = (sum_plogp_ij - sum_aloga_i - sum_blogb_i) / sum_aloga_i

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are
