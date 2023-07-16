from math import log
import pandas as pd
import numpy as np
import scanpy as sc
import sklearn.metrics
import sklearn.neighbors
from anndata import AnnData
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
# from sklearn.metrics import silhouette_score


from typing import Optional, Union
RandomState = Optional[Union[np.random.RandomState, int]]


def shannon_entropy(x, b_vec, N_b):
    tabled_values = b_vec[x > 0].value_counts() / len(b_vec[x > 0])  # class 'pandas.core.series.Series'

    tabled_val = tabled_values.tolist()

    entropy = 0.0
    for element in tabled_val:
        if element != 0:
            entropy += element * log(element)

    entropy /= log(N_b)

    return (-entropy)  # the entropy formula is the -sum, this is why we include the minus sign

def compute_entropy(adata, output_entropy=None, batch_key='batch', celltype_key='celltype'):
    print("Calculating entropy ...")
    kwargs = {}
    # batch vector(batch id of each cell)
    kwargs['batch_vector'] = adata.obs[batch_key]
    # modify index of batch vector so it coincides with matrix's index
    kwargs['batch_vector'].index = range(0, len(kwargs['batch_vector']))
    # number of batches
    kwargs['N_batches'] = len(adata.obs[batch_key].astype('category').cat.categories)

    # cell_type vector( betch id of each cell)
    kwargs['cell_type_vector'] = adata.obs[celltype_key]
    # modify index of cell_type vector so it coincides with matrix's index
    kwargs['cell_type_vector'].index = range(0, len(kwargs['cell_type_vector']))
    # number of cell_types
    kwargs['N_cell_types'] = len(adata.obs[celltype_key].astype('category').cat.categories)

    try:
        knn_graph = adata.uns['neighbors']
        print('use exist neighbors')
    except KeyError:
        # compute neighbors
        print('compute neighbors')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)

    # knn graph
    knn_graph = adata.uns['neighbors']['connectivities']
    # transforming csr_matrix to dataframe
    df = pd.DataFrame(knn_graph.toarray())

    # apply function
    batch_entropy = df.apply(shannon_entropy, axis=0, args=(kwargs['batch_vector'], kwargs['N_batches']))
    cell_type_entropy = df.apply(shannon_entropy, axis=0, args=(kwargs['cell_type_vector'], kwargs['N_cell_types']))
    print("Entropy calculated!")

    min_val = -1
    max_val = 1
    batch_entropy_norm = (batch_entropy - min_val) / (max_val - min_val)
    cell_type_entropy_norm = (cell_type_entropy - min_val) / (max_val - min_val)

    entropy_fscore = []
    fscoreARI = (2 * (1 - batch_entropy_norm) * (cell_type_entropy_norm)) / (
                1 - batch_entropy_norm + cell_type_entropy_norm)
    entropy_fscore.append(fscoreARI)

    results = pd.DataFrame(
        {'batch': batch_entropy, "cell_type": cell_type_entropy, "fscore": (pd.DataFrame(entropy_fscore).T)[0]})

    return results

def normalized_mutual_info(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering
    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`
    Returns
    -------
    nmi
        Normalized mutual information
    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`_
    """
    x = AnnData(X=x)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res)
        leiden = x.obs["leiden"]
        nmi_list.append(sklearn.metrics.normalized_mutual_info_score(
            y, leiden, **kwargs
        ).item())
    return max(nmi_list)

def avg_silhouette_width(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Cell type average silhouette width
    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`
    Returns
    -------
    asw
        Cell type average silhouette width
    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`_
    """
    return (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2

def avg_silhouette_width_batch(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width
    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`
    Returns
    -------
    asw_batch
        Batch average silhouette width
    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`_
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()


def integrate_indicators (data, tech, celltype, model_name='integrate', verbose=1):

    measure_dict = dict()
    idx = np.random.choice(len(data), size=int(len(data) * 0.8), replace=False)
    measure_dict['ASW_c'] = avg_silhouette_width(data[idx, :], celltype[idx])
    measure_dict['ASW_b'] = avg_silhouette_width_batch(data[idx, :], tech[idx], celltype[idx])

    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict


def cluster_indicators (pred, labels=None, model_name='cluster', verbose=1):

    measure_dict = dict()
    if labels is not None:
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['ari'] = ARI(labels, pred)

#如果需要打印所有指标
    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict