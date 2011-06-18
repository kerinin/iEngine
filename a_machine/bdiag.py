from scipy.sparse import *
import numpy as np

def bdiag(blocks, format=None, dtype=None):
    """
    Build a block-diagonal sparse matrix from (sparse) blocks.

    Parameters
    ----------
    blocks : sequence of matrices
        The sequence of sparse matrices to put on the block diagonal.
    format : str
        The sparse format of the result (e.g. "csr").

    Returns
    -------
    mtx : coo_matrix
        The block-diagonal sparse matrix (COO format) composed from the
        given blocks.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix, bdiag
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = np.array([[5], [6]])
    >>> C = 7
    >>> bdiag([A, B, C]).todense()
    matrix([[1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 6, 0],
            [0, 0, 0, 7]])
    """
    if not len(blocks):
        raise ValueError('no matrix blocks!')

    row_sizes = []
    col_sizes = []
    blocks_2d = []

    for mtx in blocks:
        aux = coo_matrix(mtx)
        nr, nc = aux.shape

        row_sizes.append(nr)
        col_sizes.append(nc)
        blocks_2d.append(aux)

    row_offsets = np.cumsum(np.r_[0, row_sizes])
    col_offsets = np.cumsum(np.r_[0, col_sizes])

    rows = []
    cols = []
    datas = []
    for ii, mtx in enumerate(blocks_2d):
        rows.append(mtx.row + row_offsets[ii])
        cols.append(mtx.col + col_offsets[ii])
        datas.append(mtx.data)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    datas = np.concatenate(datas)

    if dtype is None:
        dtype = datas.dtype

    mtx = coo_matrix((datas, (rows, cols)),
                     shape=(row_offsets[-1], col_offsets[-1]),
                     dtype=dtype)

    return mtx.asformat(format)