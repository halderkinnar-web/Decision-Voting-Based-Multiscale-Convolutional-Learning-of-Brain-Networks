import numpy as np
from brainnet_dv.config import Config
from brainnet_dv.preprocessing import graph_diffusion, sparsify_fc, sparsify_sc
from brainnet_dv.graphs import make_graph

def test_preprocessing_shapes_and_diagonal():
    x = np.array([[1., .5, -.2], [.5, 1., .3], [-.2, .3, 1.]])
    y = graph_diffusion(x)
    assert y.shape == x.shape
    assert np.allclose(np.diag(y), 0)
    assert np.allclose(sparsify_fc(y, 1), sparsify_fc(y, 1).T)
    assert np.allclose(np.diag(sparsify_sc(np.abs(x))), 0)

def test_make_graph():
    fc = np.eye(3, dtype=float)
    sc = np.array([[0., 1., 0.], [1., 0., 2.], [0., 2., 0.]])
    g = make_graph(fc, sc, 1)
    assert g.x.shape == (3, 3)
    assert g.edge_index.shape[0] == 2
    assert int(g.y.item()) == 1
