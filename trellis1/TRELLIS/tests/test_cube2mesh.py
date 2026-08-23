import torch

from trellis.representations.mesh.cube2mesh import SparseFeatures2Mesh


def test_training_path_avoids_inplace_update_on_sdf_view():
    model = SparseFeatures2Mesh.__new__(SparseFeatures2Mesh)
    model.device = "cpu"
    model.res = 4
    model.sdf_bias = -1.0 / model.res
    model.use_color = False
    model.reg_v = torch.zeros((model.res + 1) ** 3, 3)
    model.reg_c = torch.zeros((model.res) ** 3, 8, dtype=torch.long)
    model.mesh_extractor = lambda **kwargs: (
        torch.empty((0, 3)),
        torch.empty((0, 3), dtype=torch.long),
        torch.zeros(0),
        None,
    )

    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int64)
    feats = torch.randn(2, 53, requires_grad=True)
    cubefeats = type("SparseTensorStub", (), {"coords": coords, "feats": feats})()

    mesh = model(cubefeats, training=True)

    assert mesh.success is False
    assert mesh.reg_loss is not None
    assert mesh.tsdf_s is not None
