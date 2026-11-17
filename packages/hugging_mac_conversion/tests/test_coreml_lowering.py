import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("width", [8, 64])
def test_unit_triangular_solve_on_coreml_cpu(tmp_path, width):
    import numpy as np
    import torch
    from hugging_mac_conversion.shared.contracts import PreparedGraph
    from hugging_mac_conversion.shared.coreml import export, predict

    class Solve(torch.nn.Module):
        def forward(self, matrix, rhs):
            return torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)

    generator = torch.Generator().manual_seed(17)
    matrix = torch.tril(torch.randn(2, width, width, generator=generator) * 0.1, -1)
    matrix += torch.eye(width)
    rhs = torch.randn(2, width, 3, generator=generator)
    graph = PreparedGraph(Solve().eval(), (matrix, rhs), ("matrix", "rhs"), "solution")
    artifact = export(graph, tmp_path)
    actual = predict(artifact, {"matrix": matrix.numpy(), "rhs": rhs.numpy()})["solution"]
    np.testing.assert_allclose(actual, graph.module(matrix, rhs).numpy(), atol=1e-5, rtol=1e-5)
