import pytest
from RNAdist.nn.DISTAtteNCionE import TriangularSelfAttention, TriangularUpdate
import torch


@pytest.mark.parametrize(
    "mode",
    ["in", "out"]
)
def test_triangular_update(mode, masked_pair_rep_batch):
    pair_rep, mask, target = masked_pair_rep_batch
    module = TriangularUpdate(2, c=5, mode=mode)
    module.train()
    inv_mask = (~(mask.bool()))[..., None]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    for x in range(3):
        pred = module(pair_rep, mask=mask)
        crit = torch.nn.MSELoss()
        loss = crit(pred, target)
        loss.backward()
        optimizer.step()
        assert torch.sum(inv_mask * pred) == 0

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

@pytest.mark.parametrize(
    "mode",
    ["in", "out"]
)
def test_triangular_attention(mode, masked_pair_rep_batch):
    pair_rep, mask, target = masked_pair_rep_batch
    module = TriangularSelfAttention(2, c=1, heads=1)
    module.train()
    inv_mask = (~(mask.bool()))[..., None]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    for x in range(3):
        pred = module(pair_rep, mask=mask)
        crit = torch.nn.MSELoss()
        loss = crit(pred, target)
        loss.backward()
        optimizer.step()
        assert torch.sum(inv_mask * pred) == 0
