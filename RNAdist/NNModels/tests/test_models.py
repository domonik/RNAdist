import pytest
from RNAdist.NNModels.DISTAtteNCionE import TriangularSelfAttention, TriangularUpdate, GraphRNADISTAtteNCionE
import torch
from RNAdist.NNModels.Datasets import RNAGeometricWindowDataset
from torch_geometric.data import DataLoader as GeoDataloader
from tempfile import TemporaryDirectory


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

pytest_plugins = ["RNAdist.DPModels.tests.fixtures",
                  "RNAdist.NNModels.tests.data_fixtures"]

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


def test_graph_model(random_fasta, prefix):
    ml = 9

    with TemporaryDirectory(prefix=prefix) as tmpdir:
        dataset = RNAGeometricWindowDataset(
            data=random_fasta,
            label_dir=None,
            dataset_path=tmpdir,
            num_threads=1,
            max_length=ml,
            step_size=1
        )
        loader = GeoDataloader(
            dataset, batch_size=2, shuffle=False
        )
        model = GraphRNADISTAtteNCionE(
            8, 16, ml, upper_bound=dataset.upper_bound, graph_attention_layers=2
        )
        for batch in iter(loader):
            result = model(batch)
