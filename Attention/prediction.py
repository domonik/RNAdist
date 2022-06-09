import torch
from Attention.Datasets import RNAPairDataset
from Attention.DISTAtteNCionE import DISTAtteNCionE2
from torch.utils.data import DataLoader
import pickle
import numpy as np
from tempfile import TemporaryDirectory


def model_predict(fasta, saved_model, outfile, batch_size: int = 1, num_threads: int = 1, device: str = "cpu"):
    with TemporaryDirectory() as tmpdir:
        dataset = RNAPairDataset(
            data=fasta,
            label_dir=None,
            dataset_path=tmpdir,
            num_threads=num_threads,
            max_length=200
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )
        state_dict, config = torch.load(saved_model, map_location="cpu")
        model = DISTAtteNCionE2(17, config["nr_layers"])
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = {}
        for idx, element in enumerate(iter(data_loader)):
            with torch.no_grad():
                x, bppm, y, _ = element
                pred = model(bppm)
                pred = pred.numpy()
                description, _ = dataset.rna_graphs[idx]
                output[description] = pred
    with open(outfile, "wb") as handle:
        pickle.dump(output, handle)


def prediction_comparison(pckled_pred, pckled_sample):
    with open(pckled_pred, "rb") as handle:
        pred = pickle.load(handle)
    with open(pckled_sample, "rb") as handle:
        sample = pickle.load(handle)["undirected"]
    pred = pred[0:sample.shape[0], 0:sample.shape[0]]
    #sample = np.triu(sample)
    dif = np.abs(sample - pred)
    mae = np.sum(dif) / dif.size
    print(mae)


if __name__ == '__main__':
    pass




