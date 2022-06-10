from RNAdist.Attention.prediction import model_predict
from RNAdist.Attention.tests.data_fixtures import (
    saved_model,
    random_fasta
)
import os


def test_model_predict(saved_model, random_fasta, tmp_path):

    outfile = os.path.join(tmp_path, "predictions")
    model_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=1,
        num_threads=os.cpu_count(),
        max_length=20
    )
    assert os.path.exists(outfile)
