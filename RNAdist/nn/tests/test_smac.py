from RNAdist.nn.smac_optimize import smac_that
import os

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

def test_smac_hpo(random_fasta, expected_labels, tmpdir):
    model = os.path.join(tmpdir, "smac_model.pt")
    smac_dir = os.path.join(tmpdir, "smac_dir")
    dataset_path = os.path.join(tmpdir, "smac_dataset")
    smac_that(
        fasta=random_fasta,
        model_output=model,
        smac_dir=smac_dir,
        label_dir=expected_labels,
        dataset_path=dataset_path,
        max_length=20,
        train_val_ratio=0.8,
        device="cpu",
        max_epochs=1,
        num_threads=1,
        run_default=True,
        ta_run_limit=10
    )
    assert os.path.exists(model)
    assert os.path.exists(smac_dir)
    assert os.path.exists(dataset_path)