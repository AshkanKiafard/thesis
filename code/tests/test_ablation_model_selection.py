from pathlib import Path

import core.utils as utils


def test_ablation_model_names_include_reference_and_three_variants():
    assert utils.get_ablation_model_names("v3") == [
        "granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v3_finetuned",
        "granite-embedding-english-r2_relu_cosine_nonorm_matryoshka_v3_ablation_finetuned",
        "granite-embedding-english-r2_gelu_euclid_nonorm_matryoshka_v3_ablation_finetuned",
        "granite-embedding-english-r2_gelu_cosine_nonorm_matryoshka_v3_ablation_finetuned",
    ]


def test_ablation_model_paths_keep_the_comparison_order(tmp_path, monkeypatch):
    expected_names = utils.get_ablation_model_names("v3")
    for model_name in reversed(expected_names):
        (tmp_path / model_name).mkdir()

    monkeypatch.setattr(utils, "LIGHTNING_MODELS_DIR", Path(tmp_path))

    model_paths = utils.get_ablation_fine_tuned_models("v3")

    assert [Path(model_path).name for model_path in model_paths] == expected_names
