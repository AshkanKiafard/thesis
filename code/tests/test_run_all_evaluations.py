from evaluation import run_all_evaluations


def test_localize_model_path_discards_remote_container_prefix():
    remote_path = (
        "/app/code/data/models/lightning/"
        "granite-embedding-english-r2_relu_euclid_nonorm_"
        "matryoshka_v4_finetuned"
    )

    localized = run_all_evaluations.localize_model_path(remote_path)

    assert localized == str(
        run_all_evaluations.LIGHTNING_MODELS_DIR
        / "granite-embedding-english-r2_relu_euclid_nonorm_"
        "matryoshka_v4_finetuned"
    )
