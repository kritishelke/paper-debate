from training.prepare_po_training_data import is_bad, is_good, quality_score


def test_quality_score_weighting() -> None:
    row = {
        "debate_quality": 5,
        "argument_novelty": 4,
        "claim_coverage": 3,
        "consensus_quality": 1,
        "sycophancy_pct": 0,
    }
    assert round(quality_score(row), 2) == 3.9
    assert not is_good(row)
    row["debate_quality"] = 2
    assert is_bad(row)
