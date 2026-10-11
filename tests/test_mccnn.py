from __future__ import annotations

import pytest
import torch

from dwt_textcnn.models.mccnn import MCCNN
from dwt_textcnn.training import load_checkpoint


@pytest.mark.parametrize("dimension", [150, 300])
def test_mccnn_accepts_compressed_and_original_dimensions(dimension: int) -> None:
    model = MCCNN(
        torch.randn(20, dimension),
        kernel_sizes=(5, 6, 7),
        filters_per_size=4,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 20, (3, 9))

    logits = model(input_ids)
    logits.sum().backward()

    assert logits.shape == (3, 2)
    assert model.pretrained_embedding.weight.grad is not None
    assert model.random_embedding.weight.grad is not None
    assert all(convolution.weight.grad is not None for convolution in model.convolutions)


def test_shared_filter_is_applied_then_relu_is_summed_per_channel() -> None:
    pretrained = torch.tensor([[0.0], [1.0], [-2.0]])
    model = MCCNN(
        pretrained,
        kernel_sizes=(1,),
        filters_per_size=1,
        dropout=0.0,
        initialization_std=0.1,
    )
    with torch.no_grad():
        model.random_embedding.weight.copy_(torch.tensor([[0.0], [-3.0], [4.0]]))
        model.convolutions[0].weight.fill_(1.0)
        model.convolutions[0].bias.zero_()

    features = model.extract_features(torch.tensor([[1, 2]]))

    # token 1: relu(1) + relu(-3) = 1; token 2: relu(-2) + relu(4) = 4.
    assert features.item() == pytest.approx(4.0)


def test_padding_rows_can_be_repaired_after_an_optimizer_step() -> None:
    model = MCCNN(torch.randn(10, 8), kernel_sizes=(2,), filters_per_size=2)
    with torch.no_grad():
        model.pretrained_embedding.weight[0].fill_(1.0)
        model.random_embedding.weight[0].fill_(1.0)

    model.enforce_padding_zero()

    assert torch.count_nonzero(model.pretrained_embedding.weight[0]).item() == 0
    assert torch.count_nonzero(model.random_embedding.weight[0]).item() == 0


def test_sequence_shorter_than_widest_kernel_is_rejected() -> None:
    model = MCCNN(torch.randn(10, 8), kernel_sizes=(5, 7), filters_per_size=2)

    with pytest.raises(ValueError, match="at least 7"):
        model(torch.ones((2, 6), dtype=torch.long))


def test_extra_right_padding_does_not_change_features() -> None:
    model = MCCNN(
        torch.randn(10, 8), kernel_sizes=(2, 3), filters_per_size=2, dropout=0.0
    )
    model.eval()

    minimally_padded = torch.tensor([[2, 3, 4, 0, 0]])
    extra_padded = torch.tensor([[2, 3, 4, 0, 0, 0, 0]])

    assert torch.allclose(
        model.extract_features(minimally_padded),
        model.extract_features(extra_padded),
    )


def test_checkpoint_reload_preserves_predictions(tmp_path) -> None:
    weights = torch.randn(10, 8)
    model = MCCNN(weights, kernel_sizes=(2, 3), filters_per_size=2, dropout=0.0)
    model.eval()
    input_ids = torch.tensor([[2, 3, 4, 0]])
    expected = model(input_ids).detach()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state": model.state_dict(),
            "seed": 13,
            "epoch": 1,
            "dev_accuracy": 0.5,
            "method": "original",
        },
        checkpoint_path,
    )
    restored = MCCNN(
        weights, kernel_sizes=(2, 3), filters_per_size=2, dropout=0.0
    )
    load_checkpoint(checkpoint_path, restored, torch.device("cpu"))
    restored.eval()

    assert torch.equal(expected, restored(input_ids))
