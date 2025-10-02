import torch
from src.model import GraphMae


def test_random_masking_basic():
    torch.manual_seed(0)
    n = 20
    feats = 5
    adj = torch.eye(n)
    x = torch.arange(n * feats, dtype=torch.float32).view(n, feats)

    model = GraphMae(feats_in=feats, mask_rate=0.4, replace_rate=0.25)

    out_x, mask, replace_idx = model.random_masking(adj, x)

    # shapes
    assert out_x.shape == x.shape
    assert mask.shape[0] == n
    assert replace_idx.shape[0] == n

    num_masked = mask.sum().item()
    # mask_rate * n should be number of masked nodes
    assert num_masked == int(0.4 * n)

    # number of replaced (noisy) nodes should equal replace_rate * num_masked
    expected_replaced = int(model.replace_rate * num_masked)
    assert replace_idx.sum().item() == expected_replaced

    # verify that masked-with-token rows equal the enc_mask_token
    mask_token_idx = (mask & ~replace_idx).nonzero(as_tuple=True)[0]
    if mask_token_idx.numel() > 0:
        for i in mask_token_idx.tolist():
            assert torch.allclose(out_x[i], model.enc_mask_token.squeeze(0))


if __name__ == '__main__':
    test_random_masking_basic()
    print('ok')
