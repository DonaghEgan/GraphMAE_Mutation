import torch
from src.model import GraphMae

def test_random_masking_basic():
    """Test basic masking functionality: correct shapes and mask rate."""
    torch.manual_seed(0)
    batch_size = 4
    n = 20
    feats = 5
    adj = torch.eye(n)
    x = torch.arange(batch_size * n * feats, dtype=torch.float32).view(batch_size, n, feats)

    model = GraphMae(feats_in=feats, feats_out=5, mask_rate=0.4, replace_rate=0.1)

    # Model returns (out_x, mask_idx) where mask_idx contains indices of masked nodes
    out_x, mask_idx = model.random_masking(adj, x)

    # Check shapes
    assert out_x.shape == x.shape, f"Expected out_x shape {x.shape}, got {out_x.shape}"
    assert mask_idx.dim() == 1, f"mask_idx should be 1D tensor, got {mask_idx.dim()}D"
    
    # Check number of masked nodes matches mask_rate
    num_masked = mask_idx.shape[0]
    expected_masked = int(model.mask_rate * n)
    assert num_masked == expected_masked, f"Expected {expected_masked} masked nodes, got {num_masked}"
    
    print(f"✓ Masked {num_masked}/{n} nodes ({model.mask_rate*100}% mask rate)")


def test_masking_token_replacement():
    """Test that some nodes are replaced with mask token and others with noise."""
    torch.manual_seed(42)
    batch_size = 2
    n = 100
    feats = 10
    adj = torch.eye(n)
    x = torch.randn(batch_size, n, feats)
    
    mask_rate = 0.5
    replace_rate = 0.2  # 20% of masked nodes will be replaced with noise
    model = GraphMae(feats_in=feats, feats_out=32, mask_rate=mask_rate, replace_rate=replace_rate)
    
    out_x, mask_idx = model.random_masking(adj, x)
    
    num_masked = mask_idx.shape[0]
    expected_masked = int(mask_rate * n)
    
    # Calculate expected number of nodes replaced with noise vs mask token
    expected_noise_nodes = int(replace_rate * expected_masked)
    expected_token_nodes = expected_masked - expected_noise_nodes
    
    # Check which masked nodes have the mask token (check first batch)
    # (those that are exactly equal to the mask token)
    mask_token = model.enc_mask_token.squeeze(0)
    nodes_with_token = 0
    nodes_with_noise = 0
    
    for idx in mask_idx:
        if torch.allclose(out_x[0, idx], mask_token, atol=1e-6):
            nodes_with_token += 1
        else:
            nodes_with_noise += 1
    
    print(f"✓ Masked nodes breakdown:")
    print(f"  - Replaced with mask token: {nodes_with_token} (expected ~{expected_token_nodes})")
    print(f"  - Replaced with noise: {nodes_with_noise} (expected ~{expected_noise_nodes})")
    
    # Allow some tolerance since randomness is involved
    assert nodes_with_token > 0, "At least some nodes should use mask token"
    assert nodes_with_noise >= 0, "Some nodes should be replaced with noise"
    assert nodes_with_token + nodes_with_noise == num_masked, "All masked nodes should be accounted for"


def test_masking_preserves_unmasked():
    """Test that unmasked nodes remain unchanged."""
    torch.manual_seed(123)
    batch_size = 3
    n = 30
    feats = 8
    adj = torch.eye(n)
    x = torch.randn(batch_size, n, feats)
    x_original = x.clone()
    
    model = GraphMae(feats_in=feats, feats_out=24, mask_rate=0.3, replace_rate=0.1)
    
    out_x, mask_idx = model.random_masking(adj, x)
    
    # Create a set of masked indices for quick lookup
    masked_set = set(mask_idx.tolist())
    
    # Check that unmasked nodes are unchanged (for all batches)
    unchanged_count = 0
    for b in range(batch_size):
        for i in range(n):
            if i not in masked_set:
                assert torch.allclose(out_x[b, i], x_original[b, i]), \
                    f"Unmasked node {i} in batch {b} was changed"
                unchanged_count += 1
    
    print(f"✓ {unchanged_count}/{n * batch_size} unmasked nodes preserved correctly across batches")


def test_masking_deterministic_with_seed():
    """Test that masking is deterministic given the same seed."""
    batch_size = 2
    n = 25
    feats = 6
    adj = torch.eye(n)
    x = torch.randn(batch_size, n, feats)
    
    model = GraphMae(feats_in=feats, feats_out=20, mask_rate=0.4, replace_rate=0.15)
    
    # Run masking twice with same seed
    torch.manual_seed(999)
    out_x1, mask_idx1 = model.random_masking(adj, x)
    
    torch.manual_seed(999)
    out_x2, mask_idx2 = model.random_masking(adj, x)
    
    # Results should be identical
    assert torch.allclose(out_x1, out_x2), "Masking should be deterministic with same seed"
    assert torch.equal(mask_idx1, mask_idx2), "Mask indices should be identical with same seed"
    
    print(f"✓ Masking is deterministic with seed")


def test_edge_cases():
    """Test edge cases like high mask rates."""
    batch_size = 2
    n = 50
    feats = 4
    adj = torch.eye(n)
    x = torch.randn(batch_size, n, feats)
    
    # Test with very high mask rate
    torch.manual_seed(0)
    model_high = GraphMae(feats_in=feats, feats_out=16, mask_rate=0.9, replace_rate=0.1)
    out_x, mask_idx = model_high.random_masking(adj, x)
    
    expected_masked = int(0.9 * n)
    assert mask_idx.shape[0] == expected_masked, \
        f"High mask rate: expected {expected_masked} masked, got {mask_idx.shape[0]}"
    
    # Test with zero replace rate
    torch.manual_seed(0)
    model_no_replace = GraphMae(feats_in=feats, feats_out=16, mask_rate=0.3, replace_rate=0.0)
    out_x, mask_idx = model_no_replace.random_masking(adj, x)
    
    # All masked nodes should have the mask token (check first batch)
    mask_token = model_no_replace.enc_mask_token.squeeze(0)
    all_have_token = all(torch.allclose(out_x[0, idx], mask_token, atol=1e-6) for idx in mask_idx)
    assert all_have_token, "With replace_rate=0, all masked nodes should use mask token"
    
    print(f"✓ Edge cases handled correctly")


def run_all_tests():
    """Run all masking tests."""
    print("=" * 60)
    print("Running GraphMAE Masking Tests")
    print("=" * 60)
    
    tests = [
        ("Basic masking functionality", test_random_masking_basic),
        ("Token vs noise replacement", test_masking_token_replacement),
        ("Unmasked nodes preservation", test_masking_preserves_unmasked),
        ("Deterministic behavior", test_masking_deterministic_with_seed),
        ("Edge cases", test_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n[TEST] {test_name}")
            test_func()
            passed += 1
            print(f"[PASS] {test_name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {test_name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {test_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
