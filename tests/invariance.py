import torch
from torch import nn
from einops import repeat
from invariant_point_attention import InvariantPointAttention, IPABlock
from invariant_point_attention.utils import rot, frame_aligned_point_error

def test_ipa_invariance():
    attn = InvariantPointAttention(
        dim = 64,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4
    )

    seq           = torch.randn(1, 256, 64)
    pairwise_repr = torch.randn(1, 256, 256, 64)
    mask          = torch.ones(1, 256).bool()

    rotations     = repeat(rot(*torch.randn(3)), 'r1 r2 -> b n r1 r2', b = 1, n = 256)
    translations  = torch.randn(1, 256, 3)

    # random rotation, for testing invariance

    random_rotation = rot(*torch.randn(3))

    # get outputs of IPA

    attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    rotated_attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations @ random_rotation,
        translations = translations @ random_rotation,
        mask = mask
    )

    # output must be invariant

    diff = (attn_out - rotated_attn_out).max()
    assert diff <= 1e-6, 'must be invariant to global rotation'

def test_ipa_block_invariance():
    attn = IPABlock(
        dim = 64,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4
    )

    seq           = torch.randn(1, 256, 64)
    pairwise_repr = torch.randn(1, 256, 256, 64)
    mask          = torch.ones(1, 256).bool()

    rotations     = repeat(rot(*torch.randn(3)), 'r1 r2 -> b n r1 r2', b = 1, n = 256)
    translations  = torch.randn(1, 256, 3)

    # random rotation, for testing invariance

    random_rotation = rot(*torch.randn(3))

    # get outputs of IPA

    attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    rotated_attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations @ random_rotation,
        translations = translations @ random_rotation,
        mask = mask
    )

    # output must be invariant

    diff = (attn_out - rotated_attn_out).max()
    assert diff <= 1e-6, 'must be invariant to global rotation'

def test_fape():
    b, n, d = 1, 16, 3
    pred_rots = repeat(rot(*torch.randn(d)), 'r1 r2 -> b n r1 r2', b=b, n=n)
    pred_trans = torch.randn(b, n, d)
    target_rots = repeat(rot(*torch.randn(d)), 'r1 r2 -> b n r1 r2', b=b, n=n)
    target_trans = torch.randn(b, n, d)
    pred_positions = torch.randn(b, n, d)
    target_positions = torch.randn(b, n, d)
    pred_frames = (pred_rots, pred_trans)
    target_frames = (target_rots, target_trans)
    frames_mask, positions_mask = None, None  # not implemented yet, not as placeholders
    # loss = frame_aligned_point_error(
    #     pred_frames, target_frames, frames_mask,
    #     pred_positions, target_positions, positions_mask)
    loss = frame_aligned_point_error(
        pred_frames, pred_frames, frames_mask,
        pred_positions, pred_positions, positions_mask)

    assert loss <= 0.0001, 'identical transformation shall generate zero loss...'

if __name__ == '__main__':
    test_fape()
