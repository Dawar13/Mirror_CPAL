# A09 exactness audit: mirror_exact_s0

Model deit3_base_patch16_224.fb_in1k; 1000 validation images; forms dense, even_odd; commit 7d05b6439c. Levels: E1 exact in real arithmetic (a theorem, checked in float64 on weights re-projected in float64), E2 numerically exact (error no larger than the model's own rounding), E3 bitwise identical.

- **E1 (A09-1): PASS.** Largest relative equivariance error in float64 3.35e-13 at even_odd block 1 attn_probs (bar 1e-10).
- **E2 (A09-2): FAIL.** Float32 error within 2x the model's own float32-versus-float64 deviation (exact floor) at 375 of 376 tensors (other floor: 2 failing); worst ratio 2.0 at dense embed tokens0_prefix; largest float32 error 1.47e-04 at dense block 12 act; float32 flip consistency passes by form: {'dense': None, 'even_odd': None}.
- **bfloat16 (A09-3, report):** dense/bf16_autocast: largest error 3.95e-01, flip consistency None%; even_odd/bf16_autocast: largest error 5.13e-02, flip consistency None%.
- **Rounding floor of the original model:** largest float32-versus-float64 deviation 1.29e-03 at block 12 act.
- **Weight conditions:** largest relative residual 3.15e-08 (float32 storage).

## Bitwise check of the even/odd form (E3)

- first stage that differs: patch_embed
- embedding parameters are symmetric only to rounding in the stored even/odd basis (patch_embed.weight: 54.51% of entries identical, patch_embed.bias(odd part zero): 0.00% of entries identical, pos_embed.grid: 52.82% of entries identical, cls_token(odd part zero): 0.00% of entries identical): the projection is rounded in the standard basis before the rotation into the even/odd basis
- inside a block given an exactly mirrored input (lowest share over blocks): norm 100.00%, queries 100.00%, MLP 100.00% of entries identical; attention output 10.86% with keys in the mirrored order and 100.00% with keys reordered to the original order
- the sign flip alone (same token positions) is exact in the MLP and the attention: 100% identical
- reordering the keys to the original order raises the attention share: the order in which attention sums over keys (softmax normaliser and value sum) is a source of difference
- with exactly symmetric embedding parameters the first difference is patch_embed (patch embedding 13.17% identical: the convolution sums the flipped kernel window in another order)
- predictions identical for 100.00% of images; features identical for 0.00%

Files: components.csv (one row per form, precision and component), weights.csv, flip.json, bitwise.json, metrics.json, meta.json.
