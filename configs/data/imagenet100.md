# ImageNet-100 class list (CMC subset): source check

Checked on 2026-11-14 for PROJECT_NEXT_PHASE.md Section 4.1 ("the standard 100-class subset used by CMC, Tian et al. 2019; check the class list"). The project's copy is `data/imagenet100_classes.txt`, written by another agent; this note only compares it and does not change it.

## Verdict

All three independent sources and the project file hold the same 100 WordNet IDs in the same order. No differences.

## Sources compared

| # | Source | What it is | Format | WNIDs | Unique | sha256 of the fetched file |
|---|---|---|---|---|---|---|
| 1 | [HobbitLong/CMC `imagenet100.txt`](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/imagenet100.txt) at commit `7b227be0b10ef4e526c72af07664f5079ed9ee09` | The authors' repository | one WNID per line, LF, 1,000 bytes | 100 | 100 | `79f3f1552ccc478ce24bd934b4573625db584ddc1f4643ae23c994a94cb11397` |
| 2 | CMC paper, arXiv 1906.05849v5, Appendix A, Table 6 ("List of ImageNet-100 classes") | The authors' paper | printed table | 100 | 100 | not applicable |
| 3 | [vturrisi/solo-learn `solo/data/dataset_subset/imagenet100_classes.txt`](https://github.com/vturrisi/solo-learn/blob/9187ea39c2c3f43c455ab06664d2b019a9802954/solo/data/dataset_subset/imagenet100_classes.txt) at commit `9187ea39c2c3f43c455ab06664d2b019a9802954` | A widely used self-supervised learning library | one line, space separated, 1,000 bytes | 100 | 100 | `4fc6a97a4fa321b6...` (differs from source 1 only in separators) |
| 4 | `data/imagenet100_classes.txt` in this repo (file time 2026-11-14 19:26) | Project copy | 10 comment lines starting with `#`, then one WNID per line | 100 | 100 | `d3e0f8e043c5bc15...` (whole file, including comments) |

Comparison (script run in the scratchpad, extracting every `n\d{8}` token):

- source 1 against source 3: same set, same order;
- source 1 against source 2 (Table 6): same set, same order;
- source 1 against the project file: same set, same order. The project file's header records the sha256 `79f3f155...1397`, which matches source 1.

The list starts `n02869837, n01749939, n02488291` and ends `n03424325, n01855672, n02090622` in all four.

## What the paper says about the subset

> In this paper, we proposed a subset of ImageNet that contains randomly selected 100 classes for ablation study as well as hyper-parameter tuning.

(Appendix A, "ImageNet-100 Proposed in this Paper".) The caption of Table 6 says the classes "are randomly sampled from the original ImageNet-1k dataset [16]" and that "This list can also be downloaded at: http://github.com/ HobbitLong/CMC/blob/master/imagenet100.txt". The CMC README records "May 22, 2020 - ImageNet-100 list uploaded".

## Citation

- Yonglong Tian, Dilip Krishnan, Phillip Isola. *Contrastive Multiview Coding.* ECCV 2020. arXiv [1906.05849](https://arxiv.org/abs/1906.05849) (v5, 18 Dec 2020). DOI 10.1007/978-3-030-58621-8_45 (from [mlanthology](https://mlanthology.org/eccv/2020/tian2020eccv-contrastive), which gives booktitle "Proceedings of the European Conference on Computer Vision (ECCV)", 2020). The Springer volume and pages were not checked (verify if needed).
- BibTeX key suggestion: `tian2020contrastive`.

## Notes for use

- The order in all sources is not sorted. If labels 0 to 99 are needed, decide whether to use file order or sorted WNID order, and log it. Sorted order matches torchvision's `ImageFolder` on a folder named by WNID.
- Section 4.1 plans 72 x 72 uint8 arrays from the train split and reports on the val split; the class list does not depend on that.
