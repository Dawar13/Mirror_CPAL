# Failed checks, reported as failures

| Phase | Check | Value | Bar |
|---|---|---|---|
| earlier | F4 energy rule (P2): predicts the single-block winner | 0.4444 | >= 0.75 |
| earlier | Weights-only twin pairing (P5 part 2): agreement with the data-driven pairing | 0.4658 | >= 0.7 |
| earlier | Weights-only twin pairing, fully data-free (P5 part 2, strict) | 0.4409 | >= 0.7 |
| earlier | gate G4: below the bar, claim C6 is dropped |  | P5 passes |
| earlier | gate G2: P3.a and P3.b (else stop and discuss the framing) |  | both pass |
| earlier | all three P3 bullets |  | all pass |
| earlier | healed Mirror minus healed SVD | 1.2400 | >= 3 |
| earlier | SigLIP2 breaking share minus DINOv2-reg breaking share (row 4 definition, fit seed 0) | -0.0861 | > 0 |
| earlier | P5 (parts 1 and 2 as D2 defines them) |  | both parts pass |
| earlier | X4 folding check (plain, float32) |  |  |
| earlier | X4 folding check (plain, float64) |  |  |
| earlier | X4 folding check (reg, float32) |  |  |

Sources: `PAPER_NUMBERS.json failed_checks`
