# Original GPWNO training logs

These are unmodified copies of the original `run.log` files, downloaded on
2026-11-20 from `checkpoints_and_logs/` in the GPWNO training archive. Their MD5
checksums match the source files. They record historical training and evaluation,
not a new accuracy evaluation with the current PaddleMaterials interfaces.

| Model | Original run | MD5 |
| --- | --- | --- |
| [Benzene](gpwno_md17_benzene.log) | `gpwno_md17_benzene_t_20260527_182931_s_42` | `54a52e317bb099caacb8029c1ac1db26` |
| [Ethane](gpwno_md17_ethane.log) | `gpwno_md17_ethane_t_20260529_204928_s_42` | `332b08781bbbf60a7e227712ea78a7c1` |
| [QM9](gpwno_qm9.log) | `gpwno_qm9_t_20260626_164201_s_42` | `298e4e779d4e9451aa0df4eaac49263a` |
| [MP](gpwno_mp.log) | `gpwno_mp_resume_t_20260528_204842_s_42` | `75b535aa2937e4f36defe8f9ddfbb471` |

The logs report NMAE as a fraction under `mae(loss)`; the results table multiplies
it by 100. They print six decimal places, so the table retains the extra precision
from the saved training state where available.

| Model | Validation evidence (line) | Test evidence (line) |
| --- | --- | --- |
| Benzene | [1297](gpwno_md17_benzene.log#L1297): epoch 10, `0.035567` | No test summary recorded |
| Ethane | [4533](gpwno_md17_ethane.log#L4533): standalone validation, `0.048302` | [4738](gpwno_md17_ethane.log#L4738): `0.048339` |
| QM9 | [657](gpwno_qm9.log#L657): epoch 2, `0.070865` | [876](gpwno_qm9.log#L876): `0.074001` |
| MP | [1250](gpwno_mp.log#L1250): standalone validation, `0.348928` | [2254](gpwno_mp.log#L2254): `0.378910` |

The legacy logger labels both validation and test passes `Eval`. For ethane, the
validation pass has 13 batches and the following test pass has 201; MP has 167
and 1,000; QM9 has 1 and 209. QM9 also records standalone validation of `0.073232`
at line 663. The table uses its training-validation result, as identified above.
Ethane's best training-validation result is `0.047792` (line 4374), and MP's is
`0.348229` (line 1078); these differ from their standalone validation results
because grid points are sampled. The original checkpoint states identify best
epochs 10 (benzene), 31 (ethane), and 2 (QM9 and MP).

Benzene ends during epoch 11 after completing epoch 10. MP and QM9 contain the
resumed epoch-2 training and subsequent evaluations, without the earlier training
segment. These logs therefore do not establish all previously reported training
times. The original training configurations and checkpoint states were preserved
when the inference packages were migrated.
