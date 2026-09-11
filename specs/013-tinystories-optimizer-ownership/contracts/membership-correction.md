# Concat membership correction contract v1

Applies only to new schema-2 campaign contracts. Ordered trained labels are
g250/g500/g750/g1000, counts [4,3,2,1], factors [1,4/3,2,4]. The numerator is
configured trained-width count (4), never current action cardinality (1).

GMC: corrected backward -> arm clipping -> AdamW -> global clock/accounting.
LMC: corrected backward -> arm clipping -> AdamW -> full parameter-change scaling
-> global clock/accounting. FFN gate/up/down blocks and gate/up block biases are
corrected. Shared down bias and all common parameters have direct factor 1.
LMC scales decay too; it never directly modifies moments or counters. No-gradient
blocks are skipped even if previous wider updates populated their AdamW state.

C1 has one optimizer; C2 has independent width histories over the same weights;
C3 has five disjoint owners, active prefix plus common. Apply each correction
exactly once and only complete updates advance the sole schedule. Base scheduled
LR and correction factors are separate artifacts; block effective LR in LMC is
base LR times factor. The GMC multiplier describes gradients, not effective LR.

The correction field includes schema_version, mode, trained_widths,
membership_counts, factors, gradient_correction, parameter_change_correction,
scales_weight_decay, common_factor, parameter_scope, moment_scaling and order.
Full scientific hashing/checkpoint validation rejects any mismatch. Absence is
permitted only on original schema-1 uncorrected artifacts; never synthesize a
new contract on an old checkpoint. Partial owner/correction/bookkeeping failure
poisons live state and preserves the last durable checkpoint.
