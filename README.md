# nero

https://zeroknowledgediscovery.github.io/nero/
# NERO: Nonparametric Entropy-Rate Oracle
# NERO: Nonparametric Entropy-Rate Oracle

NERO is a model-agnostic, training-free method for estimating the entropy rate of long-form text and using it as an intrinsic complexity signature for distinguishing human-authored prose from AI-generated text.

This repository accompanies the paper **“Complexity Signature of Generated Text”** (Schmidt & Chattopadhyay, 2026). fileciteturn0file0

## What NERO does

NERO maps text to a fixed finite alphabet and treats the resulting symbol stream as a sample path of a stochastic process. It then estimates the **Shannon entropy rate** directly from the observed stream, without:
- access to the generating model,
- token probabilities / perplexity,
- supervised training or calibration.

Empirically, long-form LLM outputs show **lower entropy rates** than human prose under a shared symbolization, enabling competitive AI-text detection and tracking changes in generative behavior over time.

## Method (high level)

1. **Symbolize text**  
   Lowercase and map to a 27-symbol alphabet: `a–z` plus space (punctuation/digits removed).

2. **Estimate entropy rate (PFSA, nonparametric)**  
   Construct empirical next-symbol distributions conditioned on observed substrings and infer PFSA-style “states” as equivalence classes of histories with similar futures.

3. **Robustify across substring-frequency cutoffs**  
   Run the estimator across thresholds `{m1,…,mM}` (ignore substrings occurring `< m` times) and report the **median** entropy-rate estimate:  
   `Ĥ = median(Ĥ(m1), …, Ĥ(mM))`.

4. **Optional trained readout**  
   Use the full cutoff-profile vector as features (e.g., Gaussian Process classifier). The underlying estimator remains unchanged.

## Repository contents

Typical components in this repo include:
- Core PFSA-based entropy-rate estimator
- Preprocessing utilities (symbolization / cleaning)
- Evaluation scripts (ROC/AUC, cohort/length analyses)
- Notebooks for API-based long-form generation (where applicable)
- Figure-generation scripts used in the paper

## Computational cost

For text length `n`, alphabet size `|Σ|`, and `M` cutoffs, runtime is:

`O(n · M · |Σ|)`

For fixed `M`, the estimator is linear in input size.

## License

MIT.

## Citation

If you use this code, please cite:

Schmidt R, Chattopadhyay I. **Complexity Signature of Generated Text.** 2026.
