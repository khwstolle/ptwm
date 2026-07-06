---
title: Arithmetic
description: Adaptive and static arithmetic plane codecs (Order-0 and Order-1).
---

# Arithmetic Codecs

PTWM includes three closely related arithmetic-coding plane codecs that share a common range coder: `ArithmeticO0`, `ArithmeticO0Adaptive`, and `ArithmeticO1`.

## Theory

Arithmetic coding represents a message as a fractional number on the number line between 0 and 1. It provides near-optimal entropy compression by allowing symbols to consume fractional bits of space.

```text
[0.0                                                     1.0)
 |----------|--------------------|------------------------|
 0.0       0.2                  0.6                      1.0
  Symbol A       Symbol B                Symbol C
  (p=0.2)        (p=0.4)                 (p=0.4)
```

- **ArithmeticO0 (Static)**: Fits a static 0-order probability model over the entire plane and serializes the probability table into the codec state.
- **ArithmeticO0Adaptive**: An adaptive 0-order coder that starts with a uniform distribution and updates its probabilities dynamically as it observes symbols. It emits no inline state.
- **ArithmeticO1 (Adaptive)**: An adaptive order-1 (lag-1) coder that maintains a separate probability context for every possible previous byte value (256 contexts). This captures first-order markov dependencies (e.g., smoothly varying integers or correlated text).

## Usage

These codecs are primarily used when the data exhibits strong symbol skew or lag-1 correlations. Because they are self-delimiting, they frame the true decoded length into their payload, ignoring the source-derived length provided by the container.

## References

* I. H. Witten, R. M. Neal, and J. G. Cleary, "Arithmetic Coding for Data Compression", *Communications of the ACM* (1987). [Link](https://dl.acm.org/doi/10.1145/214762.214771)
