# fap-bench

Federated Asset Pricing Benchmark — code, configurations and the deterministic
assignments, masks, schedules and seeds that regenerate the paper's tables and
figures.

- [`SUPPLEMENT.md`](SUPPLEMENT.md) — derivations, audit figures and robustness
  detail released from the paper under its page limit. Every "see the artifact"
  pointer in the manuscript resolves here.
- [`code/`](code) — the benchmark package: code, configurations and tests.

## Results are regenerated, not shipped

This repository does not contain the result files behind the paper. The paper's
numbers are computed from CRSP (the Center for Research in Security Prices) and
Compustat, two commercial databases that require a subscription, and the full
outputs are about 40 GB.

- **With a subscription:** the publication command described in
  [`code/README.md`](code/README.md) regenerates every table and figure and
  writes them to `code/outputs/icaif_publication/`. The seeds in the
  configurations make the client assignments, feature masks and participation
  schedules deterministic, so a rerun rebuilds the same ones.
- **Without a subscription:** `code/scripts/make_synthetic.py` generates a
  synthetic panel with the same columns, and `code/configs/synthetic_smoke.yaml`
  runs the whole pipeline on it. This exercises every mechanism in the paper but
  reproduces none of its numbers.

## License

Everything in this repository is released under the [MIT License](LICENSE): you
may use, modify and redistribute it, provided the copyright notice is kept.

The license covers this repository's code, configurations and documentation only. The
underlying stock-return and accounting data come from CRSP (the Center for
Research in Security Prices) and Compustat, which are commercial databases
under their own licenses. That data is not included here and must be obtained
separately through a subscription.
