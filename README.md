# fap-bench

Federated Asset Pricing Benchmark — code, configurations and the deterministic
assignments, masks, schedules and seeds that regenerate the paper's tables and
figures.

- [`SUPPLEMENT.md`](SUPPLEMENT.md) — derivations, audit figures and robustness
  detail released from the paper under its page limit. Every "see the artifact"
  pointer in the manuscript resolves here.
- [`code/`](code) — the benchmark package. Outputs backing the paper are under
  `code/outputs/icaif_publication/`.

## License

Everything in this repository is released under the [MIT License](LICENSE): you
may use, modify and redistribute it, provided the copyright notice is kept.

The license covers this repository's code, configurations and outputs only. The
underlying stock-return and accounting data come from CRSP (the Center for
Research in Security Prices) and Compustat, which are commercial databases
under their own licenses. That data is not included here and must be obtained
separately through a subscription.
