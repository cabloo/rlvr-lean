"""GPU steps of rlvr_lean. Each runs as its own process in the `gpu` or `quantize` environment, started by
`rlvr_lean.runner.entry`, and writes one JSON result: `python -m rlvr_lean.gpu <step> --out <file>`."""
