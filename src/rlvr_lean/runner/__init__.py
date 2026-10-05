"""The stage runner: `entry` runs one stage's steps on the GPU box, each as its own process
(`python -m rlvr_lean.runner.entry --stage <stage> --out <dir>`); `heartbeat` writes TensorBoard events.
Standard library only."""
