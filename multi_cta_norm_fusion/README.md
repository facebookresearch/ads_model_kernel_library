# TLX Multi-CTA Norm Fusion

Triton TLX kernels for fused matmul with RMSNorm and LayerNorm forward and backward paths.

## Layout

- `src/` - TLX kernel implementations and shared math helpers
- `tests/` - GPU correctness tests

## Run

```bash
conda env create -f environment.yml
conda activate multi-cta-norm-fusion
python -m unittest discover -s tests -p "test_*.py"
```

The OSS test harness defaults `ADS_MKL_DISABLE_AUTOTUNE=1` before importing the kernels. This keeps each autotuned kernel on a single known config so correctness tests do not spend time benchmarking the full config space. To run the full autotune sweep manually, set `ADS_MKL_DISABLE_AUTOTUNE=0` before launching your script.

If conda channel access is restricted, create the environment with any available Python 3.12 conda channel and install the Python packages with pip:

```bash
conda create -n multi-cta-norm-fusion python=3.12 pip
conda activate multi-cta-norm-fusion
pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch
pip install fbtriton==3.6.1
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 ADS_MKL_DISABLE_AUTOTUNE=1 python -m unittest discover -s tests -p "test_tlx_*.py"
```

To verify TLX is importable:

```bash
python -c 'import triton.language.extra.tlx as tlx; print(tlx)'
```

For interactive use outside the tests, add the kernel sources to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```
