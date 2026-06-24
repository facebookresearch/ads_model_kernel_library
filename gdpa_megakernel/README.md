# TLX GDPA Megakernel

Triton TLX generalized dot product attention megakernel for Blackwell.

## Layout

- `src/` - TLX kernel implementation and local helper modules
- `tests/` - GPU correctness tests

## Run

```bash
conda env create -f environment.yml
conda activate gdpa-megakernel
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 ADS_MKL_DISABLE_AUTOTUNE=1 python -m unittest discover -s tests -p "test_*.py"
```

The OSS test harness defaults `ADS_MKL_DISABLE_AUTOTUNE=1` before importing the kernel. This keeps correctness tests on a single known config instead of benchmarking the full autotune space. To run the full autotune sweep manually, set `ADS_MKL_DISABLE_AUTOTUNE=0` before launching your script.

If conda channel access is restricted, create the environment with any available Python 3.12 conda channel and install the Python packages with pip:

```bash
conda create -n gdpa-megakernel python=3.12 pip
conda activate gdpa-megakernel
pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch
pip install fbtriton==3.6.1
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 ADS_MKL_DISABLE_AUTOTUNE=1 python -m unittest discover -s tests -p "test_*.py"
```

To verify TLX is importable:

```bash
python -c 'import triton.language.extra.tlx as tlx; print(tlx)'
```

For interactive use outside the tests, add the kernel sources to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```
