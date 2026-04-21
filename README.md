# nano-llama

A minimal implementation of the Llama-style Transformer architecture

## Features (planned)
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- SwiGLU Feedforward
- Grouped Query Attention (GQA)
- Mixture of Experts (MoE)

## Project Steps

### ✅ Step 1 — Repository Initialized
- GitHub repo created
- README added

### ✅ Step 2 — Project Structure Initialized
- Created core files:
  - `nano_llama.py`
  - `train.py`
  - `infer.py`
- Created folders:
  - `model/`
  - `utils/`
- Added `requirements.txt`

### ✅ Step 3 & 4 — Python Environment Setup
- Created virtual environment (`venv`)
- Installed dependencies (`torch`, `numpy`)
- Verified model runs successfully

### ✅ Step 5 — Base Entry Script
- Separated model definition from execution
- Created `train.py` as entry point
- Cleaned `nano_llama.py` (model-only)
- Verified execution via `train.py`

## Status
Step 5 done
