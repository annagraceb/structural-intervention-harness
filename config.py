"""
Central configuration for the Structural Intervention Experiment Harness.
All frozen parameters live here. This file is a Stage 1 artifact.
"""
import os
import torch

# === Hardware ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Determinism (Stage 2) ===
RANDOM_SEED = 42

def configure_determinism():
    """Apply all determinism flags per spec v0.4."""
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Note: strict mode (warn_only=False) causes NaN with Qwen2.5 in fp16.
    # warn_only=True is safe because Stage 2 empirically verified bitwise determinism.
    torch.use_deterministic_algorithms(True, warn_only=True)

# === Benchmark ===
# BBH tasks to evaluate — will be finalized during Stage 1 calibration
BBH_TASKS = [
    "tracking_shuffled_objects_three_objects",
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_five_objects",
]
BBH_DATASET = "lukaemon/bbh"
MIN_BENCHMARK_ITEMS = 700

# === Model — will be finalized during Stage 1 ===
MODEL_ID = "Qwen/Qwen2.5-1.5B"  # best accuracy in calibration sweep
MODEL_REVISION = None  # pin after calibration

# === Evaluation ===
MAX_NEW_TOKENS = 5  # for log-prob eval we don't generate, but safety limit

# === Statistical framework ===
BH_FDR = 0.05
TIER2_MIN_ACCURACY_DELTA_PP = 3.0  # percentage points
TIER2_MIN_DIRECTIONAL_RATIO = 0.70  # 70% of discordant pairs same direction

# === Budget ===
TOTAL_GPU_HOURS = 100
OVERHEAD_FRACTION = 0.10
EFFECTIVE_GPU_HOURS = TOTAL_GPU_HOURS * (1 - OVERHEAD_FRACTION)

# === Paths ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "experiment.db")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
CALIBRATION_DIR = os.path.join(ARTIFACTS_DIR, "stage1")
BASELINE_DIR = os.path.join(ARTIFACTS_DIR, "stage2")

for d in [ARTIFACTS_DIR, CALIBRATION_DIR, BASELINE_DIR]:
    os.makedirs(d, exist_ok=True)
