"""
Global configuration for the federated model watermarking framework.
Modify these settings to switch datasets, models, aggregation strategies,
watermark methods, and robustness-evaluation parameters.
"""

# ─── Dataset ────────────────────────────────────────────────────────────────
DATASET = "CIFAR10"          # "MNIST" | "CIFAR10"
DATA_DIR = "./data/raw"

# ─── Federated learning ──────────────────────────────────────────────────────
NUM_CLIENTS = 10             # total number of simulated clients
NUM_ROUNDS = 20              # global aggregation rounds
LOCAL_EPOCHS = 2             # local training epochs per round
LOCAL_BATCH_SIZE = 64
LOCAL_LR = 0.01
AGGREGATION = "fedavg"       # "fedavg" | "median" | "trimmed_mean"

# Fraction of clients selected per round (client dropout simulation)
CLIENT_FRACTION = 0.8        # 0 < fraction <= 1.0

# Non-IID data distribution
NON_IID = True               # True  → Dirichlet non-IID; False → IID
DIRICHLET_ALPHA = 0.5        # concentration parameter (smaller = more skewed)

# ─── Watermark ───────────────────────────────────────────────────────────────
WATERMARK_STRATEGY = "badnets"   # "badnets" | "pattern"
WATERMARK_TARGET_LABEL = 0       # label that triggered samples are mapped to
WATERMARK_POISON_RATE = 0.1      # fraction of local data poisoned with trigger
WATERMARK_CLIENTS = [0, 1]       # indices of clients that embed the watermark

# BadNets trigger: small patch in the bottom-right corner
TRIGGER_SIZE = 4             # pixel width/height of the square trigger patch
TRIGGER_VALUE = 255          # pixel intensity of the trigger patch

# Pattern trigger: repeating checkerboard pattern
PATTERN_STRIDE = 8           # distance between pattern pixels

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL = "SimpleCNN"          # "SimpleCNN" | "ResNet18"
NUM_CLASSES = 10

# ─── Pruning robustness evaluation ───────────────────────────────────────────
PRUNE_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]   # fractions of weights set to zero

# ─── Evaluation ──────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 256
DEVICE = "cpu"               # "cpu" | "cuda"

# ─── Output ──────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
SAVE_MODEL = True
MODEL_PATH = "./results/global_model.pth"
