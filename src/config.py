from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "genres_original"
ARTIFACTS = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS / "features"
MODELS_DIR = ARTIFACTS / "models"
REPORTS_DIR = ROOT / "reports" / "generated"

GENRES = ["blues", "classical", "hiphop", "metal", "pop"]
SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 20
HOP_LENGTH = 512
# A fixed frame count makes every flattened MFCC vector the same length.
TARGET_FRAMES = 1290
TEST_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-3

for folder in [FEATURES_DIR, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
