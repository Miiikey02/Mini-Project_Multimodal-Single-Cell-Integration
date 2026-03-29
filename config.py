from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MULTIOME_DIR = DATA_DIR / "multiome"
CITESEQ_DIR = DATA_DIR / "citeseq"
SRC_DIR = ROOT / "src"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORT_DIR = ROOT / "report"

TRAIN_INPUTS = MULTIOME_DIR / "train_multi_inputs.h5"
TRAIN_TARGETS = MULTIOME_DIR / "train_multi_targets.h5"
TEST_INPUTS = MULTIOME_DIR / "test_multi_inputs.h5"
METADATA = DATA_DIR / "metadata.csv"
EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

TRAIN_DONORS = [13176, 31800, 32606]
TEST_DONOR = 27678
TRAIN_DAYS = [2, 3, 4, 7]
PUBLIC_TEST_DAYS = [2, 3, 7]
PRIVATE_TEST_DAY = 10

CELL_TYPES = ["HSC", "EryP", "MkP", "NeuP", "MoP", "MasP", "BP"]

N_ATAC_PEAKS = 228_942   
N_RNA_GENES = 23_418  
   
LSI_COMPONENTS = 128    

RANDOM_SEED = 42
