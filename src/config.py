from pathlib import Path

SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROMOTION_DATA_PATH = ROOT_DIR / "data" / "promotion_train.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
TABLES_DIR = ROOT_DIR / "reports" / "tables"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TARGET_COL = "Attrition_binary"
SENSITIVE_ATTRS = ["Gender", "AgeGroup"]
DROP_COLS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

PROMOTION_TARGET_COL = "Promotion_binary"
PROMOTION_SENSITIVE_ATTRS = ["gender", "AgeGroup"]
PROMOTION_DROP_COLS = ["employee_id"]
