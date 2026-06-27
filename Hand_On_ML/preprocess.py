import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from deslib.des.knora_e import KNORAE # (Stacking)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


import xgboost as xgb
import lightgbm as lgb
import catboost as cgb

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")

"""
Healthcare Cost Prediction — Preprocessing Pipeline  v2
========================================================
Key changes from v1
  - NO temporal train/val split (2024 is test-only, not in training CSVs)
  - Stratified K-Fold applied AFTER target definition so folds are class-balanced
  - Returns X, y (regression), y_cls (binary HighCost), skf_splits for downstream use
  - process_main / dob / cpt / drg / icd helpers unchanged
"""

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = r"/home/noneo/Codes/ML/MachineLearning/Hand_On_ML/Data/Healthcare Cost Dataset"
TRAINING_PATH = os.path.join(BASE_DIR, "train", "train") + os.sep
TESTING_PATH  = os.path.join(BASE_DIR, "test",  "test")  + os.sep

COST_THRESHOLD = 30_000      # HighCost binary label
N_FOLDS        = 5
RANDOM_STATE   = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — main_df cleaning
# ═══════════════════════════════════════════════════════════════════════════════
def process_main(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only yearly roll-up rows
    df = df[df["MONTH"] == -1].reset_index(drop=True)

    # Drop leakage / constant / identifier columns
    LEAKAGE_COLS = [
        "MONTH", "ActualYear", "ActualMonthNumber", "IsYTD", "QUARTER",
        "PredictedCost", "Leakage_Cost", "IsLatest",
    ]
    df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True)

    # Date engineering
    ref_date = pd.Timestamp("2023-01-01")
    for raw_col, new_col in [("AWVLastDate", "DaysSinceAWV"),
                              ("LastPCPVisit", "DaysSinceLastPCP")]:
        if raw_col in df.columns:
            df[raw_col] = pd.to_datetime(df[raw_col], errors="coerce")
            df[new_col] = (ref_date - df[raw_col]).dt.days.clip(lower=0)
            df.drop(columns=[raw_col], inplace=True)

    # Column groups
    cost_cols = [c for c in df.columns if c.endswith("_Cost") and c != "TotalCost"]
    count_cols = [c for c in [
        "ProviderVisitCount","ERAdmisison","30dayHospitalReadmission","OutpatientVisits",
        "Inpatient","HomeHealth","Hospice","SkilledNursingFacilities",
        "EmergencyDepartmentVisits","EmergencyDepartmentVisitsWithAdmissions",
        "CTEvents","MRIEvents","RadiologyEvents","OtherImagingServices",
        "LabEventsPathalogy","LabEventsClinicalDiagnostics","PrimaryCareServicesTotal",
        "PrimaryCareServiceswithPrimaryCarePhysician","PrimaryCareServicesWithSpecialistPhysician",
        "PrimaryCareServicesWithNursePractitioner","30DayReadmission","NewPatients",
        "EstablishedPatients","PostDischargeVisits","AmbulanceEvents","Medication",
        "PCPVisits","OfficeVisits","EDVisitsWithNoFollowUp","IPVisitsWithNoFollowUp",
        "TotalClaims","TotalScriptsFilled","AWVVisits","PediatricsVisits","UrgentCareVisit",
        "AmbulatorySurgeryVisit","DentistEvents",
    ] if c in df.columns]
    los_cols    = [c for c in df.columns if "LOS" in c]
    binary_cols = [c for c in [
        "IsCovid","DoneBySelf","AWV","Admission","InNetworkPCP","Ishighrisk",
        "AWV_Compliant","AWV_Eligible","IPPE_Eligible","IPPE_Compliant",
        "AWV_Compliant_Eligible","InNetworkAWVCompliant","OutNetworkAWVCompliant",
        "OutNetworkPCP","NonClaimBasedPayment",
    ] if c in df.columns]
    disease_cols = [c for c in [
        "ChronicObstructivePulmonaryDiseaseOrAsthma","CongestiveHeartFailure",
        "BacterialPneumonia","DiabetesShortTermComplications","DiabetesLongTermComplications",
        "UncontrolledDiabetes","AmputationDiabetes","Dialysis","ChronicConditions",
    ] if c in df.columns]
    hcc_cols = [c for c in [
        "MemberHccScore","MemberHccScoreLastYear","MemberHccScoreLastTwoYear","PersiviaMemberHccScore",
    ] if c in df.columns]
    cat_cols = [c for c in [
        "AWVCode","AWVStatus","AttributionStatus","LastPCPProvider",
        "AWVProviderNetwork","Payers_key","Enrollment_key",
    ] if c in df.columns]

    # Fill strategies
    df[cost_cols]    = df[cost_cols].fillna(0).clip(lower=0)
    df[count_cols]   = df[count_cols].fillna(0).clip(lower=0)
    df[los_cols]     = df[los_cols].fillna(0).clip(lower=0)
    df[binary_cols]  = df[binary_cols].fillna(0).replace({True:1,False:0}).astype(int)
    df[disease_cols] = df[disease_cols].fillna(0).astype(int)
    df[hcc_cols]     = df[hcc_cols].apply(lambda c: c.fillna(c.median()))

    # Frequency-encode categoricals
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
        freq     = df[col].value_counts(normalize=True)
        df[col]  = df[col].map(freq)

    # Fill any remaining numeric NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    print(f"    main_df → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — dob_df
# ═══════════════════════════════════════════════════════════════════════════════
def process_dob(dob: pd.DataFrame, reference_year: int = 2023) -> pd.DataFrame:
    dob = dob.copy()
    dob["DOB_Key"]     = pd.to_datetime(dob["DOB_Key"], errors="coerce")
    dob["Age"]         = reference_year - dob["DOB_Key"].dt.year
    dob["Age"]         = dob["Age"].clip(lower=0).fillna(dob["Age"].median())
    dob["Gender_Key"]  = dob["Gender_Key"].fillna(dob["Gender_Key"].mode()[0]).astype(int)
    dob["AgeGroup_Key"]= dob["AgeGroup_Key"].fillna(dob["AgeGroup_Key"].mode()[0]).astype(int)
    dob.drop(columns=["DOB_Key"], inplace=True)
    print(f"    dob_df  → {dob.shape[0]:,} rows × {dob.shape[1]} cols")
    return dob


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — cpt_df aggregation
# ═══════════════════════════════════════════════════════════════════════════════
def aggregate_cpt(cpt: pd.DataFrame) -> pd.DataFrame:
    cpt = cpt.copy()
    cpt["Procedure_Code"] = cpt["Procedure_Code"].astype(str)
    grp = cpt.groupby(["Member_Key", "StartDate"])
    agg = grp["Procedure_Code"].agg(
        cpt_ProcedureCount   ="count",
        cpt_UniqueProcedures ="nunique",
        cpt_TopProcedure     =lambda x: x.mode().iloc[0] if len(x) else "Unknown",
    ).reset_index()
    agg["cpt_ProcedureDiversity"] = agg["cpt_UniqueProcedures"] / agg["cpt_ProcedureCount"].replace(0,1)
    freq = agg["cpt_TopProcedure"].value_counts(normalize=True)
    agg["cpt_TopProcedure"] = agg["cpt_TopProcedure"].map(freq).fillna(0)
    agg.rename(columns={"StartDate":"YEAR"}, inplace=True)
    print(f"    cpt_df  → {agg.shape[0]:,} rows after aggregation")
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — drg_df aggregation
# ═══════════════════════════════════════════════════════════════════════════════
def aggregate_drg(drg: pd.DataFrame) -> pd.DataFrame:
    drg = drg.copy()
    drg["Code"] = drg["Code"].astype(str)
    grp = drg.groupby(["Member_Key","Start_Date_Year"])
    agg = grp["Code"].agg(
        drg_ClaimsCount="count",
        drg_UniqueDRGs ="nunique",
        drg_MostCommon =lambda x: x.mode().iloc[0] if len(x) else "Unknown",
    ).reset_index()
    drg_freq  = drg["Code"].value_counts()
    rare_drgs = set(drg_freq[drg_freq == 1].index)
    drg["is_rare"] = drg["Code"].isin(rare_drgs).astype(int)
    rare_agg  = drg.groupby(["Member_Key","Start_Date_Year"])["is_rare"].sum().reset_index()
    rare_agg.rename(columns={"is_rare":"drg_RareDRGCount"}, inplace=True)
    agg = agg.merge(rare_agg, on=["Member_Key","Start_Date_Year"], how="left")
    freq = agg["drg_MostCommon"].value_counts(normalize=True)
    agg["drg_MostCommon"] = agg["drg_MostCommon"].map(freq).fillna(0)
    agg.rename(columns={"Start_Date_Year":"YEAR"}, inplace=True)
    print(f"    drg_df  → {agg.shape[0]:,} rows after aggregation")
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — icd_df aggregation
# ═══════════════════════════════════════════════════════════════════════════════
def aggregate_icd(icd: pd.DataFrame) -> pd.DataFrame:
    icd = icd.copy()
    icd["Diagnosis_Code"] = icd["Diagnosis_Code"].astype(str)
    CHRONIC_PREFIXES = ("E11","E10","I50","I25","J44","J45","N18","G20","F32","F41","M79")
    icd["is_chronic"] = icd["Diagnosis_Code"].str.startswith(CHRONIC_PREFIXES).astype(int)
    grp = icd.groupby(["Member_Key","Start_Date"])
    agg = grp.agg(
        icd_DiagnosisCount       =("Diagnosis_Code","count"),
        icd_UniqueDiagnoses      =("Diagnosis_Code","nunique"),
        icd_ChronicDiagnosisCount=("is_chronic","sum"),
        icd_TopICD               =("Diagnosis_Code", lambda x: x.mode().iloc[0] if len(x) else "Unknown"),
    ).reset_index()
    agg["icd_ICDDiversity"] = agg["icd_UniqueDiagnoses"] / agg["icd_DiagnosisCount"].replace(0,1)
    freq = agg["icd_TopICD"].value_counts(normalize=True)
    agg["icd_TopICD"] = agg["icd_TopICD"].map(freq).fillna(0)
    agg.rename(columns={"Start_Date":"YEAR"}, inplace=True)
    print(f"    icd_df  → {agg.shape[0]:,} rows after aggregation")
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — merge all tables
# ═══════════════════════════════════════════════════════════════════════════════
def build_feature_table(main, dob, cpt, drg, icd) -> pd.DataFrame:
    for df in [main, cpt, drg, icd]:
        df["Member_Key"] = df["Member_Key"].astype(int)
        if "YEAR" in df.columns:
            df["YEAR"] = df["YEAR"].astype(int)
    dob["Member_Key"] = dob["Member_Key"].astype(int)

    merged = main.merge(dob, on="Member_Key",          how="left")
    merged = merged.merge(cpt, on=["Member_Key","YEAR"], how="left")
    merged = merged.merge(drg, on=["Member_Key","YEAR"], how="left")
    merged = merged.merge(icd, on=["Member_Key","YEAR"], how="left")

    agg_fill = [c for c in merged.columns if c.startswith(("cpt_","drg_","icd_"))]
    merged[agg_fill] = merged[agg_fill].fillna(0)
    print(f"    merged  → {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE  — returns X, y_reg, y_cls, skf_splits, merged
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(main_raw, cpt_raw, drg_raw, icd_raw, dob_raw,
                 cost_threshold=COST_THRESHOLD, n_folds=N_FOLDS):
    """
    Process all training data.  No temporal split.
    Returns
      X          - feature matrix (DataFrame)
      y_reg      - TotalCost (regression target)
      y_cls      - binary HighCost label  (TotalCost > cost_threshold)
      skf_splits - list of (train_idx, val_idx) tuples from StratifiedKFold
      merged     - full merged DataFrame (for EDA)
    """
    print("\n" + "="*55)
    print("  Processing full training set")
    print("="*55)

    main_c = process_main(main_raw)
    dob_c  = process_dob(dob_raw)
    cpt_a  = aggregate_cpt(cpt_raw)
    drg_a  = aggregate_drg(drg_raw)
    icd_a  = aggregate_icd(icd_raw)

    merged = build_feature_table(main_c, dob_c, cpt_a, drg_a, icd_a)

    # Target
    y_reg = merged["TotalCost"].clip(lower=0).reset_index(drop=True)
    y_cls = (y_reg > cost_threshold).astype(int)

    # Feature matrix
    DROP = ["Member_Key", "YEAR", "TotalCost"]
    X = merged.drop(columns=[c for c in DROP if c in merged.columns]).copy()
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"    [WARN] dropping non-numeric: {non_num}")
        X.drop(columns=non_num, inplace=True)

    # ── Stratified K-Fold splits ─────────────────────────────────────────────
    skf        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    skf_splits = list(skf.split(X, y_cls))     # list of (train_idx, val_idx)

    # Quick class report
    pos_rate = y_cls.mean()
    print(f"\n  ✓  X          : {X.shape}")
    print(f"  ✓  y_reg      : min={y_reg.min():.0f}  median={y_reg.median():.0f}  max={y_reg.max():,.0f}")
    print(f"  ✓  y_cls      : {y_cls.sum():,} positive / {len(y_cls):,} total  ({pos_rate*100:.1f}%)")
    print(f"  ✓  SKF folds  : {n_folds}  (each val fold ≈ {len(y_cls)//n_folds:,} rows)")
    print()

    # Fold-level class balance check
    print("  Fold  |  Train+  Train-  |  Val+   Val-   |  Val pos%")
    print("  ─────────────────────────────────────────────────────")
    for i, (tr, va) in enumerate(skf_splits):
        tr_pos = y_cls.iloc[tr].sum(); tr_neg = len(tr) - tr_pos
        va_pos = y_cls.iloc[va].sum(); va_neg = len(va) - va_pos
        print(f"  {i+1}     |  {tr_pos:6,}  {tr_neg:6,}  |  {va_pos:5,}  {va_neg:6,}  |  {va_pos/len(va)*100:.1f}%")

    return X, y_reg, y_cls, skf_splits, merged


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os

    print("[1/3] Loading raw files …")
    main_df = pd.read_csv(TRAINING_PATH + "main_df_train.csv", low_memory=False)
    cpt_df  = pd.read_csv(TRAINING_PATH + "cpt_df_train.csv",  low_memory=False)
    drg_df  = pd.read_csv(TRAINING_PATH + "drg_df_train.csv",  low_memory=False)
    icd_df  = pd.read_csv(TRAINING_PATH + "icd_df_train.csv",  low_memory=False)
    dob_df  = pd.read_csv(TRAINING_PATH + "dob_df.csv",         low_memory=False)

    print("[2/3] Running pipeline …")
    X, y_reg, y_cls, skf_splits, merged = run_pipeline(
        main_df, cpt_df, drg_df, icd_df, dob_df
    )

    print("[3/3] Quick sanity check on fold 1 …")
    tr_idx, va_idx = skf_splits[0]
    X_tr, X_va   = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va   = y_cls.iloc[tr_idx], y_cls.iloc[va_idx]
    print(f"  Fold 1 train: {X_tr.shape}  |  val: {X_va.shape}")
    print(f"  Train pos%: {y_tr.mean()*100:.1f}%   Val pos%: {y_va.mean()*100:.1f}%")
    print("\n[Done] preprocess.py complete.")