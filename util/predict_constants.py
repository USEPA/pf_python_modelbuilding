'''
Created on Feb 10, 2026

@author: TMARTI02
'''
# predict_constants.py
from typing import Final

"""Numerical and label constants for processing and modeling."""

URL_CTX_API = "https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search"
URL_LOCAL_FILE_API = "/api/predictor_models/model/file" #relative path


# Labels
TEST_FRAGMENTS: Final[str] = "TEST Fragments"
SOURCE_CHEMINFORMATICS_MODULES: Final[str] = "Cheminformatics Modules"

# Fractions and tolerances
# Fraction agreement required to map a DSSTox conflict
CONFLICT_FRAC_AGREE: Final[float] = 1.0
# Fraction agreement required to merge binary data points
BINARY_FRAC_AGREE: Final[float] = 0.8
# Cutoff for binary classification
BINARY_CUTOFF: Final[float] = 0.5

# Multiple of dataset stdev required to exclude a property value based on its stdev
STDEV_WIDTH_TOLERANCE: Final[float] = 3.0
# Range tolerance values
LOG_RANGE_TOLERANCE: Final[float] = 1.0
TEMP_RANGE_TOLERANCE: Final[float] = 10.0
DENSITY_RANGE_TOLERANCE: Final[float] = 0.1

ZERO_TOLERANCE: Final[float] = 1e-6

# Ports for other web services
PORT_STANDARDIZER_OPERA: Final[int] = 5001
PORT_TEST_DESCRIPTORS: Final[int] = 5002
PORT_JAVA_MODEL_BUILDING: Final[int] = 5003
PORT_PYTHON_MODEL_BUILDING: Final[int] = 5004
PORT_OUTLIER_DETECTION: Final[int] = 5006
PORT_REPRESENTATIVE_SPLIT: Final[int] = 5005
PORT_STANDARDIZER_JAVA: Final[int] = 5010

# DSSTox mapping strategies
MAPPING_BY_CASRN: Final[str] = "CASRN"
MAPPING_BY_DTXCID: Final[str] = "DTXCID"
MAPPING_BY_DTXSID: Final[str] = "DTXSID"
MAPPING_BY_LIST: Final[str] = "LIST"

# Standardizer types
QSAR_READY: Final[str] = "QSAR_READY"
MS_READY: Final[str] = "MS_READY"

# Standardizers
STANDARDIZER_NONE: Final[str] = "NONE"  # Default QSAR-ready SMILES from DSSTox
STANDARDIZER_OPERA: Final[str] = "OPERA"
STANDARDIZER_SCI_DATA_EXPERTS: Final[str] = "SCI_DATA_EXPERTS"

# Descriptor set names
DESCRIPTOR_SET_TEST: Final[str] = "T.E.S.T. 5.1"
DESCRIPTOR_SET_WEBTEST: Final[str] = "WebTEST-default"
DESCRIPTOR_SET_MORDRED: Final[str] = "Mordred-default"
DESCRIPTOR_SET_PADEL_SINGLE: Final[str] = "Padelpy webservice single"
DESCRIPTOR_SET_PADEL_BATCH: Final[str] = "Padelpy_batch"

####################################################################################################################
# Applicability Domain (AD) labels (used in python webservice for AD calculations)
Applicability_Domain_TEST_Embedding_Cosine: Final[str] = "TEST Cosine Similarity Embedding Descriptors"
Applicability_Domain_TEST_Embedding_Euclidean: Final[str] = "TEST Euclidean Distance Embedding Descriptors"
Applicability_Domain_TEST_All_Descriptors_Cosine: Final[str] = "TEST Cosine Similarity All Descriptors"
Applicability_Domain_TEST_All_Descriptors_Euclidean: Final[str] = "TEST Euclidean Distance All Descriptors"
Applicability_Domain_TEST_Fragment_Counts: Final[str] = "TEST Fragment Counts"

Applicability_Domain_OPERA_local_index: Final[str] = "OPERA Local Index"
Applicability_Domain_OPERA_global_index: Final[str] = "OPERA Global Index"

Applicability_Domain_OPERA_local_index_description: Final[str] = (
    "Local applicability domain index is relative to the similarity of the query chemical to its five nearest neighbors from the training set"
)
Applicability_Domain_OPERA_global_index_description: Final[str] = "Global applicability domain via the leverage approach"

Applicability_Domain_OPERA_confidence_level: Final[str] = "OPERA Confidence Level"
Applicability_Domain_Kernel_Density: Final[str] = "Kernel Density"

Applicability_Domain_Combined: Final[str] = "Combined Applicability Domain"

####################################################################################################################
# Property names (display on dashboard; correspond to name_ccd in qsar_datasets.properties)
WATER_SOLUBILITY: Final[str] = "Water Solubility"  # OPERA
HENRYS_LAW_CONSTANT: Final[str] = "Henry's Law Constant"  # OPERA
MELTING_POINT: Final[str] = "Melting Point"  # OPERA
LOG_KOW: Final[str] = "LogKow: Octanol-Water"  # OPERA
LOG_KOA: Final[str] = "LogKoa: Octanol-Air"  # TODO add log() to it?
VAPOR_PRESSURE: Final[str] = "Vapor Pressure"  # OPERA

DENSITY: Final[str] = "Density"
VAPOR_DENSITY: Final[str] = "Vapor Density"
BOILING_POINT: Final[str] = "Boiling Point"  # OPERA
FLASH_POINT: Final[str] = "Flash Point"
VISCOSITY: Final[str] = "Viscosity"
SURFACE_TENSION: Final[str] = "Surface Tension"
THERMAL_CONDUCTIVITY: Final[str] = "Thermal Conductivity"

MOLAR_REFRACTIVITY: Final[str] = "Molar Refractivity"
MOLAR_VOLUME: Final[str] = "Molar Volume"
POLARIZABILITY: Final[str] = "Polarizability"
PARACHOR: Final[str] = "Parachor"
INDEX_OF_REFRACTION: Final[str] = "Index of Refraction"
DIELECTRIC_CONSTANT: Final[str] = "Dielectric Constant"

AUTOIGNITION_TEMPERATURE: Final[str] = "Autoignition Temperature"

PKA: Final[str] = "pKa"
LOG_BCF: Final[str] = "LogBCF"  # OLD OPERA

BCF: Final[str] = "Bioconcentration Factor"  # OPERA
BAF: Final[str] = "Bioaccumulation Factor"

ULTIMATE_BIODEG: Final[str] = "Ultimate biodegradation timeframe"
PRIMARY_BIODEG: Final[str] = "Primary biodegradation timeframe"

BIODEG_ANAEROBIC: Final[str] = "Biodegradability (anaerobic)"
BIODEG: Final[str] = "Biodegradability"

LOG_OH: Final[str] = "LogOH"  # OLD OPERA
OH: Final[str] = "Atmospheric hydroxylation rate"  # OPERA

LOG_KOC: Final[str] = "LogKOC"  # OLD OPERA
KOC: Final[str] = "Soil Adsorp. Coeff. (Koc)"  # OPERA

LOG_HALF_LIFE: Final[str] = "LogHalfLife"  # OLD OPERA
BIODEG_HL_HC: Final[str] = "Biodegradation half-life for hydrocarbons"  # OPERA

LOG_KM_HL: Final[str] = "LogKmHL"  # OLD OPERA
KmHL: Final[str] = "Fish biotransformation half-life (Km)"  # OPERA

LOG_BCF_FISH_WHOLEBODY: Final[str] = "LogBCF_Fish_WholeBody"  # should just be Fish whole body bioconcentration factor

# Additional OPERA properties:
RBIODEG: Final[str] = "Ready biodegradability"  # OPERA (binary)
FUB: Final[str] = "Fraction unbound in human plasma"  # OPERA
RT: Final[str] = "Liquid chromatography retention time"  # OPERA
CLINT: Final[str] = "Human hepatic intrinsic clearance"  # OPERA
CACO2: Final[str] = "Caco-2 permeability (Papp)"  # OPERA
LogD_pH_7_4: Final[str] = "LogD at pH=7.4"  # OPERA
LogD_pH_5_5: Final[str] = "LogD at pH=5.5"  # OPERA

PKA_A: Final[str] = "Acidic pKa"
PKA_B: Final[str] = "Basic pKa"

# Old versions for building sample models
MUTAGENICITY: Final[str] = "Mutagenicity"
LC50: Final[str] = "LC50"
LC50DM: Final[str] = "LC50DM"
IGC50: Final[str] = "IGC50"
LD50: Final[str] = "LD50"
LLNA: Final[str] = "LLNA"
DEV_TOX: Final[str] = "DevTox"

SKIN_IRRITATION: Final[str] = "Skin irritation"  # TODO This needs to be more specific
EYE_IRRITATION: Final[str] = "Eye irritation"  # TODO This needs to be more specific
EYE_CORROSION: Final[str] = "Eye corrosion"  # TODO This needs to be more specific

# New versions for dashboard
NINETY_SIX_HOUR_FATHEAD_MINNOW_LC50: Final[str] = "96 hour fathead minnow LC50"
NINETY_SIX_HOUR_FISH_LC50: Final[str] = "96 hour fish LC50"
NINETY_SIX_HOUR_BLUEGILL_LC50: Final[str] = "96 hour bluegill LC50"
FORTY_EIGHT_HR_DAPHNIA_MAGNA_LC50: Final[str] = "48 hour Daphnia magna LC50"
FORTY_EIGHT_HR_TETRAHYMENA_PYRIFORMIS_IGC50: Final[str] = "48 hour Tetrahymena pyriformis IGC50"
NINETY_SIX_HOUR_SCUD_LC50: Final[str] = "96 hour scud LC50"
NINETY_SIX_HOUR_RAINBOW_TROUT_LC50: Final[str] = "96 hour rainbow trout LC50"

ACUTE_AQUATIC_TOXICITY: Final[str] = "Acute aquatic toxicity"

ORAL_RAT_LD50: Final[str] = "Oral rat LD50"  # OPERA
ORAL_RAT_VERY_TOXIC: Final[str] = "Oral rat very toxic binary"
ORAL_RAT_NON_TOXIC: Final[str] = "Oral rat nontoxic binary"
ORAL_RAT_EPA_CATEGORY: Final[str] = "Oral rat EPA hazard category"
ORAL_RAT_GHS_CATEGORY: Final[str] = "Oral rat GHS hazard category"

FOUR_HOUR_INHALATION_RAT_LC50: Final[str] = "4 hour Inhalation rat LC50"  # OPERA

AMES_MUTAGENICITY: Final[str] = "Ames Mutagenicity"
DEVELOPMENTAL_TOXICITY: Final[str] = "Developmental toxicity"
LOCAL_LYMPH_NODE_ASSAY: Final[str] = "Local lymph node assay"

####################################################################################################################
# Units (correspond to abbreviation_ccd in qsar_datasets.units.abbreviation_ccd)
COUNT: Final[str] = "Count"
DIMENSIONLESS: Final[str] = "Dimensionless"
BINARY: Final[str] = "Binary"

G_L: Final[str] = "g/L"
MG_L: Final[str] = "mg/L"

DPM_ML: Final[str] = "dpm/mL"
BQ_ML: Final[str] = "Bq/mL"
BQ_L: Final[str] = "Bq/L"
MBQ_ML: Final[str] = "mBq/mL"
CI_MOL: Final[str] = "Ci/mol"
CI_L: Final[str] = "Ci/L"

CPM_L: Final[str] = "cpm/L"
UEQ_L: Final[str] = "ueq/L"

MG_KG: Final[str] = "mg/kg"
L_KG: Final[str] = "L/kg"
DEG_C: Final[str] = "C"
LOG_UNITS: Final[str] = "Log units"
MOLAR: Final[str] = "mol/L"

LOG_M: Final[str] = "log10(mol/L)"
NEG_LOG_M: Final[str] = "-log10(mol/L)"

NEG_LOG_MOL_KG: Final[str] = "-log10(mol/kg)"
LOG_MOL_KG: Final[str] = "log10(mol/kg)"
MOL_KG: Final[str] = "mol/kg"
UL_KG: Final[str] = "uL/kg"
LOG_L_KG: Final[str] = "log10(L/kg)"
G_CM3: Final[str] = "g/cm3"
PPM: Final[str] = "ppm"

LOG_PPM: Final[str] = "log10(ppm)"
LOG_MG_L: Final[str] = "log10(mg/L)"
LOG_MG_KG: Final[str] = "log10(mg/kg)"

POUNDS: Final[str] = "lbs"

CM3: Final[str] = "cm^3"
CM3_MOL: Final[str] = "cm^3/mol"
CUBIC_ANGSTROM: Final[str] = "Å^3"

ATM_M3_MOL: Final[str] = "atm-m3/mol"
PA_M3_MOL: Final[str] = "Pa-m3/mol"
NEG_LOG_ATM_M3_MOL: Final[str] = "-log10(atm-m3/mol)"
LOG_ATM_M3_MOL: Final[str] = "log10(atm-m3/mol)"
MMHG: Final[str] = "mmHg"
NEG_LOG_MMHG: Final[str] = "-log10(mmHg)"
LOG_MMHG: Final[str] = "log10(mmHg)"
LOG_HR: Final[str] = "log10(hr)"
LOG_DAYS: Final[str] = "log10(days)"
DAYS: Final[str] = "days"
HOUR: Final[str] = "hr"
MINUTES: Final[str] = "min"

LOG_CM3_MOLECULE_SEC: Final[str] = "log10(cm3/molecule-sec)"
CM3_MOLECULE_SEC: Final[str] = "cm3/molecule-sec"

LOG_CM_SEC: Final[str] = "log10(cm/sec)"
CM_SEC: Final[str] = "cm/sec"

UL_MIN_1MM_CELLS: Final[str] = "ul/min/10^6 cells"  # for clint
LOG_UL_MIN_1MM_CELLS: Final[str] = "log10(ul/min/10^6 cells)"  # for clint

MW_MK: Final[str] = "mW/mK"
DYN_CM: Final[str] = "dyn/cm"

LOG_CP: Final[str] = "log10(cP)"
CP: Final[str] = "cP"
CST: Final[str] = "cSt"

PCT_VOLUME: Final[str] = "%v"
PCT_WEIGHT: Final[str] = "%w"
PCT: Final[str] = "%"

TEXT: Final[str] = "Text"

G_KG_H20: Final[str] = "g/kg H2O"

####################################################################################################################
# Train/test split codes
TRAIN_SPLIT_NUM: Final[int] = 0
TEST_SPLIT_NUM: Final[int] = 1

# Splitting names
SPLITTING_RND_REPRESENTATIVE: Final[str] = "RND_REPRESENTATIVE"
SPLITTING_TRAINING_CROSS_VALIDATION: Final[str] = "TRAINING_CROSS_VALIDATION"
SPLITTING_OPERA: Final[str] = "OPERA"
SPLITTING_TEST: Final[str] = "TEST"

# Input types for DSSTox queries
INPUT_CASRN: Final[str] = "CASRN"
INPUT_OTHER_CASRN: Final[str] = "OTHER_CASRN"
INPUT_PREFERRED_NAME: Final[str] = "PREFERRED_NAME"
INPUT_SYNONYM: Final[str] = "SYNONYM"
INPUT_NAME2STRUCTURE: Final[str] = "NAME2STRUCTURE"
INPUT_MAPPED_IDENTIFIER: Final[str] = "MAPPED_IDENTIFIER"
INPUT_DTXCID: Final[str] = "DTXCID"
INPUT_DTXSID: Final[str] = "DTXSID"
INPUT_SMILES: Final[str] = "SMILES"
INPUT_INCHIKEY: Final[str] = "INCHIKEY"

# QSAR method codes for modeling web services
KNN: Final[str] = "knn"
XGB: Final[str] = "xgb"
SVM: Final[str] = "svm"
RF: Final[str] = "rf"
DNN: Final[str] = "dnn"
REG: Final[str] = "reg"
LAS: Final[str] = "las"
CONSENSUS: Final[str] = "consensus"

###################################################################################################
# Statistic names in qsar_models database
R2: Final[str] = "R2"
Q2: Final[str] = "Q2"

MAE: Final[str] = "MAE"
RMSE: Final[str] = "RMSE"
BALANCED_ACCURACY: Final[str] = "BA"
SENSITIVITY: Final[str] = "SN"
SPECIFICITY: Final[str] = "SP"
CONCORDANCE: Final[str] = "Concordance"
POS_CONCORDANCE: Final[str] = "PosConcordance"
NEG_CONCORDANCE: Final[str] = "NegConcordance"
PEARSON_RSQ: Final[str] = "PearsonRSQ"

TAG_TEST: Final[str] = "_Test"
TAG_TRAINING: Final[str] = "_Training"
TAG_CV: Final[str] = "_CV"

R2_TEST: Final[str] = R2 + TAG_TEST
Q2_TEST: Final[str] = Q2 + TAG_TEST

Q2_F3_TEST: Final[str] = "Q2_F3" + TAG_TEST
MAE_TEST: Final[str] = MAE + TAG_TEST
RMSE_TEST: Final[str] = RMSE + TAG_TEST

MAE_CV_TRAINING: Final[str] = MAE + TAG_CV + TAG_TRAINING
PEARSON_RSQ_CV_TRAINING: Final[str] = PEARSON_RSQ + TAG_CV + TAG_TRAINING
RMSE_CV_TRAINING: Final[str] = RMSE + TAG_CV + TAG_TRAINING

PEARSON_RSQ_TRAINING: Final[str] = PEARSON_RSQ + TAG_TRAINING
PEARSON_RSQ_TEST: Final[str] = PEARSON_RSQ + TAG_TEST

R2_TRAINING: Final[str] = R2 + TAG_TRAINING
MAE_TRAINING: Final[str] = MAE + TAG_TRAINING
RMSE_TRAINING: Final[str] = RMSE + TAG_TRAINING

COVERAGE: Final[str] = "Coverage"
COVERAGE_TRAINING: Final[str] = "Coverage" + TAG_TRAINING
COVERAGE_TEST: Final[str] = "Coverage" + TAG_TEST

BA_TRAINING: Final[str] = BALANCED_ACCURACY + TAG_TRAINING
SN_TRAINING: Final[str] = SENSITIVITY + TAG_TRAINING
SP_TRAINING: Final[str] = SPECIFICITY + TAG_TRAINING

BA_CV_TRAINING: Final[str] = BALANCED_ACCURACY + TAG_CV + TAG_TRAINING
SN_CV_TRAINING: Final[str] = SENSITIVITY + TAG_CV + TAG_TRAINING
SP_CV_TRAINING: Final[str] = SPECIFICITY + TAG_CV + TAG_TRAINING

BA_TEST: Final[str] = BALANCED_ACCURACY + TAG_TEST
SN_TEST: Final[str] = SENSITIVITY + TAG_TEST
SP_TEST: Final[str] = SPECIFICITY + TAG_TEST

   

    