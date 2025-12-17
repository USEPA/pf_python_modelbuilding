
import math

class PredictConstants:
    """Numerical constants for processing and modeling"""
    
    # TODO: move to separate script
    
    TEST_FRAGMENTS = "TEST Fragments"
        
    SOURCE_CHEMINFORMATICS_MODULES = "Cheminformatics Modules"

    # Fraction agreement required to map a DSSTox conflict
    CONFLICT_FRAC_AGREE = 1.0
    # Fraction agreement required to merge binary data points
    BINARY_FRAC_AGREE = 0.8
    # Cutoff for binary classification
    BINARY_CUTOFF = 0.5
    
    # Multiple of dataset stdev required to exclude a property value based on its stdev
    STDEV_WIDTH_TOLERANCE = 3.0
    # Range tolerance values
    LOG_RANGE_TOLERANCE = 1.0
    TEMP_RANGE_TOLERANCE = 10.0

    DENSITY_RANGE_TOLERANCE = 0.1
    
    ZERO_TOLERANCE = 1e-6
        
    # Ports for other web services
    PORT_STANDARDIZER_OPERA = 5001
    PORT_TEST_DESCRIPTORS = 5002
    PORT_JAVA_MODEL_BUILDING = 5003
    PORT_PYTHON_MODEL_BUILDING = 5004
    PORT_OUTLIER_DETECTION = 5006
    PORT_REPRESENTATIVE_SPLIT = 5005
    PORT_STANDARDIZER_JAVA = 5010
    
    # DSSTox mapping strategies
    MAPPING_BY_CASRN = "CASRN"
    MAPPING_BY_DTXCID = "DTXCID"
    MAPPING_BY_DTXSID = "DTXSID"
    MAPPING_BY_LIST = "LIST"
    
    # Standardizer types
    QSAR_READY = "QSAR_READY"
    MS_READY = "MS_READY"
    
    # Standardizers
    STANDARDIZER_NONE = "NONE"  # Default QSAR-ready SMILES from DSSTox
    STANDARDIZER_OPERA = "OPERA"
    STANDARDIZER_SCI_DATA_EXPERTS = "SCI_DATA_EXPERTS"
    
    # Descriptor set names
    DESCRIPTOR_SET_TEST = "T.E.S.T. 5.1"
    DESCRIPTOR_SET_WEBTEST = "WebTEST-default"
    DESCRIPTOR_SET_MORDRED = "Mordred-default"
    DESCRIPTOR_SET_PADEL_SINGLE = "Padelpy webservice single"
    DESCRIPTOR_SET_PADEL_BATCH = "Padelpy_batch"


    ####################################################################################################################
    # Following are used in python webservice for AD calculations
    # TODO should be AD constants be here or in AD class?

    Applicability_Domain_TEST_Embedding_Cosine = "TEST Cosine Similarity Embedding Descriptors"  # matches ad_method_name in db
    Applicability_Domain_TEST_Embedding_Euclidean = "TEST Euclidean Distance Embedding Descriptors"  # matches ad_method_name in db
    Applicability_Domain_TEST_All_Descriptors_Cosine = "TEST Cosine Similarity All Descriptors"  # matches ad_method_name in db
    Applicability_Domain_TEST_All_Descriptors_Euclidean = "TEST Euclidean Distance All Descriptors"  # matches ad_method_name in db
    
    Applicability_Domain_OPERA_local_index = "OPERA Local Index"
    Applicability_Domain_OPERA_global_index = "OPERA Global Index"
    
    Applicability_Domain_OPERA_local_index_description = "Local applicability domain index is relative to the similarity of the query chemical to its five nearest neighbors from the training set"
    Applicability_Domain_OPERA_global_index_description = "Global applicability domain via the leverage approach"
    
    Applicability_Domain_OPERA_confidence_level = "OPERA Confidence Level"
    Applicability_Domain_Kernel_Density = "Kernel Density"
    
    Applicability_Domain_Combined = "Combined Applicability Domain"
    
    ####################################################################################################################
    # Property names in terms of display on the dashboard (correspond to name_ccd in qsar_datasets.properties table)
    WATER_SOLUBILITY = "Water Solubility"  # OPERA
    HENRYS_LAW_CONSTANT = "Henry's Law Constant"  # OPERA
    MELTING_POINT = "Melting Point"  # OPERA
    LOG_KOW = "LogKow: Octanol-Water"  # OPERA
    LOG_KOA = "LogKoa: Octanol-Air"  # TODO add log() to it?
    VAPOR_PRESSURE = "Vapor Pressure"  # OPERA
    
    DENSITY = "Density"
    VAPOR_DENSITY = "Vapor Density"
    BOILING_POINT = "Boiling Point"  # OPERA
    FLASH_POINT = "Flash Point"
    VISCOSITY = "Viscosity"
    SURFACE_TENSION = "Surface Tension"
    THERMAL_CONDUCTIVITY = "Thermal Conductivity"
    
    MOLAR_REFRACTIVITY = "Molar Refractivity"
    MOLAR_VOLUME = "Molar Volume"
    POLARIZABILITY = "Polarizability"
    PARACHOR = "Parachor"
    INDEX_OF_REFRACTION = "Index of Refraction"
    DIELECTRIC_CONSTANT = "Dielectric Constant"
    
    AUTOIGNITION_TEMPERATURE = "Autoignition Temperature"
    
    # APPEARANCE = "Appearance"        
    # ESTROGEN_RECEPTOR_RBA = "Estrogen receptor relative binding affinity"
    # ESTROGEN_RECEPTOR_BINDING = "Estrogen receptor binding"  # OPERA
    # ESTROGEN_RECEPTOR_AGONIST = "Estrogen receptor agonist"  # OPERA
    # ESTROGEN_RECEPTOR_ANTAGONIST = "Estrogen receptor antagonist"  # OPERA    
    # ANDROGEN_RECEPTOR_AGONIST = "Androgen receptor agonist"  # OPERA
    # ANDROGEN_RECEPTOR_ANTAGONIST = "Androgen receptor antagonist"  # OPERA
    # ANDROGEN_RECEPTOR_BINDING = "Androgen receptor binding"  # OPERA
    # TTR_BINDING = "Binding to TTR (replacement of ANSA)"  # OPERA
    
    PKA = "pKa"
    LOG_BCF = "LogBCF"  # OLD OPERA
    
    BCF = "Bioconcentration Factor"  # OPERA
    BAF = "Bioaccumulation Factor"
    
    ULTIMATE_BIODEG = "Ultimate biodegradation timeframe"
    PRIMARY_BIODEG = "Primary biodegradation timeframe"
    
    BIODEG_ANAEROBIC = "Biodegradability (anaerobic)"
    BIODEG = "Biodegradability"
    
    LOG_OH = "LogOH"  # OLD OPERA
    OH = "Atmospheric hydroxylation rate"  # OPERA
    
    LOG_KOC = "LogKOC"  # OLD OPERA
    KOC = "Soil Adsorption Coefficient (Koc)"  # OPERA
    
    LOG_HALF_LIFE = "LogHalfLife"  # OLD OPERA
    BIODEG_HL_HC = "Biodegradation half-life for hydrocarbons"  # OPERA
    
    LOG_KM_HL = "LogKmHL"  # OLD OPERA
    KmHL = "Fish biotransformation half-life (Km)"  # OPERA
    
    LOG_BCF_FISH_WHOLEBODY = "LogBCF_Fish_WholeBody"  # should just be Fish whole body bioconcentration factor
    
    # Additional OPERA properties:
    
    RBIODEG = "Ready biodegradability"  # OPERA (binary)
    FUB = "Fraction unbound in human plasma"  # OPERA
    RT = "Liquid chromatography retention time"  # OPERA
    CLINT = "Human hepatic intrinsic clearance"  # OPERA
    CACO2 = "Caco-2 permeability (Papp)"  # OPERA
    LogD_pH_7_4 = "LogD at pH=7.4"  # OPERA
    LogD_pH_5_5 = "LogD at pH=5.5"  # OPERA
    
    #    PKA_A = "Strongest acidic acid dissociation constant"#OPERA
    #    PKA_B = "Strongest basic acid dissociation constant"#OPERA
    
    PKA_A = "Acidic pKa"
    PKA_B = "Basic pKa"
    
    # Old versions for building sample models
    MUTAGENICITY = "Mutagenicity"
    LC50 = "LC50"
    LC50DM = "LC50DM"
    IGC50 = "IGC50"
    LD50 = "LD50"
    LLNA = "LLNA"
    DEV_TOX = "DevTox"
    
    SKIN_IRRITATION = "Skin irritation"  # TODO This needs to be more specific
    EYE_IRRITATION = "Eye irritation"  # TODO This needs to be more specific
    EYE_CORROSION = "Eye corrosion"  # TODO This needs to be more specific
    
    # New versions for dashboard
    NINETY_SIX_HOUR_FATHEAD_MINNOW_LC50 = "96 hour fathead minnow LC50"
    NINETY_SIX_HOUR_FISH_LC50 = "96 hour fish LC50"
    NINETY_SIX_HOUR_BLUEGILL_LC50 = "96 hour bluegill LC50"
    FORTY_EIGHT_HR_DAPHNIA_MAGNA_LC50 = "48 hour Daphnia magna LC50"
    FORTY_EIGHT_HR_TETRAHYMENA_PYRIFORMIS_IGC50 = "48 hour Tetrahymena pyriformis IGC50"
    NINETY_SIX_HOUR_SCUD_LC50 = "96 hour scud LC50"
    NINETY_SIX_HOUR_RAINBOW_TROUT_LC50 = "96 hour rainbow trout LC50"
    
    ACUTE_AQUATIC_TOXICITY = "Acute aquatic toxicity"
    
    ORAL_RAT_LD50 = "Oral rat LD50"  # OPERA
    ORAL_RAT_VERY_TOXIC = "Oral rat very toxic binary"
    ORAL_RAT_NON_TOXIC = "Oral rat nontoxic binary"
    ORAL_RAT_EPA_CATEGORY = "Oral rat EPA hazard category"
    ORAL_RAT_GHS_CATEGORY = "Oral rat GHS hazard category"
    
    FOUR_HOUR_INHALATION_RAT_LC50 = "4 hour Inhalation rat LC50"  # OPERA
    
    AMES_MUTAGENICITY = "Ames Mutagenicity"
    DEVELOPMENTAL_TOXICITY = "Developmental toxicity"
    LOCAL_LYMPH_NODE_ASSAY = "Local lymph node assay"
    
    ####################################################################################################################
    # Unit name values need to correspond to abbreviation_ccd values in qsar_datasets.units.abbreviation_ccd field
    
    COUNT = "Count"
    DIMENSIONLESS = "Dimensionless"
    BINARY = "Binary"
    
    G_L = "g/L"
    MG_L = "mg/L"
    
    DPM_ML = "dpm/mL"
    BQ_ML = "Bq/mL"
    BQ_L = "Bq/L"
    MBQ_ML = "mBq/mL"
    CI_MOL = "Ci/mol"
    CI_L = "Ci/L"
    
    CPM_L = "cpm/L"
    UEQ_L = "ueq/L"
    
    MG_KG = "mg/kg"
    L_KG = "L/kg"
    DEG_C = "C"
    LOG_UNITS = "Log units"
    MOLAR = "mol/L"
    
    LOG_M = "log10(mol/L)"    
    NEG_LOG_M = "-log10(mol/L)"
    
    NEG_LOG_MOL_KG = "-log10(mol/kg)"
    LOG_MOL_KG = "log10(mol/kg)"
    MOL_KG = "mol/kg"
    UL_KG = "uL/kg"
    LOG_L_KG = "log10(L/kg)"
    G_CM3 = "g/cm3"
    PPM = "ppm"
    
    LOG_PPM = "log10(ppm)"
    LOG_MG_L = "log10(mg/L)"
    LOG_MG_KG = "log10(mg/kg)"
            
    POUNDS = "lbs"
    
    CM3 = "cm^3"
    CM3_MOL = "cm^3/mol"
    CUBIC_ANGSTROM = "Å^3"
        
    ATM_M3_MOL = "atm-m3/mol"
    PA_M3_MOL = "Pa-m3/mol"
    NEG_LOG_ATM_M3_MOL = "-log10(atm-m3/mol)"
    LOG_ATM_M3_MOL = "log10(atm-m3/mol)"
    MMHG = "mmHg"
    NEG_LOG_MMHG = "-log10(mmHg)"
    LOG_MMHG = "log10(mmHg)"
    LOG_HR = "log10(hr)"
    LOG_DAYS = "log10(days)"
    DAYS = "days"
    HOUR = "hr"
    MINUTES = "min"
    
    LOG_CM3_MOLECULE_SEC = "log10(cm3/molecule-sec)"
    CM3_MOLECULE_SEC = "cm3/molecule-sec"
    
    LOG_CM_SEC = "log10(cm/sec)"
    CM_SEC = "cm/sec"
    
    UL_MIN_1MM_CELLS = "ul/min/10^6 cells"  # for clint
    LOG_UL_MIN_1MM_CELLS = "log10(ul/min/10^6 cells)"  # for clint
    
    MW_MK = "mW/mK"
    DYN_CM = "dyn/cm"
    
    LOG_CP = "log10(cP)"
    CP = "cP"
    CST = "cSt"
    
    PCT_VOLUME = "%v"
    PCT_WEIGHT = "%w"
    PCT = "%"
    
    TEXT = "Text"
    
    G_KG_H20 = "g/kg H2O"
    
    ####################################################################################################################

    
    
    # Integer codes for train/test splitting
    TRAIN_SPLIT_NUM = 0
    TEST_SPLIT_NUM = 1
    
    # Splitting names
    SPLITTING_RND_REPRESENTATIVE = "RND_REPRESENTATIVE"
    SPLITTING_TRAINING_CROSS_VALIDATION = "TRAINING_CROSS_VALIDATION"
    SPLITTING_OPERA = "OPERA"
    SPLITTING_TEST = "TEST"
    
    # Input types for DSSTox queries
    INPUT_CASRN = "CASRN"
    INPUT_OTHER_CASRN = "OTHER_CASRN"
    INPUT_PREFERRED_NAME = "PREFERRED_NAME"
    INPUT_SYNONYM = "SYNONYM"
    INPUT_NAME2STRUCTURE = "NAME2STRUCTURE"
    INPUT_MAPPED_IDENTIFIER = "MAPPED_IDENTIFIER"
    INPUT_DTXCID = "DTXCID"
    INPUT_DTXSID = "DTXSID"
    INPUT_SMILES = "SMILES"
    INPUT_INCHIKEY = "INCHIKEY"
    
    # QSAR method codes for modeling web services
    KNN = "knn"
    XGB = "xgb"
    SVM = "svm"
    RF = "rf"
    DNN = "dnn"
    REG = "reg"
    LAS = "las"
    CONSENSUS = "consensus"
    
    # Statistic names in qsar_models database
    
    R2 = "R2"
    Q2 = "Q2"
    
    MAE = "MAE"
    RMSE = "RMSE"
    BALANCED_ACCURACY = "BA"
    SENSITIVITY = "SN"
    SPECIFICITY = "SP"
    CONCORDANCE = "Concordance"
    POS_CONCORDANCE = "PosConcordance"
    NEG_CONCORDANCE = "NegConcordance"
    PEARSON_RSQ = "PearsonRSQ"
    
    TAG_TEST = "_Test"
    TAG_TRAINING = "_Training"
    TAG_CV = "_CV"
    
    R2_TEST = R2 + TAG_TEST
    Q2_TEST = Q2 + TAG_TEST
    
    Q2_F3_TEST = "Q2_F3" + TAG_TEST
    MAE_TEST = MAE + TAG_TEST
    RMSE_TEST = RMSE + TAG_TEST
    
    MAE_CV_TRAINING = MAE + TAG_CV + TAG_TRAINING
    PEARSON_RSQ_CV_TRAINING = PEARSON_RSQ + TAG_CV + TAG_TRAINING
    RMSE_CV_TRAINING = RMSE + TAG_CV + TAG_TRAINING
    
    PEARSON_RSQ_TRAINING = PEARSON_RSQ + TAG_TRAINING
    PEARSON_RSQ_TEST = PEARSON_RSQ + TAG_TEST
    
    R2_TRAINING = R2 + TAG_TRAINING
    MAE_TRAINING = MAE + TAG_TRAINING
    RMSE_TRAINING = RMSE + TAG_TRAINING
    
    COVERAGE = "Coverage"
    COVERAGE_TRAINING = "Coverage" + TAG_TRAINING
    COVERAGE_TEST = "Coverage" + TAG_TEST
    
    BA_TRAINING = BALANCED_ACCURACY + TAG_TRAINING
    SN_TRAINING = SENSITIVITY + TAG_TRAINING
    SP_TRAINING = SPECIFICITY + TAG_TRAINING
    
    BA_CV_TRAINING = BALANCED_ACCURACY + TAG_CV + TAG_TRAINING
    SN_CV_TRAINING = SENSITIVITY + TAG_CV + TAG_TRAINING
    SP_CV_TRAINING = SPECIFICITY + TAG_CV + TAG_TRAINING
    
    BA_TEST = BALANCED_ACCURACY + TAG_TEST
    SN_TEST = SENSITIVITY + TAG_TEST
    SP_TEST = SPECIFICITY + TAG_TEST



class UnitsConverter:
        
    def get_error_message(self, property_name, unit_name, final_unit_name, chemical_id, ):
        print(chemical_id+ ": undefined conversion for "+property_name+ " for "+unit_name+" to "+final_unit_name)
    
    def handle_surface_tension(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_CP and unit_name == PredictConstants.CP:
            return math.log10(value)
        elif final_unit_name == PredictConstants.CP and unit_name == PredictConstants.LOG_CP:
            return math.pow(10, value)
        else:
            return None

    def handle_oral_rat_ld50(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight):
        if final_unit_name == PredictConstants.MOL_KG and unit_name == PredictConstants.NEG_LOG_MOL_KG:
            return math.pow(10, -value)
        else:
            return None

    def handle_inhalation_lc50(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.MG_L and unit_name == PredictConstants.LOG_MG_L:
            return math.pow(10, value)
        elif final_unit_name == PredictConstants.PPM and unit_name == PredictConstants.LOG_PPM:
            return math.pow(10, value)
        else:
            return None

    def handle_ld50(self, property_name, value, unit_name, final_unit_name, chemical_id, dsstox_record):
        if final_unit_name == PredictConstants.NEG_LOG_MOL_KG:
            if unit_name == PredictConstants.LOG_MOL_KG:
                return -value
            elif unit_name == PredictConstants.MOL_KG and value != 0:
                return -math.log10(value)
            elif unit_name == PredictConstants.MG_KG:
                if dsstox_record.mol_weight is not None:
                    return -math.log10(value / 1000.0 / dsstox_record.mol_weight)
                elif value == 0:
                    print(f"{chemical_id}: value=0 for {dsstox_record.dsstox_substance_id}, so can't convert to {final_unit_name}")
                elif dsstox_record.mol_weight is None:
                    print(f"{chemical_id}: missing MW for {dsstox_record.dsstox_substance_id}, so can't convert to {final_unit_name}")
            elif unit_name == PredictConstants.UL_KG:
                return None
        elif final_unit_name == PredictConstants.MOL_KG and unit_name == PredictConstants.NEG_LOG_MOL_KG:
            return math.pow(10, -value)
        else:
            return None

    def handle_clint(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_UL_MIN_1MM_CELLS and unit_name == PredictConstants.UL_MIN_1MM_CELLS:
            return math.log10(value)
        elif final_unit_name == PredictConstants.UL_MIN_1MM_CELLS and unit_name == PredictConstants.LOG_UL_MIN_1MM_CELLS:
            return math.pow(10, value)
        else:
            return None

    def handle_binary(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.BINARY and unit_name == PredictConstants.BINARY :
            return value
        else:
            return None

    def handle_dimensionless(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.DIMENSIONLESS and unit_name ==PredictConstants. DIMENSIONLESS :
            return value
        else:
            return None

    def handle_oh(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_CM3_MOLECULE_SEC and unit_name == PredictConstants.CM3_MOLECULE_SEC:
            return math.log10(value)
        elif final_unit_name == PredictConstants.CM3_MOLECULE_SEC and unit_name == PredictConstants.LOG_CM3_MOLECULE_SEC:
            return math.pow(10, value)
        else:
            return None

    def handle_koc(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_L_KG and unit_name == PredictConstants.L_KG:
            return math.log10(value)
        elif final_unit_name == PredictConstants.L_KG and unit_name == PredictConstants.LOG_L_KG:
            return math.pow(10, value)
        else:
            return None

    def handle_kmhl(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_DAYS and unit_name == PredictConstants.DAYS:
            return math.log10(value)
        elif final_unit_name == PredictConstants.DAYS and unit_name == PredictConstants.LOG_DAYS:
            return math.pow(10, value)
        else:
            return None

    def handle_vapor_pressure(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_MMHG and unit_name == PredictConstants.MMHG :
            return math.log10(value)
        elif final_unit_name == PredictConstants.MMHG and unit_name == PredictConstants.LOG_MMHG :
            return math.pow(10, value)
        else:
            return None

    def handle_viscosity(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == PredictConstants.LOG_CP and unit_name == PredictConstants.CP :
            return math.log10(value)
        elif final_unit_name == PredictConstants.CP and unit_name == PredictConstants.LOG_CP :
            return math.pow(10, value)
        else:
            return None

    def handle_henrys_law_constant(self, property_name, value, unit_name, final_unit_name, chemical_id):
        
        if final_unit_name == PredictConstants.NEG_LOG_ATM_M3_MOL and unit_name == PredictConstants.ATM_M3_MOL :
            return -math.log10(value)
        elif final_unit_name == PredictConstants.ATM_M3_MOL and unit_name == PredictConstants.NEG_LOG_ATM_M3_MOL :
            return math.pow(10, -value)
        else:
            return None

    def handle_water_solubility(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight):
        if final_unit_name == PredictConstants.NEG_LOG_M:
            if unit_name == PredictConstants.LOG_M:
                return -value
            elif unit_name == PredictConstants.MOLAR and value != 0:
                return -math.log10(value)
            elif unit_name == PredictConstants.G_L:
                if molecular_weight is not None:
                    return -math.log10(value / molecular_weight)
                elif value == 0:
                    print(f"{chemical_id}: value=0 for {chemical_id}, so can't convert to {final_unit_name}")
                    return NaN
                elif molecular_weight is None:
                    print(f"{chemical_id}: missing MW for {chemical_id}, so can't convert to {final_unit_name}")
                    return NaN
        elif final_unit_name == PredictConstants.MOLAR and unit_name == PredictConstants.NEG_LOG_M:
            return math.pow(10, -value)
        else:
            return None
    
    def convert_units(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight=None):
    
        if unit_name == final_unit_name:
            return value
    
        converted_value = None
    
        if property_name in [
            PredictConstants.WATER_SOLUBILITY,
            PredictConstants.ACUTE_AQUATIC_TOXICITY,
            PredictConstants.NINETY_SIX_HOUR_FATHEAD_MINNOW_LC50,
            PredictConstants.NINETY_SIX_HOUR_SCUD_LC50,
            PredictConstants.NINETY_SIX_HOUR_RAINBOW_TROUT_LC50,
            PredictConstants.NINETY_SIX_HOUR_BLUEGILL_LC50,
            PredictConstants.FORTY_EIGHT_HR_TETRAHYMENA_PYRIFORMIS_IGC50,
            PredictConstants.FORTY_EIGHT_HR_DAPHNIA_MAGNA_LC50
        ]:
            converted_value = self.handle_water_solubility(property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight)
    
        elif property_name == PredictConstants.ORAL_RAT_LD50:
            converted_value = self.handle_oral_rat_ld50(property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight)
    
        elif property_name == PredictConstants.HENRYS_LAW_CONSTANT:
            converted_value = self.handle_henrys_law_constant(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name in [PredictConstants.KmHL, PredictConstants.BIODEG_HL_HC]:
            converted_value = self.handle_kmhl(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.VAPOR_PRESSURE:
            converted_value = self.handle_vapor_pressure(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.SURFACE_TENSION:
            converted_value = self.handle_surface_tension(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.VISCOSITY:
            converted_value = self.handle_viscosity(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name in [PredictConstants.KOC, PredictConstants.BCF, PredictConstants.BAF]:
            converted_value = self.handle_koc(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.OH:
            converted_value = self.handle_oh(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.CLINT:
            converted_value = self.handle_clint(property_name, value, unit_name, final_unit_name, chemical_id)
    
        # elif property_name in [PredictConstants.FUB, PredictConstants.TTR_BINDING]:
        #     converted_value = self.handle_dimensionless(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.RBIODEG:
            converted_value = self.handle_binary(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == PredictConstants.FOUR_HOUR_INHALATION_RAT_LC50:
            converted_value = self.handle_inhalation_lc50(property_name, value, unit_name, final_unit_name, chemical_id)
    
        if converted_value is None:
            self.get_error_message(property_name=property_name, unit_name=unit_name, final_unit_name=final_unit_name, chemical_id=chemical_id)
        
        return converted_value

