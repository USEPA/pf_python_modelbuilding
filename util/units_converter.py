'''
Created on Feb 10, 2026

@author: TMARTI02
'''

import math
from util import predict_constants as pc

class UnitsConverter:
        
    def get_error_message(self, property_name, unit_name, final_unit_name, chemical_id, ):
        print(chemical_id+ ": undefined conversion for "+property_name+ " for "+unit_name+" to "+final_unit_name)
    
    def handle_surface_tension(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_CP and unit_name == pc.CP:
            return math.log10(value)
        elif final_unit_name == pc.CP and unit_name == pc.LOG_CP:
            return math.pow(10, value)
        else:
            return None

    def handle_oral_rat_ld50(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight):
        if final_unit_name == pc.MOL_KG and unit_name == pc.NEG_LOG_MOL_KG:
            return math.pow(10, -value)
        else:
            return None

    def handle_inhalation_lc50(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.MG_L and unit_name == pc.LOG_MG_L:
            return math.pow(10, value)
        elif final_unit_name == pc.PPM and unit_name == pc.LOG_PPM:
            return math.pow(10, value)
        else:
            return None

    def handle_ld50(self, property_name, value, unit_name, final_unit_name, chemical_id, dsstox_record):
        if final_unit_name == pc.NEG_LOG_MOL_KG:
            if unit_name == pc.LOG_MOL_KG:
                return -value
            elif unit_name == pc.MOL_KG and value != 0:
                return -math.log10(value)
            elif unit_name == pc.MG_KG:
                if dsstox_record.mol_weight is not None:
                    return -math.log10(value / 1000.0 / dsstox_record.mol_weight)
                elif value == 0:
                    print(f"{chemical_id}: value=0 for {dsstox_record.dsstox_substance_id}, so can't convert to {final_unit_name}")
                elif dsstox_record.mol_weight is None:
                    print(f"{chemical_id}: missing MW for {dsstox_record.dsstox_substance_id}, so can't convert to {final_unit_name}")
            elif unit_name == pc.UL_KG:
                return None
        elif final_unit_name == pc.MOL_KG and unit_name == pc.NEG_LOG_MOL_KG:
            return math.pow(10, -value)
        else:
            return None

    def handle_clint(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_UL_MIN_1MM_CELLS and unit_name == pc.UL_MIN_1MM_CELLS:
            return math.log10(value)
        elif final_unit_name == pc.UL_MIN_1MM_CELLS and unit_name == pc.LOG_UL_MIN_1MM_CELLS:
            return math.pow(10, value)
        else:
            return None

    def handle_binary(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.BINARY and unit_name == pc.BINARY :
            return value
        else:
            return None

    def handle_dimensionless(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.DIMENSIONLESS and unit_name ==pc. DIMENSIONLESS :
            return value
        else:
            return None

    def handle_oh(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_CM3_MOLECULE_SEC and unit_name == pc.CM3_MOLECULE_SEC:
            return math.log10(value)
        elif final_unit_name == pc.CM3_MOLECULE_SEC and unit_name == pc.LOG_CM3_MOLECULE_SEC:
            return math.pow(10, value)
        else:
            return None

    def handle_koc(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_L_KG and unit_name == pc.L_KG:
            return math.log10(value)
        elif final_unit_name == pc.L_KG and unit_name == pc.LOG_L_KG:
            return math.pow(10, value)
        else:
            return None

    def handle_kmhl(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_DAYS and unit_name == pc.DAYS:
            return math.log10(value)
        elif final_unit_name == pc.DAYS and unit_name == pc.LOG_DAYS:
            return math.pow(10, value)
        else:
            return None

    def handle_vapor_pressure(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_MMHG and unit_name == pc.MMHG :
            return math.log10(value)
        elif final_unit_name == pc.MMHG and unit_name == pc.LOG_MMHG :
            return math.pow(10, value)
        else:
            return None

    def handle_viscosity(self, property_name, value, unit_name, final_unit_name, chemical_id):
        if final_unit_name == pc.LOG_CP and unit_name == pc.CP :
            return math.log10(value)
        elif final_unit_name == pc.CP and unit_name == pc.LOG_CP :
            return math.pow(10, value)
        else:
            return None

    def handle_henrys_law_constant(self, property_name, value, unit_name, final_unit_name, chemical_id):
        
        if final_unit_name == pc.NEG_LOG_ATM_M3_MOL and unit_name == pc.ATM_M3_MOL :
            return -math.log10(value)
        elif final_unit_name == pc.ATM_M3_MOL and unit_name == pc.NEG_LOG_ATM_M3_MOL :
            return math.pow(10, -value)
        else:
            return None

    def handle_water_solubility(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight):
        if final_unit_name == pc.NEG_LOG_M:
            if unit_name == pc.LOG_M:
                return -value
            elif unit_name == pc.MOLAR and value != 0:
                return -math.log10(value)
            elif unit_name == pc.G_L:
                if molecular_weight is not None:
                    return -math.log10(value / molecular_weight)
                elif value == 0:
                    print(f"{chemical_id}: value=0 for {chemical_id}, so can't convert to {final_unit_name}")
                    return NaN
                elif molecular_weight is None:
                    print(f"{chemical_id}: missing MW for {chemical_id}, so can't convert to {final_unit_name}")
                    return NaN
        elif final_unit_name == pc.MOLAR and unit_name == pc.NEG_LOG_M:
            return math.pow(10, -value)
        else:
            return None
    
    def convert_units(self, property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight=None):
    
        if unit_name == final_unit_name:
            return value
    
        converted_value = None
    
        if property_name in [
            pc.WATER_SOLUBILITY,
            pc.ACUTE_AQUATIC_TOXICITY,
            pc.NINETY_SIX_HOUR_FATHEAD_MINNOW_LC50,
            pc.NINETY_SIX_HOUR_SCUD_LC50,
            pc.NINETY_SIX_HOUR_RAINBOW_TROUT_LC50,
            pc.NINETY_SIX_HOUR_BLUEGILL_LC50,
            pc.FORTY_EIGHT_HR_TETRAHYMENA_PYRIFORMIS_IGC50,
            pc.FORTY_EIGHT_HR_DAPHNIA_MAGNA_LC50
        ]:
            converted_value = self.handle_water_solubility(property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight)
    
        elif property_name == pc.ORAL_RAT_LD50:
            converted_value = self.handle_oral_rat_ld50(property_name, value, unit_name, final_unit_name, chemical_id, molecular_weight)
    
        elif property_name == pc.HENRYS_LAW_CONSTANT:
            converted_value = self.handle_henrys_law_constant(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name in [pc.KmHL, pc.BIODEG_HL_HC]:
            converted_value = self.handle_kmhl(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.VAPOR_PRESSURE:
            converted_value = self.handle_vapor_pressure(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.SURFACE_TENSION:
            converted_value = self.handle_surface_tension(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.VISCOSITY:
            converted_value = self.handle_viscosity(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name in [pc.KOC, pc.BCF, pc.BAF]:
            converted_value = self.handle_koc(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.OH:
            converted_value = self.handle_oh(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.CLINT:
            converted_value = self.handle_clint(property_name, value, unit_name, final_unit_name, chemical_id)
    
        # elif property_name in [pc.FUB, pc.TTR_BINDING]:
        #     converted_value = self.handle_dimensionless(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.RBIODEG:
            converted_value = self.handle_binary(property_name, value, unit_name, final_unit_name, chemical_id)
    
        elif property_name == pc.FOUR_HOUR_INHALATION_RAT_LC50:
            converted_value = self.handle_inhalation_lc50(property_name, value, unit_name, final_unit_name, chemical_id)
    
        if converted_value is None:
            self.get_error_message(property_name=property_name, unit_name=unit_name, final_unit_name=final_unit_name, chemical_id=chemical_id)
        
        return converted_value



if __name__ == '__main__':
    pass