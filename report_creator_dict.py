
"""
@author Todd Martin, 2025
This version uses ModelResults as a dict loaded from json string (cant use code completion but more robust to changes in class structure)

"""
from dominate import document
from dominate.tags import *
import logging

import json
import math
from typing import List, Dict, Any, Optional        
import traceback

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

# from indigo import Indigo
# from indigo.renderer import IndigoRenderer
from predict_constants import PredictConstants as pc
# import tempfile

urlChemicalDetails = "https://comptox.epa.gov/dashboard/chemical/details/"
imgURLCid = "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-files/image/by-dtxcid/";
model_file_api = "https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search/"  # may not be needed here

import numpy as np           
import base64, io


def createAnalogTile(analog, i, md, align):
    # used in two different report sections
        
    # analog_td = td(align=align, width="10%")
    analog_td = td(align=align)
    
    if "distance" in analog:
        analog_td += b("Distance:"), " " + format2(analog["distance"]), br()  # TODO: replace with similarity / Euclidean distance    
    else:
        analog_td += b("Neighbor:"), " " + str(i + 1), br()  # TODO: replace with similarity / Euclidean distance
        
    if md["propertyIsBinary"]:
        exp = get_formatted_value(True, analog["exp"], -1)
        analog_td += b("Measured: "), exp, br()    
    else:
        exp = get_formatted_value(False, analog["exp"], 3)
        analog_td += b("Measured: "), exp, br()

    if md["propertyIsBinary"]:
        pred = get_formatted_value(True, analog["pred"], -1)
        analog_td += b("Predicted: "), pred, br()    
    else:
        pred = get_formatted_value(False, analog["pred"], 3)
        analog_td += b("Predicted: "), pred, br()

    if "cid" in analog:
        imgTitle = "Analog Image for " + analog["name"]
        analog_td += img(src=imgURLCid + analog["cid"], border="1", alt=imgTitle, title=imgTitle, width="150", height="150"), br()

    if "sid" in analog:
        analog_td += a(analog["sid"], href=urlChemicalDetails + analog["sid"], title=analog["sid"] + ' on the Chemicals Dashboard', target="_blank")
    elif "cid" in analog:
        analog_td += span(analog["cid"])
        

def get_formatted_value(format_as_integer: bool, dvalue: float, nsig: int):
    # TODO: could move to ReportCreator but then would get confusing in terms of self and super()
    
    if dvalue is None:
        return "N/A"

    try:
        if format_as_integer:
            return format(dvalue, ".0f")

        # If dvalue is outside the range for scientific notation
        if dvalue != 0 and (abs(dvalue) < 0.01 or abs(dvalue) > 1e3):
            return format(dvalue, ".2E")

        return set_significant_digits(dvalue, nsig)

    except Exception as ex:
        import traceback
        traceback.print_exc()  # This will print the traceback to the console
        return None

    
def set_significant_digits(value, significant_digits):
    if significant_digits < 0:
        raise ValueError("Significant digits must be non-negative")

    # Set precision for decimal operations
    getcontext().prec = significant_digits

    # Convert value to Decimal and round to significant digits
                
    try:
        
        if isinstance(value, float):
            value = format(value, '.10g')  # Use a general format to avoid excessive precision

        # Set the precision context to handle larger numbers
        getcontext().prec = significant_digits + 5  # Add some buffer to the precision

        # Convert the value to a Decimal
        decimal_value = Decimal(value)

        # Quantize the Decimal to the specified number of significant digits
        quantized_value = decimal_value.quantize(Decimal('1e-{0}'.format(significant_digits - 1)), rounding=ROUND_HALF_UP)
                    
        return str(quantized_value)
    except (InvalidOperation, ValueError) as e:
        # Handle exceptions related to invalid operations or value conversion
        print(e)
        return "error:" + str(value)

       
def format2(value):
    return format(value, ".2f")
            
        
def fmt_val(v: Any, default: str="—") -> str:
    """Format a value for display; replace None/NaN/empty with a placeholder."""
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    s = str(v)
    return s if s.strip() != "" else default
            

def fmt_num(n: Optional[float], precision: int=6) -> str:
    """Pretty format a numeric value if present; else placeholder."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "—"
    # Keep integers as-is; format floats with limited precision
    if isinstance(n, (int,)) or (isinstance(n, float) and n.is_integer()):
        return str(int(n))
    return f"{round(float(n), precision)}"


class ReportCreator:
    
    class ChemicalIdentifiersSection:
        
        def create_chemical_identifiers_table(self, chemical):
                with table(border="0", width="100%"):
                    with tbody():
                        with tr(bgcolor="black"):
                            with td():
                                font("Chemical Identifiers", color="white")
                        with tr():
                            my_td = td()
                            my_td += b("Preferred name:"), " " + chemical.get("name", "N/A"), br()
                            my_td += b("DTXSID:"), " " + chemical.get("sid", "N/A"), br()
                            my_td += b("DTXCID:"), " " + chemical.get("cid", "N/A"), br()
                            my_td += b("CASRN:"), " " + chemical.get("casrn", "N/A"), br()
            
                            if "averageMass" in chemical:
                                my_td += b("Molecular weight:"), " ", "{:.2f}".format(chemical.get("averageMass")), br()
                            else:
                                my_td += b("Molecular weight:"), " N/A", br()        
    
    class ModelResultsSection:
        
        def create_model_results_table(self, report):
            
            # print(report.to_json())
            
            md = report["modelDetails"]
            
            mr = report["modelResults"]
        
            with table(border="0", width="100%"):
                with tbody():
                    with tr(bgcolor="black"):
                        with td():
                            font("Model Results", color="white")
                    with tr():
                        my_td = td()
    
                        # my_td += b("Model name:"), " " + md["modelName"], br()
                        
                        my_td += b("Model name:"), " " + md["modelName"],
                        
                        # link_qmrf = model_file_api + "?modelId="+report["modelDetails"].modelId+"&typeId=1"
                        # link_excel = model_file_api + "?modelId="+report["modelDetails"].modelId+"&typeId=2"
                        link_qmrf = md["urlQMRF"]
                        link_excel = md["urlExcelSummary"]
                        
                        my_td += " ("
                        my_td += a('QMRF', href=link_qmrf, title='Model summary in QSAR Model Reporting Format', target="_blank")
                        my_td += ", "
                        my_td += a('Excel summary', href=link_excel, title='All model details in Excel format', target="_blank")
                        my_td += ")", br()
                        
                        my_td += b("Model method:"), a(md["modelMethod"], title=md["modelMethodDescription"], href=md["modelMethodDescriptionURL"]), br()
                        my_td += b("Model source:"), " " + md["modelSource"], br()
                        
                        my_td += b("Property name:"), " " + md["propertyName"], br()
                        my_td += b("Property description:"), " " + md["propertyDescription"], br()
        
                        if mr["experimentalValueUnitsModel"]: 
                            str_exp_model_units = get_formatted_value(md["propertyIsBinary"], mr["experimentalValueUnitsModel"], 3)
                            my_td += b("Experimental value:"), " " + str_exp_model_units + " " + md["unitsModel"]
                        
                            if md["unitsDisplay"] != md["unitsModel"]:
                                str_exp_display_units = get_formatted_value(md["propertyIsBinary"], mr["experimentalValueUnitsDisplay"], 3)
                                my_td += " = " + str_exp_display_units + " " + md["unitsDisplay"]
    
                            my_td += em(" (in " + mr["experimentalValueSet"].lower() + " set)"), br()
    
                        else:
                            my_td += b("Experimental value:"), " N/A", br()
            
                        str_pred_model_units = get_formatted_value(md["propertyIsBinary"], mr["predictionValueUnitsModel"], 3)
                        
                        if mr["predictionError"]:
                            my_td += b("Predicted value:"), " N/A", br()
                            my_td += b("Prediction error:"), " "
                            my_td += b(span(mr["predictionError"], style="color:red")), br()
                            
                        else:
                            my_td += b("Predicted value:"), " " + str_pred_model_units + " " + md["unitsModel"]
                            if md["unitsDisplay"] != md["unitsModel"]:
                                str_pred_display_units = get_formatted_value(md["propertyIsBinary"], mr["predictionValueUnitsDisplay"], 3)
                                my_td += " = " + str_pred_display_units + " " + md["unitsDisplay"], br()
                            else:
                                my_td += br()
                            
                            self.addApplicabilityDomains(md, mr, my_td)
                        
        def addApplicabilityDomains(self, md, mr, my_td):
            # my_td += br()
            my_td += b("Applicability domains:")
            my_td += br()
            
            with my_td:
                with table(style="table-layout: fixed; width: 100%; border: 0; padding: 0; margin: 0; border-collapse: collapse;"):
                    with tr():
                        for ad in mr["adEstimates"]:
                            
                            with td(style="vertical-align: top; padding: 0; margin: 0; padding-left: 20px;"):
                                                                
                                method = ad["adMethod"]["name"]
                                
                                if method == pc.Applicability_Domain_TEST_Embedding_Euclidean:
                                    # print(method,"Distance") 
                                    self.addAnalogADTable(ad, md)
                                elif method == pc.TEST_FRAGMENTS:
                                    self.addFragmentADTable(ad)
                                else:
                                    print("TODO handle " + method + " in addApplicabilityDomains")
    
        def addAnalogADTable(self, ad, md):
            
            if md["applicabilityDomainName"] == pc.Applicability_Domain_TEST_Embedding_Euclidean: 
                # b("AD measure: "), span("If the avg. Euclidean distance of training set analogs  < "+format2(ad['AD_Cutoff'])+" (in terms of model variables"), br()
                b("AD measure: "), span("Avg. distance to training set analogs < cutoff value"), br()
            else:
                print("TODO in addAnalogADTable(), handle", pc.Applicability_Domain_TEST_Embedding_Euclidean)
            
            with span() as container:
                container.add(b("AD result: "))
                
                if ad["conclusion"] == "Inside": 
                    container.add(span("Inside AD", style="color: green;"))
                    container.add(" (" + ad["reasoning"] + ")")
                else:
                    container.add(span("Outside AD", style="color: red;"))                    
                    container.add(" (" + ad["reasoning"] + ")")
                            
            br(), br()
            
            # Table 1: Analogs from Training Set
            with table():
                cap = caption()
                # cap.add(strAD)
                # cap += br(), br()
                
                unitsModel = md["unitsModel"]
                
                cap += f"Training analogs (values in {unitsModel}, CV predictions)"
            
                with tbody():
                    tr()
                    analogs = ad["analogs"]
                    for i in range(len(analogs)):
                        createAnalogTile(analogs[i], i, md, "left")                                
    
        def addFragmentADTable(self, ad):
            
            b("AD measure: "), span("If the fragment counts are within the range for the training set"), br()
                        
            with span() as container:
                container.add(b("AD result: "))
                
                if ad["conclusion"] == "Inside":
                    container.add(span("Inside AD", style="color: green;"))
                    container.add(" (fragment counts were within the training set range)")
                else:
                    container.add(span("Outside AD", style="color: red;"))
                    container.add(" (fragment counts were outside the training set range)")
                        
            with table(cellspacing="0", cellpadding="5", border="1"):
                
                cap = caption()
            
                with tr(style="background-color: #d3d3d3"):
                    th("Fragment") 
                    th("Test Chemical")
                    th("Training Min")
                    th("Training Max")
                with tbody():
                    for col_name in ad["fragmentTable"]["test_chemical"].keys():
                        # Retrieve values
            
                        test_value = int(ad["fragmentTable"]["test_chemical"][col_name])
                        training_min = int(ad["fragmentTable"]["training_min"][col_name])
                        training_max = int(ad["fragmentTable"]["training_max"][col_name])
            
                        # Determine if the row should be highlighted
                        if test_value < training_min or test_value > training_max:
                            row_style = "background-color: pink;"
                        else:
                            row_style = "background-color: #E4FAE4;"
            
                        with tr(style=row_style):
                            td(col_name)
                            td(test_value, align="center")
                            td(training_min, align="center")
                            td(training_max, align="center")
            
                # Add AD status to the caption based on the flag
                ad_status_text = span()
    
                cap.add(ad_status_text)
                cap += br()
                cap += "Fragment counts for test chemical"
    
    class ModelPerformanceSection:
        
        def write_model_performance(self, md):
        
            with table(border="0", width="100%"):
                with tbody():
                    with tr(bgcolor="black"):
                        with td(colspan="3"):
                            font("Model performance", color="white")
                    with tr():
                        with td(align="center"): 
                            img(src=md["imgSrcPlotScatter"], alt="Scatter plot for " + md["modelName"], height=400)
                        with td(align="center"): 
                            img(src=md["imgSrcPlotHistogram"], alt="Histogram plot for " + md["modelName"], height=400)
                        with td(align="left"):
                            self.addStatsTableTraining(md)
                            br()
                            self.addStatsTableTest(md)
                            
                            with p():
                                span("R")
                                sup("2")
                                span(" = Pearson correlation coefficient, RMSE = root mean squared error, MAE = mean absolute error")                
    
        def addStatsTableTraining(self, md): 
            
                # ms = md["modelStatistics"]
                
                ms = md["performance"]
            
                with table(border=1, cellpadding="0", cellspacing="0", width="100%"):
                    caption("Model Training Set Statistics")
            
                    with tbody():
                        with tr():
                            # Row for "Training" and "Test" headers
                            td("Training (80%)", colspan="3", align="center", style="background-color: #d3d3d3; width: 25%;")
                            td("5-fold CV (80%)", colspan="3", align="center", style="background-color: #ccffcc; width: 25%;")
                
                        with tr():
                            
                            for _ in range(2):
                                td(["R", sup("2")], align="center")
                                td("RMSE", align="center")
                                td("MAE", align="center")
                                            
                        with tr():
                            # Training set stats
                            
                            td(format2(ms["train"]["R2"]), align="center")
                            td(format2(ms["train"]["RMSE"]), align="center")
                            td(format2(ms["train"]["MAE"]), align="center")
            
                            # CV stats
                            td(format2(ms["fiveFoldICV"]["R2"]), align="center")
                            td(format2(ms["fiveFoldICV"]["RMSE"]), align="center")
                            td(format2(ms["fiveFoldICV"]["MAE"]), align="center")
    
        def addStatsTableTest(self, md): 
        
            ms = md["performance"]
        
            with table(border=1, cellpadding="0", cellspacing="0", width="100%"):
                    caption("Model Test Set Statistics")
            
                    with tbody():
                        with tr():
                            # Row for "Training" and "Test" headers
                            td("Test (20%)", colspan="3", align="center", style="background-color: #ccccff; width: 25%;")
                            td("Test Set Applicability Domain Statistics", colspan="3", align="center",
                               style="background-color: #ffffcc; width: 25%;")
            
                        with tr():
                            # Header row for metrics
            
                            td(["R", sup("2")], align="center")
                            td("RMSE", align="center")
                            td("MAE", align="center")
            
                            td("MAE Test inside AD", align="center")
                            td("MAE Test outside AD", align="center")
                            td("Fraction Inside AD", align="center")
            
                        with tr():
            
                            td(format2(ms["external"]["R2"]), align="center")
                            td(format2(ms["external"]["RMSE"]), align="center")
                            td(format2(ms["external"]["MAE"]), align="center")
    
                            # AD stats
                            td(format2(ms["externalAD"]["MAE_inside_AD"]), align="center")
                            td(format2(ms["externalAD"]["MAE_outside_AD"]), align="center")
                            td(format2(ms["externalAD"]["Fraction_inside_AD"]), align="center")    
    
    class RawExpDataSection:
        
        def safe_text(self, v, placeholder="N/A"):
            # None
            if v is None:
                return placeholder
            # pandas or numpy NA/NaN
            try:
                import pandas as pd
                if pd.isna(v):
                    return placeholder

            except Exception:
                pass
            if isinstance(v, float) and math.isnan(v):
                return placeholder
            # Empty string
            s = str(v)
            
            return s if s.strip() else placeholder

        def addSource(self, rec, fieldName):
            source = self.safe_text(rec[fieldName + "_name"])
            
            # print(source)
            
            if source and source != "N/A":
                description = self.safe_text(rec[fieldName + "_description"])
                url = self.safe_text(rec[fieldName + "_url"])
                
                with li(cls="no-indent"):
                    if url != "N/A" and description != "N/A":
                        a(source, href=url, title=description, target="_blank")
                    elif description != "N/A":
                        span(description)
                    else:
                        span(source)

            return source

        def addParam(self, params, param_name):
            param = params[param_name]
            try:
                with li():
                    if param["value_min"] and param["value_max"]:
                        minValue = get_formatted_value(False, param["value_min"], 3)
                        maxValue = get_formatted_value(False, param["value_max"], 3)
                        print(minValue, maxValue, param["units"])
                        if param["units"]:
                            span(param_name + ": " + minValue + " < " + param_name + " " + param["units"] + " < " + maxValue + " ")
                        else:
                            span(param_name + ": " + minValue + " < " + param_name + " < " + maxValue)
                    elif param["value_text"]:
                        span(param_name + ": " + param["value_text"])
                    elif param["value_point_estimate"]:
                        if param["value_qualifier"]:
                            span(param_name + ": " + param["value_qualifier"] + " " + get_formatted_value(param["value_point_estimate"]) + " " + param["units"])
                        else:
                            span(param_name + ": " + get_formatted_value(False, param["value_point_estimate"], 3) + " " + param["units"])
                    else:
                        span(param_name + ": TODO")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        def addParams(self, rec, param_names):
            with td():
                params = rec["params"]
                if isinstance(params, dict):
                    with ul(cls="no-indent"): 
                        for param_name in param_names:
                            if param_name in params:
                                self.addParam(params, param_name)
                else:
                    br()
        
        def add_property_value_record(self, rec: Dict[str, Any], param_names) -> Any:

            keys = ("source_chemical_name", "source_casrn", "source_smiles", "source_dtxsid", "source_dtxrid")
            identifiers = [self.safe_text(rec.get(k), placeholder=None) for k in keys]
            identifiers = [v for v in identifiers if v]  
            with td():
                with ul(cls="no-indent"):
                    for identifier in identifiers:
                        li(identifier, cls="no-indent")
                        
            property_value = float(self.safe_text(rec["property_value"]))
            str_property_value = get_formatted_value(False, property_value, 3)
            td(str_property_value)
            td(self.safe_text(rec["property_units"]), cls="compact")
            
            with td():
                with ul(cls="no-indent"):
                    self.addSource(rec, "public_source")
                    self.addSource(rec, "public_source_original")
                    self.addSource(rec, "literature_source")
                    
                    direct_url = self.safe_text(rec["direct_url"])
                    if direct_url != "N/A":
                        li(a("Direct link", href=direct_url, title="Direct link"), cls="no-indent", target="_blank")
                        
                    brief_citation = self.safe_text(rec["brief_citation"])
                    
                    if brief_citation != "N/A":
                        li(span(brief_citation, cls="no-indent"))
                                    
            self.addParams(rec, param_names)        
            
            # literature_source_name=self.safe_text(rec["literature_source_name"])
            # public_source_original_name=self.safe_text(rec["public_source_original_name"])

        def create_experimental_records_table(self, records, param_names):
                # caption("Results for neighbors compared with entire set")
            with table(cls="compact"):
                with body():
                    with tr(style="background-color: #d3d3d3"):
                        th("Source Chemical", cls="compact")
                        th("Property Value", cls="compact")
                        th("Property Units", cls="compact")
                        th("Source", width="400px", cls="compact")
                        th("Parameters", width="400px", cls="compact")
                    for rec in records:
                        
                        qsar_exp_prop_property_values_id = self.safe_text(rec["qsar_exp_prop_property_values_id"])
                        qsar_exp_prop_property_values_ids = qsar_exp_prop_property_values_id.split("|")
                        property_values_id = self.safe_text(rec["property_values_id"])
                        
                        if property_values_id in qsar_exp_prop_property_values_ids:
                            with tr(style="background-color: #90EE90;"):
                                self.add_property_value_record(rec, param_names)
                        else:
                            with tr():
                                self.add_property_value_record(rec, param_names)

        def getPageStyle(self):
            return """
ul.no-indent {
  list-style: disc;
  list-style-position: outside;
  margin: 5px;
  padding-left: 15px;
}
ul.no-indent li {
  margin: 0;
}

table.compact {
  border-collapse: collapse;
  border: 1px solid #000;
}

table.compact th {
  border-top: 1px solid #000;
  border-bottom: 1px solid #000;
  text-align: center;
  padding: 4px;
}

table.compact td {
  vertical-align: top;
  padding: 4px;
  border-bottom: 1px solid #000;
}
"""

        def create_exp_records_webpage(self, records: List[Dict[str, Any]], param_names, title_text: str="Experimental Records") -> str:
            """Render a list of df_pv-like records into an HTML page and return the HTML string."""
            
            doc = document(title=title_text)
            
            with doc.head:
                style(self.getPageStyle())
        
            records2 = records.to_dict(orient="records")
            
            with doc:
                h3(title_text)
                self.create_experimental_records_table(records2, param_names)
            return doc.render()

        def writeExpData(self):
            pass
    
    class NeighborSection:
    
        def write_neighbors(self, report, neighborsInSet):
            
            nset = neighborsInSet["set"]
            
            mr = report["modelResults"]
            md = report["modelDetails"]
            chemical = report["chemicalIdentifiers"]                     
                
            with table(border="0", width="100%"):
                                        
                with tbody():
                    
                    with tr(bgcolor="black"):
                        with td(colspan="4"):
                            font(neighborsInSet["title"], color="white")
                    with tr(): 
                        
                        with td(valign="top"):
                            br(), br()                  
                            plotTitle = "Nearest neighbors from " + nset
                            plot_base64 = self.generateScatterPlot(neighborsInSet["neighbors"], md["unitsModel"], plotTitle, "Exp. vs Pred.")        
                            img(src=f'data:image/png;base64,{plot_base64}', alt='Plot of experimental vs. predicted for ' + nset + " set", height="400")
                        
                        td_tc = td(valign="top")
                        self.createTestChemicalTile(td_tc, chemical, md, mr, "center")
                        self.addMaeTable(td_tc, md, neighborsInSet)
                                                
                        with td():
                            self.addNeighborTileTable(report, neighborsInSet["neighbors"])    
                                
        def createTestChemicalTile(self, td_tc, chemical, md, mr, align):
                
                td_tc += br(), br()
                td_tc += b("Test chemical"), br()
                
                exp_float = mr["experimentalValueUnitsModel"]
                pred_float = mr["predictionValueUnitsModel"]
                
                # print(exp_float, pred_float)
                
                if md["propertyIsBinary"]:
                    exp = get_formatted_value(True, exp_float, -1)
                    td_tc += b("Measured: "), exp, br()    
                else:
                    exp = get_formatted_value(False, exp_float, 3)
                    td_tc += b("Measured: "), exp, br()
        
                if md["propertyIsBinary"]:
                    pred = get_formatted_value(True, pred_float, -1)
                    td_tc += b("Predicted: "), pred, br()    
                else:
                    pred = get_formatted_value(False, pred_float, 3)
                    td_tc += b("Predicted: "), pred, br()
                
                if chemical.get("sid", "N/A") != "N/A": 
                    title = "Image for " + chemical["name"]
                    td_tc += img(src=chemical["imageSrc"], border="1", alt=title, title=title, width="150", height="150"), br()
                    td_tc += a(chemical["chemId"], href=urlChemicalDetails + chemical["sid"], title=chemical["sid"] + ' on the Chemicals Dashboard', target="_blank")
                else:
                    title = "Image for " + chemical["smiles"]
                    td_tc += img(src=chemical["imageSrc"], border="1", alt=title, title=title, width="150", height="150"), br()
                    td_tc += chemical["chemId"]        
    
        def addMaeTable(self, td_tc, md, neighborsInSet):
                
                with td_tc:
                    p()
                    with table(border=1, cellpadding="5", cellspacing="0"):
                        caption("Results for neighbors compared with entire set")
                            
                        with tbody():
                            
                            with tr(style="background-color: #d3d3d3"):
                                th("Chemicals")
                                th("MAE*")
                            
                            with tr():
                                td("Analogs from set")
                                td(get_formatted_value(False, neighborsInSet["MAE"], 3))
            
                            with tr():
                                td("Entire set")
                                
                                if "train" in neighborsInSet["set"].lower():
                                    td(get_formatted_value(False, md["performance"]["fiveFoldICV"]["MAE"], 3))    
                                else:
                                    td(get_formatted_value(False, md["performance"]["external"]["MAE"], 3))
                
                    p('* Mean absolute error in ' + md["unitsModel"])
                    
        def addNeighborTileTable(self, report, neighbors):
                
                md = report["modelDetails"]        
        
                with table(border="0", width="100%", cellpadding=10):
                    caption("Neighbor values in " + md["unitsModel"])
                    # following is hardcoded to use top 10 analogs but could be made to only have the analogs that are similar enough
                    with tr():
                        for i in range(0, 5):
                            analog = neighbors[i]
                            createAnalogTile(analog, i, md, "left")
                    
                    with tr():
                        for i in range(5, 10):
                            
                            if i >= len(neighbors):
                                print(report["chemicalIdentifiers"]["chemId"] + " insufficient neighbors")  # TODO figure out why this happens
                            else:
                                analog = neighbors[i]
                                createAnalogTile(analog, i, md, "left")

        def generateScatterPlot(self, modelPredictions, unitName, plotTitle, seriesName):
            
            x, y = self.getArraysOmitNullPreds(modelPredictions)
            if len(x) == 0 and len(y) == 0:
                return
        
            fig, ax = plt.subplots(figsize=(5, 5), layout='constrained', dpi=150)
        
            plt.xlabel('Experimental ' + unitName)
            plt.ylabel('Predicted  ' + unitName)
            plt.title(plotTitle)
        
            ax.scatter(x, y, label=seriesName, color="red", edgecolor='black')
            ax.plot(x, x, label='Y=X', color="black")
            
            self.setScatterplotBounds(unitName, x, y, ax)
            
            plt.legend(loc="lower right")
            # plt.savefig(fileOutPNG)
            figure = plt.gcf()  # get current figure
            
            # plt.show()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()  # Close the plot to free memory
            buf.seek(0)  # Rewind the buffer
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return image_base64
    
        def getArraysOmitNullPreds(self, mps):
            exps = []
            preds = []
        
            for mp in mps:
                if 'pred' not in mp:
                    continue
                exps.append(mp['exp'])
                preds.append(mp['pred'])
        
            exps = np.array(exps)
            preds = np.array(preds)
            return exps, preds

        def setScatterplotBounds(self, unitName, x, y, ax):
            # Determine the range for the axes
            min_value = min(min(x), min(y))
            max_value = max(max(x), max(y))
        # Check if "log" is in unitName
            if "log" in unitName.lower():
                min_int = int(np.floor(min_value))
                max_int = int(np.ceil(max_value))
                # Determine if padding is needed
                if (min_value - min_int) < 0.25:
                    min_value = min_int - 1
                else:
                    min_value = min_int
                if (max_int - max_value) < 0.25:
                    max_value = max_int + 1
                else:
                    max_value = max_int
                ax.set_xticks(range(min_value, max_value + 1))
                ax.set_yticks(range(min_value, max_value + 1))
            elif unitName == "°C":
                min_value_50 = (np.floor(min_value / 50) * 50) 
                max_value_50 = (np.ceil(max_value / 50) * 50)
                
                if abs(min_value - min_value_50) < 10:
                    min_value = min_value_50 - 50
                else:
                    min_value = min_value_50
                if abs(max_value_50 - max_value) < 10:
                    max_value = max_value_50 + 50
                else:
                    max_value = max_value_50
                    
                ax.set_xticks(range(int(min_value), int(max_value), 50))
                ax.set_yticks(range(int(min_value), int(max_value), 50))
                             
            ax.set_xlim(min_value, max_value)
            ax.set_ylim(min_value, max_value)
    
                    # TODO add fragment count table as second AD measure
             
            # analog["sid"]  # Adjust the image path and attributes as needed
    
    def create_html_report_from_json(self, reportJson): 
        '''
        Method to create report from ModelResults json
        This approach loads from json into dictionary. It was not attempted to reinstantiate the original class which may change over time
                
        :param reportJson: ModelResults report as json string
        '''
        modelResults = json.loads(reportJson)
        return self.create_html_report(modelResults)
    
    def write_first_row(self, report):
        
        chemical = report["chemicalIdentifiers"]
    
        with table(style="table-layout: fixed; width: 100%; border: 0px"):
            with tbody():
                with tr():  # first row of main table
                    with td(valign="top", width="150px"):
                        
                        if chemical["imageSrc"] == "N/A":
                                div("No structure image", style="border: 2px solid black; padding: 10px;")                        
                        else:
                            title = "Structural image of " + chemical["chemId"]
                            img(src=chemical["imageSrc"], alt=title, title=title,
                                height=150, width=150, border="2")
                    
                    cis = self.ChemicalIdentifiersSection()
                    
                    with td(valign="top", width="20%"):
                        cis.create_chemical_identifiers_table(chemical)
        
                    mrs = self.ModelResultsSection()
                    
                    with td(valign="top"):
                        mrs.create_model_results_table(report)
    
    # def create_report(self, model: Model, modelResults: ModelResults):
    #     print('enter create_json_report')
    #     report = Report(model,modelResults)
    #     return report
    
    def create_html_report(self, report):
        '''
        Method to create report from ModelResults dictionary
        :param report: ModelResults dictionary
        '''
    
        try:
    
            # print(chemical)
            md = report["modelDetails"]
            
            # print("sid", chemical["sid"])
        
            page_title = md["modelName"] + " Model Calculation Details: " + md["propertyName"]
            doc = document(lang='en', title=page_title)  # title has to be set here, the title object in the head doesnt work
        
            with doc.head:
                meta(charset="UTF-8")
                meta(name="viewport", content="width=device-width, initial-scale=1.0")
        
            with doc:
                h3(page_title)
        
                with table(border="0", width="100%"):  # main table
                    with tbody():
                        with tr():
                            with td():
                                self.write_first_row(report)
                        with tr():
                            with td():
                                mps = self.ModelPerformanceSection()
                                mps.write_model_performance(md)
                        
                        nrs = self.NeighborSection()
                        
                        if report["neighborResultsPrediction"]:
                        
                            with tr():
                                with td():
                                    nrs.write_neighbors(report, report["neighborResultsPrediction"])

                        if report["neighborResultsTraining"]:
                                    
                            with tr():
                                with td():
                                    nrs.write_neighbors(report, report["neighborResultsTraining"])
    
            return str(doc)

        except:
            logging.error("Error creating html report")
            traceback.print_exc()  # Prints the full traceback
            return None


def create_report_from_json_file():
    
    import os, json, webbrowser, pathlib
    rc = ReportCreator()
    
    # file_name = model_id + "_report_todd_" + safe_smiles(smiles) + ".json"
    # file_name = "1066_DTXSID8031865.json"
    # file_name = "1066_QLWFRFCRJULPCK-UHFFFAOYNA-N.json" # has error
    # file_name = "1065_DTXSID3039242.json"  # WS_BZ
    # file_name = "1065_DTXSID8031865.json"  # WS_PFOA
    file_name = "1069_DTXSID3039242.json"
    
    current_directory = os.getcwd()
    # Join the path components using os.path.join()
    file_path = os.path.join(current_directory, "data", "reports", file_name)
    
    # Use the full path string
    print(file_path)    
    try:
        with open(file_path, 'r') as file:
            report = json.load(file)
            # print(modelResults)
            
            html = rc.create_html_report(report)
            
            # print(html)
            
            file_path_html = file_path.replace(".json", ".html")
            
            with open(file_path_html, 'w', encoding='utf-8') as f:
                f.write(html)
                
            htmlPath = pathlib.Path(file_path_html)
        
            webbrowser.open(htmlPath)
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

if __name__ == '__main__':
    create_report_from_json_file()

