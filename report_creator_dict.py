
"""
This version uses ModelResults as a dict loaded from json string

"""
from dominate import document
from dominate.tags import meta, title, h3, table, tbody, tfoot, tr, td, th, img, font, b, br, caption, sup, a, p, span, em

import json
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

from indigo import Indigo
from indigo.renderer import IndigoRenderer
from predict_constants import PredictConstants as pc
# import tempfile


# from fastjsonschema import indent
# import json

urlChemicalDetails = "https://comptox.epa.gov/dashboard/chemical/details/"
imgURLCid = "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-files/image/by-dtxcid/";
model_file_api = "https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search/"
import numpy as np           
import base64, io


class ReportCreator:

    def addFragmentADTable(self, ad, mr, my_td):
                
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
                        row_style = ""
        
                    with tr(style=row_style):
                        td(col_name)
                        td(test_value, align="center")
                        td(training_min, align="center")
                        td(training_max, align="center")
        
            # Add AD status to the caption based on the flag
            ad_status_text = span()
        
            if ad["AD"]:
                ad_status_text.add(span("Inside AD", style="color: green;"))
                ad_status_text += ": fragment count(s) were within the training set range"
            else:
                ad_status_text.add(span("Outside AD", style="color: red;"))
                ad_status_text += ": fragment count(s) were outside the training set range"

            cap.add(ad_status_text)
            cap += br()
            cap += "Fragment counts for test chemical"
        


    def addAnalogADTable(self, ad, mr, my_td):
    
        strAD = None
        
        if ad["AD"]:  # TODO make this clearer
            ad_status = span("Inside AD", style="color: green;")
            strAD = span()
            strAD.add(ad_status)
            strAD += ": average distance to the training set analogs < cutoff distance"
        else:
            ad_status = span("Outside AD", style="color: red;")
            strAD = span()
            strAD.add(ad_status)
            strAD += ": average distance to the training set analogs > cutoff distance"
    
        # Table 1: Analogs from Training Set
        with table():
            cap = caption()
            cap.add(strAD)
            cap += br()
            
            unitsModel=mr["unitsModel"]
            
            cap += f"Analogs from Training Set (values in {unitsModel}, CV predictions)"
        
            with tbody():
                tr()
                analogs = ad["analogs"]
                for i in range(len(analogs)):
                    self.createAnalogTile(analogs[i], i, mr, "left")
                
                

    def addApplicabilityDomains(self, mr, my_td):
        my_td += br()
        my_td += b("Applicability domain:")
        my_td += br()
        
        with my_td:
            with table(style="table-layout: fixed; width: 100%; border: 0px; cellpadding = 10px"):
                with tr():
                    for ad in mr["applicabilityDomains"]:
                        with td(style="vertical-align: top;"):
                            if ad["method"] == pc.Applicability_Domain_TEST_Embedding_Euclidean:
                                self.addAnalogADTable(ad, mr, my_td)
                            elif ad["method"] == pc.TEST_FRAGMENTS:
                                self.addFragmentADTable(ad, mr, my_td)


        
        
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

    def generateScatterPlot(self, modelPredictions, unitName, plotTitle, seriesName):
        
        x, y = self.getArraysOmitNullPreds(modelPredictions)
        if len(x) == 0 and len(y) == 0:
            return
    
        fig, ax = plt.subplots(figsize=(5, 5), layout='constrained',dpi=150)
    
        plt.xlabel('Experimental ' + unitName)
        plt.ylabel('Predicted  '+ unitName)
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
    
                    

    def create_model_results_table(self, mr):
        
        # print(mr.to_json())
        
        md = mr["modelDetails"]
    
        with table(border="0", width="100%"):
            with tbody():
                with tr(bgcolor="black"):
                    with td():
                        font("Chemical Identifiers", color="white")
                with tr():
                    my_td = td()

                    # my_td += b("Model name:"), " " + md["modelName"], br()
                    
                    my_td += b("Model name:"), " " + md["modelName"], 
                    
                    # link_qmrf = model_file_api + "?modelId="+mr["modelDetails"].modelId+"&typeId=1"
                    # link_excel = model_file_api + "?modelId="+mr["modelDetails"].modelId+"&typeId=2"
                    link_qmrf = md["urlQMRF"]
                    link_excel = md["urlExcelSummary"]
                    
                    
                    my_td +=" ("
                    my_td+=a('QMRF', href=link_qmrf, title='Model summary in QSAR Model Reporting Format', target="_blank")
                    my_td+=", "
                    my_td+=a('Excel summary', href=link_excel, title='All model details in Excel format', target="_blank")
                    my_td +=")",br()

                    
                    my_td += b("Model source:"), " " + md["modelSource"], br()
                                        
                    
                    my_td += b("Property name:"), " " + md["propertyName"], br()
                    my_td += b("Property description:"), " " + md["propertyDescription"], br()
    
                    if mr["experimentalValueUnitsModel"]:
                        
                        str_exp_model_units = self.get_formatted_value(md["is_binary"], mr["experimentalValueUnitsModel"], 3)
                        my_td += b("Experimental value:"), " " + str_exp_model_units + " " + mr["unitsModel"]
                    
                        if mr["unitsDisplay"] != mr["unitsModel"]:
                            str_exp_display_units = self.get_formatted_value(md["is_binary"], mr["experimentalValueUnitsDisplay"], 3)
                            my_td += " = " + str_exp_display_units + " " + mr["unitsDisplay"]

                        my_td += em(" (in "+mr["experimentalValueSet"].lower()+" set)"), br()


                    else:
                        my_td += b("Experimental value:"), " N/A", br()
        
                    str_pred_model_units = self.get_formatted_value(md["is_binary"], mr["predictionValueUnitsModel"], 3)
                    my_td += b("Predicted value:"), " " + str_pred_model_units + " " + mr["unitsModel"]

                    if mr["unitsDisplay"] != mr["unitsModel"]:
                        str_pred_display_units = self.get_formatted_value(md["is_binary"], mr["predictionValueUnitsDisplay"], 3)
                        my_td += " = " + str_pred_display_units + " " + mr["unitsDisplay"], br()
                    
                    self.addApplicabilityDomains(mr, my_td)
                    
                    # TODO add fragment count table as second AD measure
    
    def get_formatted_value(self, format_as_integer: bool, dvalue: float, nsig: int):

        if dvalue is None:
            return "N/A"
    
        try:
            if format_as_integer:
                return format(dvalue, ".0f")
    
            # If dvalue is outside the range for scientific notation
            if dvalue != 0 and (abs(dvalue) < 0.01 or abs(dvalue) > 1e3):
                return format(dvalue, ".2E")
    
            return self.set_significant_digits(dvalue, nsig)
    
        except Exception as ex:
            import traceback
            traceback.print_exc()  # This will print the traceback to the console
            return None
    
    def set_significant_digits(self, value, significant_digits):
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
            return "error:"+str(value)
    
    
    def createTestChemicalTile(self, td_tc, mr, align):
        
        # print(set, i)
        td_tc += br(), br()
        chemical = mr["chemical"]
        td_tc += b("Test chemical"), br()
        
        md = mr["modelDetails"]
        
        exp_float = mr["experimentalValueUnitsModel"]
        pred_float = mr["predictionValueUnitsModel"]
        
        # print(exp_float, pred_float)
        
        
        if md["is_binary"]:
            exp = self.get_formatted_value(True, exp_float, -1)
            td_tc += b("Measured: "),exp, br()    
        else:
            exp = self.get_formatted_value(False, exp_float, 3)
            td_tc += b("Measured: "), exp, br()
        

        if md["is_binary"]:
            pred = self.get_formatted_value(True, pred_float, -1)
            td_tc += b("Predicted: "), pred, br()    
        else:
            pred = self.get_formatted_value(False, pred_float, 3)
            td_tc += b("Predicted: "), pred, br()

        
        if chemical.get("sid","N/A") != "N/A":        
            td_tc += img(src=chemical["imageSrc"], border="1", alt="Image for " + chemical["name"], width="150", height="150"), br()
            td_tc += a(chemical["sid"], href=urlChemicalDetails+chemical["sid"], title=chemical["sid"]+' on the Chemicals Dashboard', target="_blank")
        else:
            td_tc += img(src=chemical["imageSrc"], border="1", alt="Image for " + chemical["smiles"], width="150", height="150"), br()
            td_tc += chemical["smiles"]
            

    def createAnalogTile(self, analog, i, mr,align):
        
        # print(set, i)
        
        analog_td = td(align=align, width="10%")
                
        analog_td+= b("Neighbor:"), " " + str(i+1), br()
        
        md = mr["modelDetails"]
        
        if md["is_binary"]:
            exp = self.get_formatted_value(True, analog["exp"], -1)
            analog_td+= b("Measured: "),exp, br()    
        else:
            exp = self.get_formatted_value(False, analog["exp"], 3)
            analog_td+= b("Measured: "), exp, br()
        

        if md["is_binary"]:
            pred = self.get_formatted_value(True, analog["pred"], -1)
            analog_td+= b("Predicted: "), pred, br()    
        else:
            pred = self.get_formatted_value(False, analog["pred"], 3)
            analog_td+= b("Predicted: "), pred, br()

        
        analog_td+= img(src=imgURLCid + analog["cid"], border="1", alt="Analog Image for " + analog["name"], width="150", height="150"), br()
        
        analog_td+= a(analog["sid"], href=urlChemicalDetails+analog["sid"], title=analog["sid"]+' on the Chemicals Dashboard', target="_blank")
            
            # analog["sid"]  # Adjust the image path and attributes as needed


    def addNeighborTileTable(self, mr, neighbors):

        with table(border="0", width="100%", cellpadding=10):
            caption("Neighbor values in " + mr["unitsModel"])
            # following is hardcoded to use top 10 analogs but could be made to only have the analogs that are similar enough
            with tr():
                for i in range(0, 5):
                    analog = neighbors[i]
                    self.createAnalogTile(analog, i, mr, "left")
            
            with tr():
                for i in range(5, 10):
                    analog = neighbors[i]
                    self.createAnalogTile(analog, i, mr, "left")

    def write_neighbors(self, mr, neighborsInSet):
        
        
        with table(border="0", width="100%"):
                                    
            with tbody():
                
                with tr(bgcolor="black"):
                    with td(colspan="4"):
                        set = neighborsInSet["set"] 
                        if set == "Test":
                            font("Nearest Neighbors from " + set+" (External Predictions)", color="white")
                        else:
                            font("Nearest Neighbors from " + set+" (Cross Validation Predictions)", color="white")    
                        
                with tr(): 
                    
                    with td(valign="top"):
                        br(), br()                  
                        plotTitle= "Nearest neighbors from "+set
                        plot_base64 = self.generateScatterPlot(neighborsInSet["neighbors"], mr["unitsModel"], plotTitle, "Exp. vs Pred.")        
                        img(src=f'data:image/png;base64,{plot_base64}', alt='Plot of experimental vs. predicted for '+set+" set", height="400")
                    
                    td_tc = td(valign="top")
                    self.createTestChemicalTile(td_tc, mr, "center")
                    self.addMaeTable( td_tc, mr, neighborsInSet)
                                            
                    with td():
                        self.addNeighborTileTable(mr, neighborsInSet["neighbors"])

    
    def create_html_report_from_json(self, reportJson): 
        '''
        Method to create report from ModelResults json
        This approach loads from json into dictionary. It was not attempted to reinstantiate the original class which may change over time
                
        :param reportJson: ModelResults report as json string
        '''
        modelResults=json.loads(reportJson)
        
        
        
        return self.create_html_report(modelResults)
        
    
    def create_html_report(self, mr):
        '''
        Method to create report from ModelResults dictionary
        :param mr: ModelResults dictionary
        '''
    
        # print(chemical)
        md = mr["modelDetails"]
        
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
                            self.write_first_row(mr)
                    with tr():
                        with td():
                            self.write_model_performance(md)

                    
                    for neighborsInSet in mr["neighborsForSets"]:

                        with tr():
                            with td():
                                self.write_neighbors(mr, neighborsInSet)

    
                    # with tr():
                    #     td("Cell 3")
                    #     td("Cell 4")
    
        # with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        #     f.write(str(doc))
        #     temp_file_path = f.name
        # webbrowser.open(f'file://{temp_file_path}')
    
        # print(doc)
        
        return str(doc)
    
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
    
    def addMaeTable(self, td_tc, mr, neighborsInSet):
        
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
                        td(self.get_formatted_value(False,neighborsInSet["MAE"], 3))
    
                    with tr():
                        td("Entire set")
                        
                        if "train" in neighborsInSet["set"].lower():
                            td(self.get_formatted_value(False,mr["modelDetails"]["modelStatistics"]["MAE_CV_Training"],3))    
                        else:
                            td(self.get_formatted_value(False,mr["modelDetails"]["modelStatistics"]["MAE_Test"], 3))
                        
        
            p('* Mean absolute error in '+mr["modelDetails"]["unitsModel"]) 
    
    def addStatsTableTraining(self, md):    
    
        ms=md["modelStatistics"]
    
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
                    td(self.format2(ms["PearsonRSQ_Training"]), align="center")
                    td(self.format2(ms["RMSE_Training"]), align="center")
                    td(self.format2(ms["MAE_Training"]), align="center")
    
                    # CV stats
                    td(self.format2(ms["PearsonRSQ_CV_Training"]), align="center")
                    td(self.format2(ms["RMSE_CV_Training"]), align="center")
                    td(self.format2(ms["MAE_CV_Training"]), align="center")
    
    
        
    
    def addStatsTableTest(self, md):    
    
        ms=md["modelStatistics"]
    
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
        
                        td(self.format2(ms["PearsonRSQ_Test"]), align="center")
                        td(self.format2(ms["RMSE_Test"]), align="center")
                        td(self.format2(ms["MAE_Test"]), align="center")

                        # AD stats
                        td(self.format2(ms["MAE_Test_inside_AD"]), align="center")
                        td(self.format2(ms["MAE_Test_outside_AD"]), align="center")
                        td(self.format2(ms["Coverage_Test"]), align="center")
    
    
    # MAE_Test_inside_AD	MAE_Test_outside_AD	Coverage_Test
    
    #     TODO add a footer to the table that describes the metrics
    
    
    def smiles_to_base64(self, smiles_string):
        
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        
        mol = indigo.loadMolecule(smiles_string)
        # 1. Set the output format (this sets the context to 2D molecule rendering implicitly in most cases)
        indigo.setOption("render-output-format", "png") 
        
        # 2. (Optional but Recommended) Set a default image size
        # The renderer often needs dimensions defined.
        indigo.setOption("render-image-width", 400)
        indigo.setOption("render-image-height", 400)
        
        # Use renderToBuffer() to get the image data as bytes
        img_bytes = renderer.renderToBuffer(mol)
        
        # Encode the bytes to a base64 string
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        # print(base64_string)
        
        return base64_string
    
    def write_first_row(self, modelResults):
    
        chemical = modelResults["chemical"]
    
        with table(style="table-layout: fixed; width: 100%; border: 0px"):
            with tbody():
                with tr():  # first row of main table
                    with td(valign="top", width="150px"):
                        img(src=chemical["imageSrc"], alt="Structural image of " + chemical["name"], 
                            height=150, width=150, border="2")
    
                    with td(valign="top", width="20%"):
                        self.create_chemical_identifiers_table(chemical)
    
                    with td(valign="top"):
                        self.create_model_results_table(modelResults)
    
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
    
    # def create_report(self, model: Model, modelResults: ModelResults):
    #     print('enter create_json_report')
    #     report = Report(model,modelResults)
    #     return report
    
    def format2(self, value):
        return format(value, ".2f")

# may not be needed since chemical comes from standardizer API
# class Chemical:
#     def __init__(self):
#         chemID = None  # can be sid, inchiKey etc
#         sid = None
#         cid = None
#         casrn = None
#         name = None
#         smiles = None
#         canonicalSmiles = None
#         inchi = None
#         inchiKey = None
#         mol = None
#         molFormula = None
#         averageMass = None
#         monisotopicMass = None


def create_report_from_json():
    
    import os, json, webbrowser, pathlib
    rc=ReportCreator()
    
    
    model_id = "1065"
    smiles = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"     
    
    # file_name = model_id + "_report_todd_" + smiles + ".json"
    file_name = "bob.json"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_dir, "data","reports", file_name)
    file_path = os.path.join("data/reports/"+ file_name)
    
    try:
        with open(file_path, 'r') as file:
            modelResults = json.load(file)
            # print(modelResults)
            
            html = rc.create_html_report(modelResults)
            
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
    create_report_from_json()

