'''
Created on Aug 10, 2026

@author: TMARTI02
'''

from dominate import document
from dominate.tags import *
from dominate.util import raw

from report_creator_dict import get_formatted_value


from starlette.responses import HTMLResponse, JSONResponse
from util.helpers import build_error_response
import logging
from model_service_common.config import get_env as _get_env

from API_Utilities import SearchAPI
import os
import json
from util.database_utilities import getSession

from statistics import median
from dotenv import load_dotenv
from fontTools.t1Lib import write

load_dotenv('../personal.env')
session=getSession()

PROJECT_ROOT = _get_env("PROJECT_ROOT")

if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT environment variable is not set")


def write_results_to_json(data, filename="results.json"):
    

    temp_dir = os.path.join(PROJECT_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return file_path



PROJECT_ROOT = os.getenv("PROJECT_ROOT")




def build_summary_table(data, prop_units):
    all_props = sorted(prop_units.keys())

    with div(cls="table-wrapper"):
        with table(id="propertyTable"):
            with thead():
                tr(
                    th("DTXSID"),
                    *[
                        th(
                            div(prop),
                            div(prop_units[prop], cls="unit") if prop_units.get(prop) else ""
                        )
                        for prop in all_props
                    ]
                )

            with tbody():
                for dtxsid, chem in data.items():
                    with tr():
                        td(dtxsid, cls="dtxsid")
                        props = chem.get("properties", {})
                        for prop in all_props:
                            prop_info = props.get(prop)
                            if prop_info:
                                median_value = prop_info.get("median_prop_value")
                                records = prop_info.get("records", [])

                                if median_value is None:
                                    td("")
                                else:
                                    td(
                                        get_formatted_value(
                                            prop_units.get(prop) == 'Binary 0/1',
                                            median_value,
                                            -1 if prop_units.get(prop) == 'Binary 0/1' else 3
                                        ),
                                        cls="clickable",
                                        **{
                                            "data-dtxsid": dtxsid,
                                            "data-prop": prop,
                                            "data-prop-unit": prop_units.get(prop, ""),
                                            "data-records": json.dumps(records, default=str)
                                        }
                                    )
                            else:
                                td("")


def build_selected_records_script():
    return raw("""
        document.addEventListener("DOMContentLoaded", function () {
            const panel = document.getElementById("detailPanel");
            const cells = document.querySelectorAll("#propertyTable td.clickable");

            function escapeHtml(str) {
                return String(str)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            function makeLink(text, url, tooltip) {
                if (text == null || text === "") return "";
                const safeText = escapeHtml(text);
                const safeUrl = escapeHtml(url);
                const titleAttr = tooltip ? ` title="${escapeHtml(tooltip)}"` : "";
                return `<a href="${safeUrl}" target="_blank"${titleAttr}>${safeText}</a>`;
            }

            function renderRecordsTable(records, propName, propUnit, dtxsid) {
                if (!records || records.length === 0) {
                    return "<p>No records found.</p>";
                }

                const datasetName = records[0].dataset || "";

                const columns = [
                    "prop_value_display",
                    "public_source_name",
                    "public_source_original_name",
                    "literature_source_display",
                    "experimental_parameters_display"
                ];

                const headerLabels = {
                    "experimental_parameters_display": "Experimental Parameters",
                    "public_source_name": "Public Source",
                    "public_source_original_name": "Public Source Original",
                    "literature_source_display": "Literature Source",
                    "literature_source_citation": "Literature Source Citation"
                };

                let html = `<h4 class="record-title">${escapeHtml(dtxsid)} — ${escapeHtml(propName)} (dataset = ${escapeHtml(datasetName)})</h4>`;
                html += '<div class="records-wrapper"><table class="records-table"><thead><tr>';

                columns.forEach(col => {
                    if (col === "prop_value_display") {
                        html += `<th>${escapeHtml(propName)}${propUnit ? `<div class="unit">${escapeHtml(propUnit)}</div>` : ""}</th>`;
                    } else {
                        html += `<th>${escapeHtml(headerLabels[col] || col)}</th>`;
                    }
                });

                html += '</tr></thead><tbody>';

                records.forEach(record => {
                    html += '<tr>';

                    columns.forEach(col => {
                        if (col === "prop_value_display") {
                            const val = record["prop_value_display"];
                            const directUrl = record["direct_url"];
                            if (directUrl && val != null && val !== "") {
                                html += `<td>${makeLink(val, directUrl, "")}</td>`;
                            } else {
                                html += `<td>${escapeHtml(val == null ? "" : val)}</td>`;
                            }
                            return;
                        }

                        if (col === "public_source_name") {
                            const val = record["public_source_name"] || "";
                            if (record["public_source_url"]) {
                                const sourceUrl = String(record["public_source_url"]);
                                const tooltip = record["public_source_description"] || "";
                                html += `<td>${makeLink(val, sourceUrl, tooltip)}</td>`;
                            } else {
                                html += `<td>${escapeHtml(val)}</td>`;
                            }
                            return;
                        }
                        
                        if (col === "experimental_parameters_display") {
                            const val = record["experimental_parameters_display"] || "";
                            html += `<td style="white-space: pre-line;">${escapeHtml(val)}</td>`;
                            return;
                        }

                        if (col === "public_source_original_name") {
                            const val = record["public_source_original_name"] || "";
                            if (record["public_source_original_url"]) {
                                const sourceUrl = String(record["public_source_original_url"]);
                                const tooltip = record["public_source_original_description"] || "";
                                html += `<td>${makeLink(val, sourceUrl, tooltip)}</td>`;
                            } else {
                                html += `<td>${escapeHtml(val)}</td>`;
                            }
                            return;
                        }

                        if (col === "literature_source_display") {
                            const name = record["literature_source_display"] || "";
                            const citation = record["literature_source_citation"] || "";
                            const doi = record["literature_source_doi"] || "";
                        
                            let nameHtml = "";
                            if (name) {
                                if (doi) {
                                    const rawDoi = String(doi);
                                    const doiUrl = rawDoi.startsWith("http") ? rawDoi : `https://doi.org/${rawDoi}`;
                                    nameHtml = makeLink(name, doiUrl, "");
                                } else {
                                    nameHtml = escapeHtml(name);
                                }
                            }
                        
                            if (nameHtml && citation) {
                                html += `<td>${nameHtml}: ${escapeHtml(citation)}</td>`;
                            } else if (nameHtml) {
                                html += `<td>${nameHtml}</td>`;
                            } else if (citation) {
                                html += `<td>${escapeHtml(citation)}</td>`;
                            } else {
                                html += `<td></td>`;
                            }
                            return;
                        }                        

                        const val = record[col];
                        html += `<td>${escapeHtml(val == null ? "" : val)}</td>`;
                    });

                    html += '</tr>';
                });

                html += '</tbody></table></div>';
                return html;
            }

            cells.forEach(cell => {
                cell.addEventListener("click", function () {
                    const dtxsid = this.getAttribute("data-dtxsid");
                    const prop = this.getAttribute("data-prop");
                    const propUnit = this.getAttribute("data-prop-unit") || "";
                    const records = JSON.parse(this.getAttribute("data-records") || "[]");

                    panel.innerHTML = renderRecordsTable(records, prop, propUnit, dtxsid);
                });
            });
        });
    """)


def build_html_style():
    return raw("""
        body { font-family: Arial, sans-serif; margin: 20px; }

        .table-wrapper {
            width: 100%;
            overflow-x: auto;
            overflow-y: visible;
            margin-bottom: 16px;
            border: 1px solid #ddd;
        }

        table {
            border-collapse: collapse;
            min-width: max-content;
            width: 100%;
        }

        th, td {
            border: 1px solid #ccc;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
            white-space: nowrap;
        }

        th {
            background: #f3f3f3;
            position: sticky;
            top: 0;
            z-index: 1;
        }

        .dtxsid {
            font-weight: bold;
            white-space: nowrap;
        }

        .unit {
            font-size: 0.85em;
            color: #666;
            display: block;
            margin-top: 2px;
            white-space: nowrap;
        }

        .clickable {
            cursor: pointer;
            color: #0b5ed7;
            text-decoration: underline;
        }

        #detailPanel {
            margin-top: 20px;
            padding: 12px;
            border: 1px solid #ccc;
            background: #fafafa;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }

        .records-wrapper {
            width: 100%;
            max-width: 100%;
            overflow: hidden;
            box-sizing: border-box;
        }

        .records-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            margin-top: 10px;
            font-family: Arial, sans-serif;
        }

        .records-table th, .records-table td {
            border: 1px solid #ddd;
            padding: 6px 8px;
            vertical-align: top;
            text-align: left;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }

        .records-table th {
            background: #f0f0f0;
        }

        .record-title {
            margin: 0 0 8px 0;
        }
    """)


def format_experimental_parameters(parameters):
    if not parameters:
        return ""

    lines = []
    for name, parameterValue in parameters.items():
        if name.strip().lower() == "pubchem cid":
            continue

        value_unit = (parameterValue.get("value_unit") or "").strip()
        value_text = parameterValue.get("value_text")
        value_point_estimate = parameterValue.get("value_point_estimate")
        value_min = parameterValue.get("value_min")
        value_max = parameterValue.get("value_max")

        def to_float(v):
            try:
                if v is None or v == "":
                    return None
                return float(v)
            except Exception:
                return None

        if value_unit.lower() == "text":
            display_value = str(value_text) if value_text is not None else ""
        else:
            if value_point_estimate is not None:
                num = to_float(value_point_estimate)
                formatted = get_formatted_value(False, num, 3) if num is not None else str(value_point_estimate)
                display_value = f"{formatted} {value_unit}".strip()

            elif value_min is not None and value_max is not None:
                num_min = to_float(value_min)
                num_max = to_float(value_max)
                formatted_min = get_formatted_value(False, num_min, 3) if num_min is not None else str(value_min)
                formatted_max = get_formatted_value(False, num_max, 3) if num_max is not None else str(value_max)
                display_value = f"{formatted_min} - {formatted_max} {value_unit}".strip()

            elif value_min is not None:
                num_min = to_float(value_min)
                formatted_min = get_formatted_value(False, num_min, 3) if num_min is not None else str(value_min)
                display_value = f"{formatted_min} {value_unit}".strip()

            elif value_max is not None:
                num_max = to_float(value_max)
                formatted_max = get_formatted_value(False, num_max, 3) if num_max is not None else str(value_max)
                display_value = f"{formatted_max} {value_unit}".strip()

            elif value_text is not None:
                display_value = f"{value_text} {value_unit}".strip()

            else:
                display_value = ""

        if display_value != "":
            lines.append(f"{name}: {display_value}")

    return "\n".join(lines)


def write_results_to_html(data, write_to_file=True, filename="chemical_properties.html"):
    if not PROJECT_ROOT:
        raise ValueError("PROJECT_ROOT environment variable is not set")

    temp_dir = os.path.join(PROJECT_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, filename)

    prop_units = {}
    for dtxsid, chem in data.items():
        for prop_name, prop_info in chem.get("properties", {}).items():
            prop_units[prop_name] = prop_info.get("prop_unit") or ""

    doc = document(title="Chemical Property Table")

    with doc.head:
        meta(charset="utf-8")
        style(build_html_style())

    with doc:
        h2("Chemical Properties by DTXSID")
        p("Click a property value to view the underlying records.")

        build_summary_table(data, prop_units)

        h3("Selected Records")
        div(id="detailPanel", _text="Click a value above to see its records.")

        script(build_selected_records_script())

    if write_to_file:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.render())
        return file_path
    else:
        return doc.render()



def get_exp_data_response(identifiers=None, report_format="json", properties=None):
    """
    Resolve identifiers, fetch experimental property data, and write JSON/HTML reports.
    
    :param identifiers:
    :param report_format:
    :param properties: filter to only these properties (TODO)
    """
    RESOLVER_API = _get_env(
        "resolver.url",
        "RESOLVER_API",
        default="http://resolver-api:8600/api/resolver"
    )

    sids = []
    chem_dict = {}

    if identifiers:
        for identifier in identifiers:
            try:
                chemicals, code = SearchAPI.call_resolver_get(RESOLVER_API, identifier)
                if len(chemicals) == 0:
                    continue

                sid = (chemicals[0].get("chemical") or {}).get("sid")
                sids.append(sid)
                chem_dict[sid] = chemicals[0]

            except Exception as exc:
                logging.exception("Resolver lookup failed for identifier=%s", identifier)
                return JSONResponse(
                    build_error_response(
                        identifier,
                        "resolver_error",
                        "Resolver lookup failed",
                        str(exc)
                    ),
                    status_code=500,
                )

            if code != 200 or not chemicals:
                return JSONResponse(
                    build_error_response(identifier, "not_found", f"Could not find {identifier}"),
                    status_code=404,
                )

        print(sids)

        data = fetch_property_values_grouped_by_chemical(session, sids)

        output_file = write_results_to_json(data, "chemical_properties.json")
        print(f"Wrote JSON to: {output_file}")

        html_path = write_results_to_html(data, True, "chemical_properties.html")
        print(f"Wrote HTML to: {html_path}")

        return {
            "json": output_file,
            "html": html_path,
            "data": data
        }
        
        

def get_exp_data(sids, properties=None):
    """
    Resolve identifiers, fetch experimental property data, and write JSON/HTML reports.
    
    :param sids:
    :param properties: what properties to return TODO
    """
    session = getSession()
    data = fetch_property_values_grouped_by_chemical(session, sids)
    return data


from sqlalchemy import text
from collections import defaultdict

def fetch_property_values(session, dtxsids):
    main_sql = text("""
        select
            --row_number() over (order by dpc.dtxsid, p."name", ps.name, ls.citation) as id,
            dpc.dtxsid,
            dpc.dtxcid,
            dpc.smiles,
            p.name_ccd as prop_name,
            'experimental' as prop_type,
            pc.name as prop_category,
            d."name" as dataset,
            dpc.property_value as prop_value,
            u.abbreviation_ccd as prop_unit,
            pv.id as prop_value_id,
--            pv.value_original as prop_value_original,
--            pv.value_text as prop_value_text,
--            case when ps.name is not null then ps.name else ls.name end as source_name,
--            case when ps.name is not null then ps.description else ls.citation end as source_description,
--            case when ps.name is not null then ps.url else ls.doi end as source_url,
            ps."name" as public_source_name,
            ps.description as public_source_description,
            ps.url as public_source_url,
            pv.page_url as direct_url,
            ls."name" as literature_source_name,
            ls.citation as literature_source_citation,
            ls.doi as literature_source_doi,
            pv.document_name as brief_citation,
            ps2."name" as public_source_original_name,
            ps2.description as public_source_original_description,
            ps2.url as public_source_original_url,
            --current_date as export_date,
            '2.1.1' as data_version
        from qsar_datasets.data_points dp
        join qsar_datasets.data_point_contributors dpc
            on dpc.fk_data_point_id = dp.id
        join exp_prop.property_values pv
            on dpc.exp_prop_property_values_id = pv.id
        left join exp_prop.literature_sources ls
            on pv.fk_literature_source_id = ls.id
        left join exp_prop.public_sources ps
            on pv.fk_public_source_id = ps.id
        left join exp_prop.public_sources ps2
            on pv.fk_public_source_original_id = ps2.id
        join qsar_datasets.datasets d
            on dp.fk_dataset_id = d.id
        join qsar_datasets.properties p
            on d.fk_property_id = p.id
        join qsar_datasets.datasets_in_dashboard did
            on did.fk_property_id = d.fk_property_id
        join qsar_datasets.units u
            on u.id = d.fk_unit_id_contributor
        left join qsar_datasets.properties_in_categories pic
            on p.id = pic.fk_property_id
        left join qsar_datasets.property_categories pc
            on pic.fk_property_category_id = pc.id
        where d.id = did.fk_datasets_id
          and keep = true
          and dpc.dtxsid = any(:dtxsids)
    """)

    result = session.execute(main_sql, {"dtxsids": dtxsids})
    return [dict(row._mapping) for row in result]


def fetch_parameters_by_property_value_id(session, prop_value_ids):
    if not prop_value_ids:
        return {}

    param_sql = text("""
        select
            pv.fk_property_value_id,
            u.abbreviation,
            p.id as parameter_id,
            p.name as parameter_name,
            pv.value_text,
            pv.value_point_estimate,
            pv.value_min,
            pv.value_max,
            pv.value_error,
            pv.value_qualifier
        from exp_prop.parameter_values pv
        join exp_prop.parameters p
            on pv.fk_parameter_id = p.id
        join exp_prop.units u
            on pv.fk_unit_id = u.id
        where pv.fk_property_value_id = any(:prop_value_ids)
    """)

    param_map = defaultdict(dict)

    result = session.execute(param_sql, {"prop_value_ids": prop_value_ids})
    for row in result:
        r = dict(row._mapping)
        prop_value_id = r["fk_property_value_id"]
        parameter_name = r["parameter_name"]

        param_map[prop_value_id][parameter_name] = {
            "parameter_id": r["parameter_id"],
            "value_unit": r["abbreviation"],
            "value_text": r["value_text"],
            "value_point_estimate": r["value_point_estimate"],
            "value_min": r["value_min"],
            "value_max": r["value_max"],
            "value_error": r["value_error"],
            "value_qualifier": r["value_qualifier"]
        }

    return param_map

def fetch_property_values_grouped_by_chemical(session, dtxsids):
    '''
    Gets the experimental data as dictionary
    
    :param session:
    :param dtxsids:
    '''
    
    rows = fetch_property_values(session, dtxsids)

    prop_value_ids = list({
        row["prop_value_id"] for row in rows
        if row.get("prop_value_id") is not None
    })

    param_map = fetch_parameters_by_property_value_id(session, prop_value_ids)

    chemicals = {}
    temp = defaultdict(lambda: defaultdict(list))

    for row in rows:
        row["parameters"] = param_map.get(row["prop_value_id"], {})
        
        row["experimental_parameters_display"] = format_experimental_parameters(row.get("parameters", {}))
        
        # Clean up bad direct_url which isnt a direct link to the record:
        if row.get("direct_url") == "https://ochem.eu/home/show.do":
            row["direct_url"] = None

        # Precompute display fields to keep JS simple
        row["prop_value_display"] = get_formatted_value(
            row.get("prop_unit") == "Binary 0/1",
            row.get("prop_value"),
            -1 if row.get("prop_unit") == "Binary 0/1" else 3
        )
        row["literature_source_display"] = row.get("brief_citation") or row.get("literature_source_name") or ""

        dtxsid = row["dtxsid"]
        prop_name = row["prop_name"]
        temp[dtxsid][prop_name].append(row)

    for dtxsid, props in temp.items():
        if dtxsid not in chemicals:
            first_prop_records = next(iter(props.values()))
            first_row = first_prop_records[0]
            chemicals[dtxsid] = {
                "dtxcid": first_row.get("dtxcid"),
                "smiles": first_row.get("smiles"),
                "properties": {}
            }

        for prop_name, records in props.items():
            numeric_values = [
                r["prop_value"] for r in records
                if isinstance(r.get("prop_value"), (int, float))
            ]

            med = median(numeric_values) if numeric_values else None
            prop_unit = records[0].get("prop_unit") if records else None

            chemicals[dtxsid]["properties"][prop_name] = {
                "median_prop_value": med,
                "prop_unit": prop_unit,
                "records": records
            }

    return chemicals
if __name__ == '__main__':
    

    get_exp_data_response(identifiers=["benzene","toluene"], report_format="json", properties=None)