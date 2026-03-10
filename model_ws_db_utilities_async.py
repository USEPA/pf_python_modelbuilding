import asyncio
import json
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import httpx
import pandas as pd

from API_Utilities import DescriptorsAPI, QsarSmilesAPI
from applicability_domain import applicability_domain_utilities as adu
from db.mongo_cache import get_cached_prediction, cache_prediction
from model_ws_db_utilities import ModelDetails, ModelInitializer, ModelPredictor
from model_ws_utilities import call_do_predictions_from_df
from util import predict_constants as pc


def _timing_enabled() -> bool:
    return os.getenv("PREDICT_TIMING_LOG", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(min_value, value)


def _chunked_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _split_evenly(items, n_parts):
    if not items:
        return []

    n_parts = max(1, min(n_parts, len(items)))
    base = len(items) // n_parts
    remainder = len(items) % n_parts

    chunks = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < remainder else 0)
        if size == 0:
            continue
        end = start + size
        chunks.append(items[start:end])
        start = end

    return chunks


_CPU_POOL = None
_CPU_POOL_LOCK = threading.Lock()
_CPU_POOL_SIZE = 0

_WORKER_PREDICTOR = None
_WORKER_MODEL_CACHE = {}


def _ad_failure_result(ad_name: str, err: Exception):
    return {
        "AD": False,
        "adMethod": {
            "name": ad_name,
            "description": "Applicability domain calculation failed",
        },
        "conclusion": "Outside",
        "reasoning": f"AD calculation failed: {err}",
    }


def _init_cpu_worker():
    global _WORKER_PREDICTOR, _WORKER_MODEL_CACHE
    _WORKER_PREDICTOR = ModelPredictor()
    _WORKER_MODEL_CACHE = {}


def _get_cpu_pool(max_workers: int) -> ProcessPoolExecutor:
    global _CPU_POOL, _CPU_POOL_SIZE
    if _CPU_POOL is None:
        with _CPU_POOL_LOCK:
            if _CPU_POOL is None:
                _CPU_POOL = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_cpu_worker,
                )
                _CPU_POOL_SIZE = max_workers
                logging.info("Async CPU pool created with %d workers", max_workers)
    return _CPU_POOL


def _get_worker_model_context(model_id: int):
    global _WORKER_MODEL_CACHE

    if model_id in _WORKER_MODEL_CACHE:
        return _WORKER_MODEL_CACHE[model_id]

    mi = ModelInitializer()
    model = mi.init_model(model_id)
    if not hasattr(model, "modelId"):
        raise ValueError(f"Invalid model_id: {model_id}")

    file_api = os.getenv("FILE_API_SERVER", pc.URL_CTX_API)
    predictor = _WORKER_PREDICTOR or ModelPredictor()

    model_details = ModelDetails(model)
    predictor.addLinks(model_details, file_api)
    predictor.addPerformance(model_details)

    ctx = {
        "model": model,
        "predictor": predictor,
        "model_details": model_details,
    }
    _WORKER_MODEL_CACHE[model_id] = ctx
    return ctx


def _cpu_predict_chunk(args):
    model_id, chunk_items, generate_neighbors, skip_images = args

    ctx = _get_worker_model_context(model_id)
    model = ctx["model"]
    predictor = ctx["predictor"]
    model_details = ctx["model_details"]

    outputs = []
    valid_items = []
    valid_dfs = []

    for item in chunk_items:
        idx = item["idx"]
        smiles = item["smiles"]
        chemical = item.get("chemical")

        if chemical is None:
            outputs.append((idx, predictor._build_error_report_dict(smiles, model_details, item.get("std_error"))))
            continue

        if not skip_images:
            predictor._set_chemical_image(chemical)

        descriptor_payload = item.get("descriptor")
        if descriptor_payload is None:
            err = item.get("descriptor_error") or "Descriptor calculation failed"
            outputs.append((idx, predictor._build_error_report_dict(smiles, model_details, err, chemical=chemical)))
            continue

        df_pred = pd.DataFrame(descriptor_payload["rows"], columns=descriptor_payload["columns"])
        valid_items.append(item)
        valid_dfs.append(df_pred)

    if valid_items:
        all_pred_dfs = pd.concat(valid_dfs, ignore_index=True)

        batch_json = call_do_predictions_from_df(all_pred_dfs, model)
        batch_pred_results = json.loads(batch_json)

        pred_values_by_pos = [row["pred"] for row in batch_pred_results]

        ad_results_by_pos = [[] for _ in range(len(valid_items))]
        if model.applicabilityDomainName:
            applicability_domains = model_details.applicabilityDomainName.split(" and ")
            if pc.Applicability_Domain_TEST_Fragment_Counts not in applicability_domains:
                applicability_domains.append(pc.Applicability_Domain_TEST_Fragment_Counts)

            for ad_name in applicability_domains:
                if ad_name == pc.Applicability_Domain_TEST_Fragment_Counts:
                    for i, df_pred in enumerate(valid_dfs):
                        try:
                            ad_result = predictor.determineApplicabilityDomain(model, ad_name, df_pred)
                        except Exception as exc:
                            logging.exception("Per-row AD failed for %s", ad_name)
                            ad_result = _ad_failure_result(ad_name, exc)
                        ad_results_by_pos[i].append(ad_result)
                else:
                    try:
                        output, ad_cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                            train_df=model.df_training,
                            test_df=all_pred_dfs.copy(),
                            remove_log_p=model.remove_log_p_descriptors,
                            embedding=model.embedding,
                            applicability_domain=ad_name,
                            filterColumnsInBothSets=True,
                        )
                        for i in range(len(valid_items)):
                            ad_result = predictor._parse_ad_result_for_row(output, i, ad_cutoff, ad_name, model)
                            ad_results_by_pos[i].append(ad_result)
                    except Exception:
                        logging.exception("Batch AD failed for %s; fallback to per-row", ad_name)
                        for i, df_pred in enumerate(valid_dfs):
                            try:
                                ad_result = predictor.determineApplicabilityDomain(model, ad_name, df_pred)
                            except Exception as exc:
                                logging.exception("Per-row AD fallback failed for %s", ad_name)
                                ad_result = _ad_failure_result(ad_name, exc)
                            ad_results_by_pos[i].append(ad_result)

        precomputed_neighbors_by_pos = [None for _ in range(len(valid_items))]
        if generate_neighbors:
            try:
                precomputed = predictor._precompute_neighbors_for_batch(model, all_pred_dfs)
                for i in range(len(valid_items)):
                    precomputed_neighbors_by_pos[i] = {
                        "training": precomputed["training"][i],
                        "prediction": precomputed["prediction"][i],
                    }
            except Exception:
                logging.exception("Batch neighbor precompute failed; fallback per-row")

        for i, item in enumerate(valid_items):
            report = predictor._build_report_from_precomputed(
                model_id=model_id,
                smiles=item["smiles"],
                model=model,
                modelDetails=model_details,
                chemical=item["chemical"],
                pred_value=pred_values_by_pos[i],
                ad_estimates=ad_results_by_pos[i],
                df_prediction=valid_dfs[i],
                generate_neighbors=generate_neighbors,
                precomputed_neighbors=precomputed_neighbors_by_pos[i],
            )
            outputs.append((item["idx"], report))

    return outputs


class AsyncModelPredictor:

    def __init__(self):
        self._descriptor_api = DescriptorsAPI()
        self._io_model_cache = {}

    async def _get_io_model_config(self, model_id):
        if model_id in self._io_model_cache:
            return self._io_model_cache[model_id]

        cfg = await asyncio.to_thread(self._load_io_model_config_sync, model_id)
        self._io_model_cache[model_id] = cfg
        return cfg

    @staticmethod
    def _load_io_model_config_sync(model_id):
        mi = ModelInitializer()
        model = mi.init_model(model_id)
        if not hasattr(model, "modelId"):
            return None

        return {
            "modelId": model.modelId,
            "qsarReadyRuleSet": model.qsarReadyRuleSet,
            "omitSalts": model.omitSalts,
            "descriptorService": model.descriptorService,
            "descriptorHeaders": list(model.df_training.columns[2:]),
        }

    async def predict_from_db(self, model_id, smiles):
        if isinstance(smiles, str):
            key = f"{smiles}-{model_id}"
            prediction = get_cached_prediction(key)
            if prediction is not None:
                return prediction

            results = await self.predict_model_smiles_batch(model_id, [smiles])
            prediction = results[0] if results else {}
            cache_prediction(key, prediction)
            return prediction

        smiles_list = list(smiles)
        if not smiles_list:
            return []

        result = [None] * len(smiles_list)
        missing = []

        for idx, smi in enumerate(smiles_list):
            key = f"{smi}-{model_id}"
            prediction = get_cached_prediction(key)
            if prediction is not None:
                result[idx] = prediction
            else:
                missing.append((idx, smi))

        if missing:
            missing_smiles = [smi for _, smi in missing]
            predictions = await self.predict_model_smiles_batch(model_id, missing_smiles)

            for (idx, smi), prediction in zip(missing, predictions):
                cache_prediction(f"{smi}-{model_id}", prediction)
                result[idx] = prediction

        return result

    async def predict_model_smiles_batch(self, model_id, smiles_list):
        if not smiles_list:
            return []

        timing_on = _timing_enabled()
        total_start = perf_counter() if timing_on else None

        server_apis = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
        serverapis_timeout = _env_int("PREDICT_SERVERAPIS_TIMEOUT_SEC", 120)

        standardized, std_errors = await self._standardize_all_async(
            server_host=server_apis,
            model_id=model_id,
            smiles_list=smiles_list,
            timeout_sec=serverapis_timeout,
        )

        descriptors_by_idx, descriptor_errors = await self._descriptors_all_async(
            server_host=server_apis,
            model_id=model_id,
            standardized=standardized,
            timeout_sec=serverapis_timeout,
        )

        generate_neighbors = _env_flag("PREDICT_GENERATE_NEIGHBORS", True)
        skip_images = _env_flag("PREDICT_SKIP_IMAGES", False)

        payload_items = []
        for idx, smiles in enumerate(smiles_list):
            chemical = standardized[idx]
            descriptor_df = descriptors_by_idx[idx]

            descriptor_payload = None
            if descriptor_df is not None:
                descriptor_payload = {
                    "columns": list(descriptor_df.columns),
                    "rows": descriptor_df.to_dict(orient="records"),
                }

            payload_items.append({
                "idx": idx,
                "smiles": smiles,
                "chemical": chemical,
                "std_error": std_errors[idx],
                "descriptor": descriptor_payload,
                "descriptor_error": descriptor_errors[idx],
            })

        cpu_workers = _env_int("PREDICT_CPU_WORKERS", os.cpu_count() or 1)
        chunks = _split_evenly(payload_items, cpu_workers)

        loop = asyncio.get_running_loop()
        pool = _get_cpu_pool(cpu_workers)

        cpu_tasks = [
            loop.run_in_executor(
                pool,
                _cpu_predict_chunk,
                (model_id, chunk, generate_neighbors, skip_images),
            )
            for chunk in chunks
        ]

        chunk_results = await asyncio.gather(*cpu_tasks)

        ordered = [None] * len(smiles_list)
        for chunk_result in chunk_results:
            for idx, report in chunk_result:
                ordered[idx] = report

        if timing_on:
            total_elapsed = perf_counter() - total_start if total_start is not None else 0.0
            logging.info(
                "timing.async_batch model_id=%s size=%d cpu_workers=%d chunks=%d total=%.3fs",
                model_id,
                len(smiles_list),
                cpu_workers,
                len(chunks),
                total_elapsed,
            )

        return ordered

    async def _standardize_all_async(self, server_host, model_id, smiles_list, timeout_sec):
        model_cfg = await self._get_io_model_config(model_id)

        standardized = [None] * len(smiles_list)
        errors = [None] * len(smiles_list)

        if model_cfg is None:
            for i, smi in enumerate(smiles_list):
                errors[i] = f"Invalid model_id: {model_id}"
            return standardized, errors

        qsar_rule_set = model_cfg["qsarReadyRuleSet"]
        omit_salts = model_cfg["omitSalts"]

        if server_host == "https://hcd.rtpnc.epa.gov/" and qsar_rule_set == "qsar-ready_04242025_0":
            qsar_rule_set = "qsar-ready_04242025"

        batch_size = _env_int("PREDICT_STANDARDIZE_BATCH_SIZE", 250)
        workers = _env_int("PREDICT_STANDARDIZE_WORKERS", 4)

        tasks = list(_chunked_list(list(enumerate(smiles_list)), batch_size))
        timeout = httpx.Timeout(timeout_sec)
        limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
        semaphore = asyncio.Semaphore(workers)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def run_task(task):
                async with semaphore:
                    smiles_chunk = [smi for _, smi in task]
                    chemicals, status_code = await QsarSmilesAPI.call_qsar_ready_standardize_post_async(
                        client=client,
                        server_host=server_host,
                        smiles=smiles_chunk,
                        full=False,
                        workflow=qsar_rule_set,
                    )
                    return task, chemicals, status_code

            results = await asyncio.gather(*(run_task(task) for task in tasks), return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logging.exception("Async standardization chunk failed", exc_info=result)
                continue

            task, chemicals, status_code = result
            idxs = [idx for idx, _ in task]
            smiles_chunk = [smi for _, smi in task]

            if status_code != 200 or not isinstance(chemicals, list) or len(chemicals) != len(task):
                for idx, smi in zip(idxs, smiles_chunk):
                    errors[idx] = f"{smi} failed standardization"
                continue

            for idx, smi, item in zip(idxs, smiles_chunk, chemicals):
                chemical, code = self._parse_batch_standardized_item(item, smi, omit_salts)
                if code == 200:
                    standardized[idx] = chemical
                else:
                    errors[idx] = chemical

        return standardized, errors

    async def _descriptors_all_async(self, server_host, model_id, standardized, timeout_sec):
        model_cfg = await self._get_io_model_config(model_id)

        dfs_by_idx = [None] * len(standardized)
        errors = [None] * len(standardized)

        if model_cfg is None:
            for i, smi in enumerate(standardized):
                if smi is not None:
                    errors[i] = f"Invalid model_id: {model_id}"
            return dfs_by_idx, errors

        valid_pairs = [(idx, item["canonicalSmiles"]) for idx, item in enumerate(standardized) if item is not None]
        if not valid_pairs:
            return dfs_by_idx, errors

        batch_size = _env_int("PREDICT_DESCRIPTORS_BATCH_SIZE", 250)
        workers = _env_int("PREDICT_DESCRIPTORS_WORKERS", 4)

        tasks = list(_chunked_list(valid_pairs, batch_size))
        timeout = httpx.Timeout(timeout_sec)
        limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
        semaphore = asyncio.Semaphore(workers)

        descriptor_headers = model_cfg["descriptorHeaders"]
        descriptor_service = model_cfg["descriptorService"]

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def run_task(task):
                async with semaphore:
                    smiles_chunk = [smi for _, smi in task]
                    payload, status_code = await self._descriptor_api.call_descriptors_post_with_status_async(
                        client=client,
                        server_host=server_host,
                        qsar_smiles=smiles_chunk,
                        descriptor_name=descriptor_service,
                    )
                    return task, payload, status_code

            results = await asyncio.gather(*(run_task(task) for task in tasks), return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logging.exception("Async descriptor chunk failed", exc_info=result)
                continue

            task, payload, status_code = result
            idxs = [idx for idx, _ in task]
            smiles_chunk = [smi for _, smi in task]

            if status_code != 200 or not isinstance(payload, dict):
                for idx in idxs:
                    errors[idx] = "Descriptor calculation failed"
                continue

            dfs = self._descriptor_api.response_json_to_dfs(
                payload,
                smiles_chunk,
                descriptor_headers=descriptor_headers,
            )
            if dfs is None or len(dfs) != len(task):
                for idx in idxs:
                    errors[idx] = "Descriptor calculation failed"
                continue

            for idx, df in zip(idxs, dfs):
                if df is None:
                    errors[idx] = "Descriptor calculation failed"
                else:
                    dfs_by_idx[idx] = df

        return dfs_by_idx, errors

    def _parse_batch_standardized_item(self, item, smiles, omit_salts):
        if isinstance(item, list):
            if len(item) == 0:
                return (f"{smiles} failed standardization" if smiles else "No Structure"), 400
            if len(item) > 1 and omit_salts:
                return f"{smiles}: model can't run mixtures", 400
            item = item[0]

        if isinstance(item, dict):
            if "chemicals" in item and isinstance(item["chemicals"], list):
                return self._parse_batch_standardized_item(item["chemicals"], smiles, omit_salts)
            if "canonicalSmiles" in item:
                return item, 200
            if "error" in item:
                return item["error"], 400

        return (f"{smiles} failed standardization" if smiles else "No Structure"), 400
