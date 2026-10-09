import ast,pathlib,unittest,types,json
from unittest.mock import Mock
from starlette.responses import JSONResponse
from db.mongo_hosts import normalize_mongo_hosts
ROOT=pathlib.Path(__file__).resolve().parents[1]
# Isolate the HTTP/initialization boundary from optional scientific runtime dependencies.
_error_node = next(n for n in ast.parse((ROOT/'db/model_cache.py').read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ModelArtifactCacheUnavailableError')
exec(compile(ast.Module(body=[_error_node], type_ignores=[]), 'model_cache', 'exec'))
def load_function(file,name,env):
 tree=ast.parse((ROOT/file).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),env);return env[name]
class DependencyResponses(unittest.TestCase):
 def test_hosts(self):
  self.assertEqual(normalize_mongo_hosts(' 192.168.1.4, 192.168.1.2 '),'192.168.1.4,192.168.1.2')
  uri='mongodb://user:password@host/db?replicaSet=rs';self.assertEqual(normalize_mongo_hosts(uri),uri)
 def test_get_storage_failure(self):
  predictor=Mock();predictor.predictFromDB.side_effect=ModelArtifactCacheUnavailableError('private connection details')
  env=dict(JSONResponse=JSONResponse,ModelArtifactCacheUnavailableError=ModelArtifactCacheUnavailableError,_get_request_predictor=lambda:predictor,build_error_response=lambda smi,code,message:{'error':{'code':code,'message':message}})
  fn=load_function('util/helpers.py','make_predictdb_response',env);r=fn(1066,smiles='CCO');self.assertEqual(r.status_code,503);self.assertNotIn(b'private',r.body)
  self.assertEqual(fn(1066).status_code,400)
 def test_batch_storage_failure(self):
  predictor=Mock();predictor.predictFromDB.side_effect=ModelArtifactCacheUnavailableError('private')
  env=dict(JSONResponse=JSONResponse,ModelArtifactCacheUnavailableError=ModelArtifactCacheUnavailableError,_get_request_predictor=lambda:predictor,_parse_predictdb_post_input=lambda b:((1066,['CCO'],[None]),None,None),dedupe_smiles_preserve_order=lambda x:(x,{'CCO':[0]}),os=types.SimpleNamespace(getenv=lambda *a:'thread'),build_batch_error_response=lambda code,message:{'error':{'code':code,'message':message}})
  r=load_function('util/helpers.py','make_predictdb_post_response',env)({});self.assertEqual(r.status_code,503)
 def test_initializer_preserves_unavailability_and_missing(self):
  tree=ast.parse((ROOT/'model_ws_db_utilities.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelInitializer');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='initModel');fn.decorator_list=[]
  cache=Mock();cache.ModelArtifactCacheUnavailableError=ModelArtifactCacheUnavailableError;cache.postgres_fallback_enabled.return_value=False;cache.read_model_snapshot.side_effect=ModelArtifactCacheUnavailableError('offline')
  env={'model_cache':cache,'logging':Mock()};exec(compile(ast.Module(body=[fn],type_ignores=[]),'initializer','exec'),env)
  with self.assertRaises(ModelArtifactCacheUnavailableError):env['initModel'](Mock(),1066)
  cache.read_model_snapshot.side_effect=None;cache.read_model_snapshot.return_value=None;self.assertIsNone(env['initModel'](Mock(),999999))
 def test_process_worker_propagates_storage_failure(self):
  predictor=Mock();predictor.predictFromDB.side_effect=ModelArtifactCacheUnavailableError('offline')
  env=dict(_PROCESS_PREDICTOR=predictor,ModelArtifactCacheUnavailableError=ModelArtifactCacheUnavailableError)
  fn=load_function('util/helpers.py','predict_smiles_in_process',env)
  with self.assertRaises(ModelArtifactCacheUnavailableError):fn((1066,'CCO',False))
 def test_snapshot_missing_vs_broken(self):
  client=Mock();collection=Mock();collection.find_one.return_value=None
  env=dict(connect_mongo=lambda:(client,collection,Mock(),Mock()),_model_key=str,MODEL_CACHE_SCHEMA_VERSION=1,ModelArtifactCacheUnavailableError=ModelArtifactCacheUnavailableError,PyMongoError=ConnectionError,logging=Mock(),_read_blob=lambda *a:None)
  fn=load_function('db/model_cache.py','read_model_snapshot',env)
  self.assertIsNone(fn(999999))
  collection.find_one.return_value={'owner_key':'model:1066','chunk_count':2}
  with self.assertRaises(ModelArtifactCacheUnavailableError):fn(1066)
  collection.find_one.side_effect=ConnectionError('offline')
  with self.assertRaises(ModelArtifactCacheUnavailableError):fn(1066)
if __name__=='__main__':unittest.main()
