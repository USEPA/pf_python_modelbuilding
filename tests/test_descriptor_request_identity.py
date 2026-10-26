import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'API_Utilities.py'

class DescriptorRequestIdentityTests(unittest.TestCase):
    def test_single_descriptor_request_declares_smiles_type(self):
        cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DescriptorsAPI')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='call_descriptors_get')
        session=SimpleNamespace(get=Mock(return_value='response'))
        env={'get_requests_session':lambda:session}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
        self.assertEqual(env[node.name](None,'http://router','C[C@H](O)F','webtest'),'response')
        params=session.get.call_args.kwargs['params']
        self.assertEqual(params['chemIdType'],'SMILES')
        self.assertEqual(params['smiles'],'C[C@H](O)F')

if __name__=='__main__':unittest.main()
