import json
import logging
import os
from typing import Optional

from sqlalchemy import URL, Engine, text, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from util import predict_constants as pc
from models.case_studies import run_model_building_db as rmdb

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


class RunDataset:
    def __init__(self, dataset_name, qsar_method, embedding=None, folder_embedding=None, cross_validate=True,
                 run_AD=True, feature_selection=True, fs_previous_embedding=True, params=None,
                 descriptor_set_name="WebTEST-default", splitting_name="RND_REPRESENTATIVE",
                 ad_measure=None, add_LOGP_Martin=False, logp_columns=None, write_to_db=False, user="tmarti02",
                 unique_identifier=None, append_to_models_folder="", subfolder=None, save_initial_dfs=False,
                 outlier_filter_methods=None, include_feature_count_in_subfolder=False):
        self.dataset_name = dataset_name
        self.qsar_method = qsar_method
        self.embedding = embedding
        self.folder_embedding = folder_embedding
        self.cross_validate = cross_validate
        self.run_AD = run_AD
        self.feature_selection = feature_selection
        self.fs_previous_embedding = fs_previous_embedding
        self.params = params
        self.descriptor_set_name = descriptor_set_name
        self.splitting_name = splitting_name
        self.ad_measure = ad_measure
        self.add_LOGP_Martin = add_LOGP_Martin
        self.logp_columns = logp_columns
        self.write_to_db = write_to_db
        self.user = user
        self.unique_identifier = unique_identifier
        self.append_to_models_folder = append_to_models_folder
        self.subfolder = subfolder
        self.save_initial_dfs = save_initial_dfs
        self.outlier_filter_methods = outlier_filter_methods or []
        self.include_feature_count_in_subfolder = include_feature_count_in_subfolder

        self.post_init()

    def post_init(self):
        # Initialize database engine and session
        self.engine = self.engine or self.getEngine()
        self.session = self.session or self.getSession(self.engine)
        self.current_user_id = self.get_current_user_id()

        # Set default splitting name if not provided
        self.splitting_name = self.splitting_name or "RND_REPRESENTATIVE"

        # Set feature selection to False for certain QSAR methods
        if self.qsar_method in {"gcm", "svm"}:
            self.feature_selection = False

        # Set embedding and feature selection based on provided parameters
        if self.embedding is not None:
            self.folder_embedding = "custom"
            self.feature_selection = False
        elif self.folder_embedding is not None:
            self.feature_selection = False
            file_name_embedding = "results.json"
            file_path_embedding = os.path.join(PROJECT_ROOT, "data/models", self.dataset_name, self.folder_embedding, file_name_embedding)
            
            with open(file_path_embedding, "r", encoding="utf-8") as f:
                results = json.load(f)
                embedding = results["model_details"]["embedding"]
                # print(f"from {folder_embedding}:{embedding}")
        else:
            self.fs_previous_embedding = False

        # Set default applicability domain measures if not provided
        self.ad_measure = self.ad_measure or [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_All_Descriptors_Euclidean]
        if self.descriptor_set_name == "WebTEST-default":
            self.ad_measure.append(pc.Applicability_Domain_TEST_Fragment_Counts)

        # Set hyperparameters if not provided
        if self.params is None:
            self.params = rmdb.set_hyper_parameters(
                qsar_method=self.qsar_method,
                feature_selection=self.feature_selection,
                descriptor_set_name=self.descriptor_set_name,
                splitting_name=self.splitting_name,
                dataset_name=self.dataset_name,
                ad_measure=self.ad_measure
            )
        self.params.feature_selection = self.feature_selection

        # Set parameters related to outlier handling
        if not hasattr(self.params, "outlier_filter_methods") and self.outlier_filter_methods is not None and self.outlier_filter_methods != False:
            # outlier_filter_methods = ["iqr", "hampel", "robust_z", "esd"]
            self.params.outlier_filter_methods = self.outlier_filter_methods

        if getattr(self.params, "outlier_filter_methods", False) and not hasattr(self.params, "outlier_filter_columns"):
            self.params.outlier_filter_columns = ["Property"]

        if getattr(self.params, "outlier_filter_methods", False) and not hasattr(self.params, "outlier_filter_trim_mode"):
            self.params.outlier_filter_trim_mode = "remove"

    @staticmethod
    def getEngine(connect_args: Optional[dict] = {}) -> Engine:
        """Create SQLAlchemy engine for PostgreSQL connection.
        
        Uses environment variables for database credentials:
            DEV_QSAR_USER, DEV_QSAR_PASS, DEV_QSAR_HOST, DEV_QSAR_PORT, DEV_QSAR_DATABASE
        
        Args:
            connect_args: Additional arguments to pass to URL.create(). Defaults to empty dict.
        
        Returns:
            Engine: SQLAlchemy engine configured for PostgreSQL connection.
        """
        connect_url = URL.create(
            # drivername='postgresql+psycopg2',
            # username=os.getenv('DEV_QSAR_USER'),
            # password=os.getenv('DEV_QSAR_PASS'),
            # host=os.getenv('DEV_QSAR_HOST', 'localhost'),
            # port=os.getenv('DEV_QSAR_PORT', 5432),
            # database=os.getenv('DEV_QSAR_DATABASE'),
            drivername='postgresql+psycopg2',
            username=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB'),
            **connect_args
        )
        engine = create_engine(connect_url, echo=False)
        
        return engine

    @staticmethod
    def getSession(engine: Engine = None) -> Session:
        """Create SQLAlchemy session for database queries.
        
        Args:
            engine: SQLAlchemy engine instance. If None, creates new engine.
        
        Returns:
            Session: SQLAlchemy session object for executing queries.
        """
        if engine is None:
            engine = RunDataset.getEngine()
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    
    @staticmethod
    def get_current_user_id():
        user_id = os.getenv("USER_ID")
        if user_id is None:
            logging.warning("USER_ID environment variable is not set, falling back to default (tmarti02)")
            user_id = "tmarti02"
        return user_id

    def ensure_session(self) -> Session:
        session = self.session
        try:
            # cheap no-op to force a DB round trip
            session.execute(text("SELECT 1"))
            return session
        except OperationalError:
            try:
                session.close()
            except Exception:
                pass
            session = RunDataset.getSession(self.engine)
            self.session = session
            return session

    @staticmethod
    def ensure_static_session(session):
        try:
            # cheap no-op to force a DB round trip
            session.execute(text("SELECT 1"))
            return session
        except OperationalError:
            try:
                session.close()
            except Exception:
                pass
            return RunDataset.getSession()

    def run(self):
        pass  # Placeholder for the main logic of running the dataset processing. Actual implementation would go here.
