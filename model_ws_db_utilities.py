import os
import pickle

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

from qsar_models import ModelByte, Model


def getDatabaseSession():
    DB_USER = os.getenv('DEV_QSAR_USER')
    DB_PASSWORD = os.getenv('DEV_QSAR_PASS')
    DB_HOST = os.getenv('DEV_QSAR_HOST', 'localhost')
    DB_PORT = os.getenv('DEV_QSAR_PORT', 5432)
    DB_NAME = os.getenv('DEV_QSAR_DATABASE', 'qsar')
    connect_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connect_url, echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def saveModelToDatabase(model, model_id):
    session = getDatabaseSession()
    genericModel = session.query(Model).filter_by(id=model_id).first()

    modelBytes = ModelByte(created_at=func.now(),
                           created_by=os.getenv("LAN_ID"),
                           updated_at=func.now(),
                           updated_by=None,
                           bytes=pickle.dumps(model),
                           fk_model=genericModel)
    session.add(modelBytes)
    session.flush()
    session.commit()
