
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import pickle
from qsar_models import ModelByte, Model

def getDatabaseSession():
    connect_url = URL('postgresql',
                      username=os.getenv('DEV_QSAR_USER'),
                      password=os.getenv('DEV_QSAR_PASS'),
                      host=os.getenv('DEV_QSAR_HOST'),
                      port=os.getenv('DEV_QSAR_PORT'),
                      database=os.getenv('DEV_QSAR_DATABASE'))
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
