# coding: utf-8
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, LargeBinary, String, UniqueConstraint, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class DescriptorEmbedding(Base):
    __tablename__ = 'descriptor_embeddings'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True, server_default=text("nextval('qsar_models.descriptor_embeddings_id_seq'::regclass)"))
    created_at = Column(DateTime)
    created_by = Column(String(255))
    description = Column(String(2047))
    descriptor_set_name = Column(String(255))
    embedding_tsv = Column(String(2047), nullable=False)
    name = Column(String(255), unique=True)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    dataset_name = Column(String(255))
    importance_tsv = Column(String(2047), nullable=False)


class Method(Base):
    __tablename__ = 'methods'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    description = Column(String(2047))
    hyperparameters = Column(String(2047))
    is_binary = Column(Boolean, nullable=False)
    name = Column(String(255), unique=True)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))


class ModelSet(Base):
    __tablename__ = 'model_sets'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    description = Column(String(1000))
    name = Column(String(255), unique=True)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))


class Statistic(Base):
    __tablename__ = 'statistics'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    description = Column(String(1000))
    is_binary = Column(Boolean, nullable=False)
    name = Column(String(255), unique=True)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))


class ModelSetReport(Base):
    __tablename__ = 'model_set_reports'
    __table_args__ = (
        UniqueConstraint('fk_model_set_id', 'dataset_name', 'splitting_name'),
        {'schema': 'qsar_models'}
    )

    id = Column(BigInteger, primary_key=True, server_default=text("nextval('qsar_models.model_set_reports_id_seq'::regclass)"))
    created_at = Column(DateTime)
    created_by = Column(String(255))
    dataset_name = Column(String(255), nullable=False)
    splitting_name = Column(String(255), nullable=False)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_set_id = Column(ForeignKey('qsar_models.model_sets.id'), nullable=False)
    file = Column(LargeBinary)

    fk_model_set = relationship('ModelSet')


class Model(Base):
    __tablename__ = 'models'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    dataset_name = Column(String(255), nullable=False)
    descriptor_set_name = Column(String(255), nullable=False)
    splitting_name = Column(String(255), nullable=False)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_method_id = Column(ForeignKey('qsar_models.methods.id'), nullable=False)
    fk_descriptor_embedding_id = Column(ForeignKey('qsar_models.descriptor_embeddings.id'))
    fk_descriptor_embedding = relationship('DescriptorEmbedding')
    fk_method = relationship('Method')


class ModelByte(Base):
    __tablename__ = 'model_bytes'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False, unique=True)
    bytes = Column(LargeBinary, nullable=False)

    fk_model = relationship('Model', uselist=False)


class ModelQmrf(Base):
    __tablename__ = 'model_qmrfs'
    __table_args__ = {'schema': 'qsar_models'}

    id = Column(BigInteger, primary_key=True, server_default=text("nextval('qsar_models.model_qmrfs_id_seq'::regclass)"))
    created_at = Column(DateTime)
    created_by = Column(String(255))
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)
    file = Column(LargeBinary)

    fk_model = relationship('Model')


class ModelStatistic(Base):
    __tablename__ = 'model_statistics'
    __table_args__ = (
        UniqueConstraint('fk_statistic_id', 'fk_model_id'),
        {'schema': 'qsar_models'}
    )

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    statistic_value = Column(Float(53), nullable=False)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)
    fk_statistic_id = Column(ForeignKey('qsar_models.statistics.id'), nullable=False)

    fk_model = relationship('Model')
    fk_statistic = relationship('Statistic')


class ModelsInConsensusModel(Base):
    __tablename__ = 'models_in_consensus_models'
    __table_args__ = (
        UniqueConstraint('fk_consensus_model_id', 'fk_model_id'),
        {'schema': 'qsar_models'}
    )

    id = Column(BigInteger, primary_key=True, server_default=text("nextval('qsar_models.models_in_consensus_models_id_seq'::regclass)"))
    created_at = Column(DateTime)
    created_by = Column(String(255))
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    model_weight = Column(Float(53))
    fk_consensus_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)

    fk_consensus_model = relationship('Model', primaryjoin='ModelsInConsensusModel.fk_consensus_model_id == Model.id')
    fk_model = relationship('Model', primaryjoin='ModelsInConsensusModel.fk_model_id == Model.id')


class ModelsInModelSet(Base):
    __tablename__ = 'models_in_model_sets'
    __table_args__ = (
        UniqueConstraint('fk_model_id', 'fk_model_set_id'),
        {'schema': 'qsar_models'}
    )

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime)
    created_by = Column(String(255))
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)
    fk_model_set_id = Column(ForeignKey('qsar_models.model_sets.id'), nullable=False)

    fk_model = relationship('Model')
    fk_model_set = relationship('ModelSet')


class Prediction(Base):
    __tablename__ = 'predictions'
    __table_args__ = (
        UniqueConstraint('canon_qsar_smiles', 'fk_model_id'),
        {'schema': 'qsar_models'}
    )

    id = Column(BigInteger, primary_key=True)
    canon_qsar_smiles = Column(String(255))
    created_at = Column(DateTime)
    created_by = Column(String(255))
    qsar_predicted_value = Column(Float(53), nullable=False)
    updated_at = Column(DateTime)
    updated_by = Column(String(255))
    fk_model_id = Column(ForeignKey('qsar_models.models.id'), nullable=False)

    fk_model = relationship('Model')
