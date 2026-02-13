'''
Created on Feb 5, 2026
Creates records in database using SQLAlchemy with automapping
@author: TMARTI02
'''

"""
Reflects ORM classes from the database and provides helpers to create rows
for arbitrary tables. Reflection results are cached per (Engine, schema, table).

Usage:
  loader = DatabaseLoader(schema="qsar_models")
  with Session(engine) as session:
      pk = loader.create_row(
          session,
          table="models",
          record={"name": "My QSAR model", "created_by": "tmarti02", ...},
      )
      session.commit()
"""

from typing import Any, Iterable, Mapping, Optional, Dict, Tuple, List, Sequence
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import MetaData, create_engine, text, bindparam
from sqlalchemy import inspect as sa_inspect, select as sa_select, update as sa_update
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

import os
import json

 
class DatabaseUtilities:

    def __init__(self, schema, session=None):
        
        self.schema = schema
        
        # cache key: (id(engine), schema or None, table)
        self._class_cache: Dict[Tuple[int, Optional[str], str], type] = {}

        # Build the engine once and reuse it
        connect_url = URL.create(
            drivername='postgresql+psycopg2',
            username=os.getenv('DEV_QSAR_USER'),
            password=os.getenv('DEV_QSAR_PASS'),
            host=os.getenv('DEV_QSAR_HOST', 'localhost'),
            port=int(os.getenv('DEV_QSAR_PORT', 5432)),  # ensure int
            database=os.getenv('DEV_QSAR_DATABASE')
        )
        self.engine = create_engine(connect_url, echo=False, pool_pre_ping=True, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

        if session:
            self.session = session
        else:
        # If you need a ready-to-use session attached, create it here:
            self.session: Session = self.getSession()

    def getSession(self) -> Session:
        return self.SessionLocal()
    
    # ---- internals ---------------------------------------------------------
    
    def _engine_from_session(self, session: Session):
        bind = session.get_bind()
        return getattr(bind, "engine", bind)
    
    
    from sqlalchemy import update as sa_update

    def update_row(self, table, values: dict, **filters):
        """
        Update rows in 'table' matching the provided filters and return the first
        updated row (or None if no rows matched).
    
        Usage:
            dl.update_row(
                table="models",
                values={"name": "user_123", "updated_by": "user"},
                id=123
            )
    
        Notes:
        - Requires at least one filter (to avoid mass updates).
        - Validates both filter and update column names against the automapped model.
        - Uses PostgreSQL RETURNING to fetch the updated row.
        """
        if not values:
            raise ValueError("Provide at least one column=value to update in 'values'.")
        if not filters:
            raise ValueError("Provide at least one column=value filter.")
    
        # Resolve the mapped class
        Model = self._get_mapped_class(table)
    
        # Validate provided column names
        unknown_filters = [c for c in filters if not hasattr(Model, c)]
        if unknown_filters:
            raise ValueError(f"Unknown filter columns for {Model.__name__}: {unknown_filters}")
    
        unknown_values = [c for c in values if not hasattr(Model, c)]
        if unknown_values:
            raise ValueError(f"Unknown update columns for {Model.__name__}: {unknown_values}")
    
        # Build WHERE conditions
        conditions = [getattr(Model, col) == val for col, val in filters.items()]
    
        # Build UPDATE ... RETURNING Model
        stmt = sa_update(Model).where(*conditions).values(**values).returning(Model)
    
        # Execute and commit
        result = self.session.execute(stmt)
        self.session.commit()
    
        # Return the first updated row (or None if no row matched)
        return result.scalar_one_or_none()



    
    def get_row(self, table, **filters):
        """
        Return the first matching mapped row from the given table.
        `table` can be a string table name or a sqlalchemy.Table object.
        filters constrain the search:

        dl.get_row(session=session, table="descriptor_embeddings", embedding_tsv="col1\tcol2", dataset_name="KOC v1 modeling")
        
        """
        if not filters:
            raise ValueError("Provide at least one column=value filter.")
    
        # Resolve the mapped class without hard-coding it.
        # If you need a schema, adapt to pass schema="<schema_name>".
        Model = self._get_mapped_class(table)
    
        # Optional: validate provided column names early
        unknown = [c for c in filters if not hasattr(Model, c)]
        if unknown:
            raise ValueError(f"Unknown columns for {Model.__name__}: {unknown}")
    
        stmt = sa_select(Model).filter_by(**filters).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()
    
    def get_rows(self, table, **filters):
        """
        Return all mapped rows that match the provided filters.
        If no filters are provided, return all rows.
        """
        Model = self._get_mapped_class(table)
    
        if filters:
            unknown = [c for c in filters if not hasattr(Model, c)]
            if unknown:
                raise ValueError(f"Unknown columns for {Model.__name__}: {unknown}")
            stmt = sa_select(Model).filter_by(**filters)
        else:
            # No filters: select all rows
            stmt = sa_select(Model)
    
        return self.session.execute(stmt).scalars().all()
    
    def instance_to_dict(self, obj):
        """
        Convert an ORM instance to a dict of its column attributes.
        Relationships and non-column attributes are not included.
        """
        mapper = sa_inspect(obj.__class__).mapper
        return {attr.key: getattr(obj, attr.key) for attr in mapper.column_attrs}
    
    def print_row_as_json(self, row):
        """
        Print a single row (ORM instance) as JSON.
        Accepts the row returned by get_row or an element from get_rows.
        """
        if row is None:
            print("null")
            return
        print(json.dumps(self.instance_to_dict(row), default=str))
    
    def print_rows_as_json(self, rows):
        """
        Print each row (ORM instance) from get_rows as a separate JSON line.
        """
        for row in rows:
            self.print_row_as_json(row)
    
    def _get_mapped_class(self, table: str):
        """
        Creates a mapping class on the fly and stores in cache. This way you dont have to create a hard coded table class
        """
        engine = self._engine_from_session(self.session)
        
        cache_key = (id(engine), self.schema, table)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
    
        # Reflect the single table
        metadata = MetaData()
        metadata.reflect(bind=engine, schema=self.schema, only=[table])
    
        Base = automap_base(metadata=metadata)
        Base.prepare()  # no 'only' here
    
        # Find the mapped class by matching table name/schema
        mapped_cls = None
        for mapper in Base.registry.mappers:
            lt = mapper.local_table
            if lt.name == table and lt.schema == self.schema:
                mapped_cls = mapper.class_
                break
    
        if mapped_cls is None:
            raise LookupError(f"Automap did not produce a class for {self.schema+'.' if self.schema else ''}{table}")
    
        self._class_cache[cache_key] = mapped_cls
        return mapped_cls
    
    @staticmethod
    def _extract_pk(row: Any) -> Any:
        """
        Return the primary key of a row:
        - If single-column PK: return the scalar value (commonly 'id').
        - If composite PK: return a dict of {column_key: value}.
        """
        state = sa_inspect(row)
        identity = state.identity  # tuple of PK values or None if not flushed
        mapper = state.mapper
        if not identity:
            return None
        if len(identity) == 1:
            return identity[0]
        return {col.key: val for col, val in zip(mapper.primary_key, identity)}
    
    # ---- public API --------------------------------------------------------
    
    def get_class(
        self,
        session: Session,
        table: str,
        schema: Optional[str]=None,
    ):
        """
        Return the reflected ORM class for the given table (and optional schema).
        """
        return self._get_mapped_class(table)
    
    # cant be static if want to make use of default schema
    def create_row(
        self,
        table: str,
        record: Mapping[str, Any],
        schema: Optional[str]=None,
    ) -> Any:
        """
        Insert a single row into schema.table and return the primary key.
        - Does not commit; caller should session.commit() or rollback().
        - 'record' can include any subset of columns; DB defaults will be applied if defined.
        """
        cls = self._get_mapped_class(table)
        row = cls(**record)
        self.session.add(row)
        self.session.flush()  # ensures PK is populated        
        self.session.commit()
        return self._extract_pk(row)
    
    def chunked(self, seq: Sequence[Mapping[str, Any]], size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
        for i in range(0, len(seq), size):
            yield seq[i:i + size]


    def create_many_chunked(
        self,
        table: str,
        records: Sequence[Mapping[str, Any]],
        chunk_size: int = 1000,
    ) -> int:
        """
        Insert records into `table` in chunks using self.create_many.
    
        Transaction behavior:
          - If no transaction is active on the Session, open one and commit on success.
          - If a transaction is already active, do not start a new one; rely on the outer transaction.
    
        Returns the total number of inserted records.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    
        if not records:
            return 0
    
        # We don't manually commit at the end; the context manager (or outer tx) handles it.
        try:
            if self.session.in_transaction():
                # Execute within the existing transaction.
                for batch in self.chunked(records, chunk_size):
                    self.create_many(table=table, records=batch, commit=False)
            else:
                # Manage our own transaction; it auto-commits on success.
                with self.session.begin():
                    for batch in self.chunked(records, chunk_size):
                        self.create_many(table=table, records=batch, commit=False)
    
            return len(records)
    
        except SQLAlchemyError:
            # If we started a transaction with `begin()`, it will auto-rollback on exception.
            # If a caller started the transaction, let them decide how to handle rollback.
            raise

    def create_many(
        self,
        table: str,
        records: Iterable[Mapping[str, Any]],
        schema: Optional[str] = None,
        commit: bool = True,
    ) -> List[Any]:
        """
        Bulk insert multiple rows and return a list of primary keys in the same order.
        - Uses add_all + flush to populate PKs (safe for typical defaults/sequences).
        - Only commits when `commit=True` and no transaction is active.
        """
        cls = self._get_mapped_class(table)
        rows = [cls(**rec) for rec in records]
        if not rows:
            return []
    
        self.session.add_all(rows)
        self.session.flush()  # populate PKs
    
        # Commit only if requested and we're not inside a transaction context
        if commit and not self.session.in_transaction():
            self.session.commit()
    
        return [self._extract_pk(r) for r in rows]

def getSession():
    connect_url = URL.create(
        drivername='postgresql+psycopg2',
        username=os.getenv('DEV_QSAR_USER'),
        password=os.getenv('DEV_QSAR_PASS'),
        host=os.getenv('DEV_QSAR_HOST', 'localhost'),
        port=os.getenv('DEV_QSAR_PORT', 5432),
        database=os.getenv('DEV_QSAR_DATABASE')
    )
    # print(connect_url)
    engine = create_engine(connect_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


if __name__ == '__main__':
    
    from dotenv import load_dotenv, find_dotenv
    
    # Searches upward from the current working directory for ".env.res_qsar"
    env_path = find_dotenv(".env.res_qsar", raise_error_if_not_found=True)    
    load_dotenv(env_path)
    # print(os.getenv('DEV_QSAR_HOST'))
        
    dl = DatabaseUtilities("qsar_models")
    
    # Get row in database using filter:
    row = dl.get_row(table="descriptor_embeddings", embedding_tsv="col1\tcol2", dataset_name="KOC v1 modeling")

    dl.print_row_as_json(row)
            
    rows = dl.get_rows("descriptor_embeddings", dataset_name="KOC v1 modeling")    
    dl.print_rows_as_json(rows)
    
    # row = dl.get_rows(session, "descriptor_embeddings", dataset_name="KOC v1 modeling", embedding_tsv="col1\tcol2")    
    # dl.print_row_as_json(row)
    
