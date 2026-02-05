'''
Created on Feb 5, 2026
Creates records in database using SQLAlchemy with automapping
@author: TMARTI02
'''

"""
Reflects ORM classes from the database and provides helpers to create rows
for arbitrary tables. Reflection results are cached per (Engine, schema, table).

Usage:
  loader = DatabaseLoader(default_schema="qsar_models")
  with Session(engine) as session:
      pk = loader.create_row(
          session,
          table="models",
          record={"name": "My QSAR model", "created_by": "tmarti02", ...},
      )
      session.commit()
"""


from typing import Any, Iterable, Mapping, Optional, Dict, Tuple, List
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import MetaData

class DatabaseLoader:

    def __init__(self, default_schema: Optional[str] = None):
        self.default_schema = default_schema
        # cache key: (id(engine), schema or None, table)
        self._class_cache: Dict[Tuple[int, Optional[str], str], type] = {}
    
    # ---- internals ---------------------------------------------------------
    
    def _engine_from_session(self, session: Session):
        bind = session.get_bind()
        return getattr(bind, "engine", bind)
    
    def _get_mapped_class(self, session, table: str, schema: str | None = None):
        """
        Creates a mapping class on the fly and stores in cache. This way you dont have to create a hard coded table class
        """
        engine = self._engine_from_session(session)
        schema = schema if schema is not None else self.default_schema
        cache_key = (id(engine), schema, table)
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
    
        # Reflect the single table
        metadata = MetaData()
        metadata.reflect(bind=engine, schema=schema, only=[table])
    
        Base = automap_base(metadata=metadata)
        Base.prepare()  # no 'only' here
    
        # Find the mapped class by matching table name/schema
        mapped_cls = None
        for mapper in Base.registry.mappers:
            lt = mapper.local_table
            if lt.name == table and lt.schema == schema:
                mapped_cls = mapper.class_
                break
    
        if mapped_cls is None:
            raise LookupError(f"Automap did not produce a class for {schema+'.' if schema else ''}{table}")
    
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
        schema: Optional[str] = None,
    ):
        """
        Return the reflected ORM class for the given table (and optional schema).
        """
        return self._get_mapped_class(session, table=table, schema=schema)
    
    def create_row(
        self,
        session: Session,
        table: str,
        record: Mapping[str, Any],
        schema: Optional[str] = None,
    ) -> Any:
        """
        Insert a single row into schema.table and return the primary key.
        - Does not commit; caller should session.commit() or rollback().
        - 'record' can include any subset of columns; DB defaults will be applied if defined.
        """
        cls = self._get_mapped_class(session, table=table, schema=schema)
        row = cls(**record)
        session.add(row)
        session.flush()  # ensures PK is populated
        return self._extract_pk(row)
    
    def create_many(
        self,
        session: Session,
        table: str,
        records: Iterable[Mapping[str, Any]],
        schema: Optional[str] = None,
    ) -> List[Any]:
        """
        Bulk insert multiple rows and return a list of primary keys in the same order.
        - Uses normal add_all + flush to populate PKs (safe for defaults/triggers).
        - For very large batches, consider chunking for memory usage.
        """
        cls = self._get_mapped_class(session, table=table, schema=schema)
        rows = [cls(**rec) for rec in records]
        session.add_all(rows)
        session.flush()
        return [self._extract_pk(r) for r in rows]    

if __name__ == '__main__':
    pass