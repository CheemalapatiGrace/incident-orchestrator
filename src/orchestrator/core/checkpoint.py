"""LangGraph checkpointer on the domain store's SQLite connection.

``SqliteSaver`` (langgraph-checkpoint-sqlite) owns the ``checkpoints`` / ``writes`` tables: the durable
state of every run, keyed by ``thread_id == run_id``. We bind it to the *same* connection as
:class:`~orchestrator.core.store.Store` (audit log, approvals, spans, events ...) instead of using
``AsyncSqliteSaver`` for two reasons found the hard way:

* ``AsyncSqliteSaver`` talks to SQLite from its own thread. Our synchronous store writes on the event-loop
  thread to the same file, so a store write could block the loop while the saver thread was waiting for the
  loop to send its COMMIT - a lock inversion that stalls the whole run until ``busy_timeout``.
* Every statement is microseconds long (as in the store), so a thread hop per checkpoint write only costs
  throughput.

The async methods therefore run the (fast) synchronous implementation inline. Each save is one explicit
transaction, so a crash can never leave half of a task's writes behind.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.sqlite import SqliteSaver

from orchestrator.core.store import Store


class InlineSqliteSaver(SqliteSaver):
    def __init__(self, store: Store) -> None:
        super().__init__(store._db)
        self.lock = store._lock  # one lock for the shared connection

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[Any]:
        with self.lock:
            self.setup()
            cur = self.conn.cursor()
            cur.row_factory = None  # the store's connection returns sqlite3.Row; the saver unpacks tuples
            try:
                if transaction:
                    cur.execute("BEGIN IMMEDIATE")
                yield cur
                if transaction:
                    self.conn.commit()
            except BaseException:
                if transaction and self.conn.in_transaction:
                    self.conn.rollback()
                raise
            finally:
                cur.close()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
                    before: RunnableConfig | None = None, limit: int | None = None) -> AsyncIterator[CheckpointTuple]:
        for t in self.list(config, filter=filter, before=before, limit=limit):
            yield t

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    async def aget_delta_channel_history(self, *, config: RunnableConfig, channels: Sequence[str]) -> Mapping[str, Any]:
        return self.get_delta_channel_history(config=config, channels=channels)
