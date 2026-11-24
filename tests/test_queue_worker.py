import asyncio

from conftest import alert_for, approve_when_pending, make_engine, wait_until
from orchestrator.core.queue import InMemoryQueue, RedisQueue


async def test_inmemory_queue_fifo_ack_and_empty_timeout():
    q = InMemoryQueue()
    await q.enqueue("a")
    await q.enqueue("b")
    l1 = await q.dequeue(0.1)
    assert l1.item == "a" and await q.size() == 1
    await q.ack(l1)
    assert (await q.dequeue(0.1)).item == "b"
    assert await q.dequeue(0.05) is None


async def test_nack_redelivers_then_dead_letters_after_max_deliveries():
    q = InMemoryQueue(max_deliveries=2)
    await q.enqueue("poison")
    await q.nack(await q.dequeue(0.1))
    second = await q.dequeue(0.1)
    assert second.deliveries == 2
    await q.nack(second)
    assert await q.dequeue(0.05) is None and await q.dead_letters() == ["poison"]


async def test_unacked_work_is_redelivered_after_visibility_timeout():
    q = InMemoryQueue(visibility_timeout_s=0.05)
    await q.enqueue("job")
    await q.dequeue(0.1)  # worker takes it and "dies"
    await asyncio.sleep(0.08)
    again = await q.dequeue(0.1)
    assert again is not None and again.item == "job" and again.deliveries == 2


class FakeRedis:
    """Implements just the redis.asyncio commands RedisQueue uses."""

    def __init__(self):
        self.lists, self.hashes = {}, {}

    async def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)

    async def blmove(self, src, dst, timeout, a, b):
        s = self.lists.get(src, [])
        if not s:
            return None
        v = s.pop()
        self.lists.setdefault(dst, []).insert(0, v)
        return v.encode()

    async def hincrby(self, h, k, n):
        d = self.hashes.setdefault(h, {})
        d[k] = d.get(k, 0) + n
        return d[k]

    async def hdel(self, h, k):
        self.hashes.get(h, {}).pop(k, None)

    async def lrem(self, k, n, v):
        if v in self.lists.get(k, []):
            self.lists[k].remove(v)

    async def llen(self, k):
        return len(self.lists.get(k, []))

    async def lrange(self, k, a, b):
        return list(self.lists.get(k, []))


async def test_redis_adapter_satisfies_the_queue_contract():
    q = RedisQueue(FakeRedis(), max_deliveries=2)
    await q.enqueue("r1")
    lease = await q.dequeue(1)
    assert lease.item == "r1" and lease.deliveries == 1
    await q.nack(lease)  # back to pending
    lease2 = await q.dequeue(1)
    assert lease2.deliveries == 2
    await q.nack(lease2)  # exceeded -> DLQ
    assert await q.size() == 0 and await q.dead_letters() == ["r1"]
    await q.enqueue("r2")
    await q.ack(await q.dequeue(1))
    assert await q.size() == 0 and await q.dead_letters() == ["r1"]


async def test_worker_pool_processes_many_runs_concurrently():
    eng = make_engine(worker_concurrency=4)
    ids = [eng.submit(alert_for(s, id=f"w{i}")) for i, s in enumerate(["checkout-bad-deploy", "search-traffic-spike", "inventory-disk-full", "auth-config-regression"] * 2)]
    eng.start_workers(4)
    ap = asyncio.create_task(_approve_all(eng))
    for r in ids:
        await eng.queue.enqueue(r)
    await wait_until(lambda: all(eng.store.get_run(r)["status"] == "completed" for r in ids), 10)
    ap.cancel()
    await eng.stop_workers()
    assert all(len(eng._envs[r].mutations) == 1 for r in ids)


async def test_two_workers_never_execute_the_same_run_twice():
    eng_a = make_engine(worker_id="a")
    eng_b = type(eng_a)(eng_a.cfg, store=eng_a.store, llm=eng_a.deps.llm, env_factory=eng_a.env_factory, worker_id="b")
    rid = eng_a.submit(alert_for("inventory-disk-full"))
    ap = asyncio.create_task(approve_when_pending(eng_a, rid))
    r1, r2 = await asyncio.gather(eng_a.execute(rid), eng_b.execute(rid))
    ap.cancel()
    assert (r1 is None) != (r2 is None)  # exactly one of them ran it
    assert len((eng_a._envs.get(rid) or eng_b._envs[rid]).mutations) == 1


async def _approve_all(eng):
    while True:
        for ap in eng.store.list_approvals(status="pending"):
            eng.hitl.approve(ap["approval_id"], "t", "ok")
        await asyncio.sleep(0.005)
