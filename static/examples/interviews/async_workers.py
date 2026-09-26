"""Bound queued/active work and keep transport failures separate from scores."""
import asyncio
from dataclasses import dataclass

@dataclass(frozen=True)
class Result:
    case_id: str
    status: str
    value: object = None

async def run_cases(cases, operation, workers=3, capacity=6, timeout=1.0):
    if workers < 1 or capacity < 1 or timeout <= 0:
        raise ValueError("positive limits required")
    queue = asyncio.Queue(maxsize=capacity)
    results = []
    stop = object()
    seen = set()
    async def produce():
        for case_id, payload in cases:
            if case_id in seen:
                raise ValueError(f"duplicate case ID: {case_id}")
            seen.add(case_id)
            await queue.put((case_id, payload))
        for _ in range(workers):
            await queue.put(stop)
    async def consume():
        while True:
            item = await queue.get()
            try:
                if item is stop:
                    return
                case_id, payload = item
                try:
                    async with asyncio.timeout(timeout):
                        value = await operation(payload)
                    results.append(Result(case_id, "ok", value))
                except TimeoutError:
                    results.append(Result(case_id, "timeout"))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    results.append(Result(case_id, "error", type(exc).__name__))
            finally:
                queue.task_done()
    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        for _ in range(workers):
            group.create_task(consume())
    # Results/seen are O(number of cases). Stream them to durable storage at scale.
    return results

async def demo():
    async def operation(value):
        await asyncio.sleep(0)
        if value == 2:
            raise ValueError("synthetic malformed response")
        return value * value
    print(sorted(await run_cases(((str(i), i) for i in range(4)), operation), key=lambda r: r.case_id))

if __name__ == "__main__":
    asyncio.run(demo())
