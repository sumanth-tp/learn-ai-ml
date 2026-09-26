"""Crash and resume against a real SQLite checkpoint file, across two service instances."""

from __future__ import annotations

import dataclasses

import pytest

from research_analyst.service import ResearchService
from tests.conftest import SODIUM_Q
from tests.fakes import CrashOnceInWriter, SimulatedCrash


async def test_resume_after_crash_does_not_redo_finished_work(settings, deps):
    brain = CrashOnceInWriter()
    d = dataclasses.replace(deps, brain=brain)

    async with ResearchService(settings, deps=d) as svc:  # SQLite at settings.checkpoint_db
        with pytest.raises(SimulatedCrash):
            async for _ in svc.stream(SODIUM_Q, "crash-1"):
                pass
    assert settings.checkpoint_db.exists()
    grades_before = brain.calls["grade"]

    async with ResearchService(settings, deps=d) as svc:  # a "new process"
        st = await svc.status("crash-1")
        assert st.status == "running_or_interrupted" and st.next_nodes == ["write"]
        events = [ev.type async for ev in svc.resume("crash-1")]
        st = await svc.status("crash-1")

    assert events[0] == "run_resumed" and events[-1] == "report_ready"
    assert st.status == "complete" and st.report.sections
    assert brain.calls["plan"] == 1  # planning was not repeated
    assert brain.calls["grade"] == grades_before  # researchers were not re-run


async def test_status_of_unknown_thread(service):
    st = await service.status("does-not-exist")
    assert st.status == "not_found"


async def test_delete_and_purge_enforce_retention(settings, deps):
    async with ResearchService(settings, deps=deps) as svc:
        await svc.run(SODIUM_Q, "keep-me")
        await svc.run(SODIUM_Q, "erase-me")
        assert await svc.delete("erase-me") is True
        assert (await svc.status("erase-me")).status == "not_found"
        assert await svc.delete("erase-me") is False
        assert await svc.purge(older_than_days=30) == []  # nothing is old yet
        assert await svc.purge(older_than_days=-1) == ["keep-me"]  # everything is "old"
        assert (await svc.status("keep-me")).status == "not_found"
