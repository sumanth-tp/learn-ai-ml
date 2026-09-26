"""Triage suggestions behind a provider-agnostic chat-model interface."""

from helpdesk_mcp.triage.service import TriageResult, TriageService, build_chat_model

__all__ = ["TriageResult", "TriageService", "build_chat_model"]
