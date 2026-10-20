"""Session-local Hermes plan selection, issue #328."""
from __future__ import annotations

import importlib
import importlib.util
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / ".hermes" / "plugins" / "planning-with-files"
SPEC = importlib.util.spec_from_file_location(
    "pwf_session_binding_test", PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
assert SPEC is not None and SPEC.loader is not None
plugin = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plugin
SPEC.loader.exec_module(plugin)
hooks = importlib.import_module(SPEC.name + ".hooks")
sessions = importlib.import_module(SPEC.name + ".session_binding")
tools = importlib.import_module(SPEC.name + ".tools")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    for key in ("PLAN_ID", "PWF_PLAN_ROOT", "PLANNING_DISABLED"):
        monkeypatch.delenv(key, raising=False)
    runtime = types.ModuleType("agent.runtime_cwd")
    runtime.resolve_agent_cwd = lambda: root
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.runtime_cwd", runtime)
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: profile
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.chdir(root)
    handlers = {}

    class Context:
        def register_tool(self, name, handler, **kw):
            handlers[name] = handler

        def register_hook(self, *args, **kw):
            pass

    plugin.register(Context())
    return root, profile, handlers


def plan(root, slug, text=None):
    directory = root / ".planning" / slug
    directory.mkdir(parents=True)
    (directory / "task_plan.md").write_text(text or f"# {slug} ONLY\n", encoding="utf-8")
    return directory


def context(session):
    payload = hooks.pre_llm_call(session_id=session, user_message="continue")
    return payload["context"] if payload else ""


def select(handlers, session, value, **args):
    return json.loads(handlers["planning_with_files_select"]({"plan": value, **args}, session_id=session))


def test_two_sessions_select_different_plans_without_environment_mutation(workspace):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    before = dict(os.environ)
    assert select(handlers, "thread-a", "plan-a")["ok"]
    assert select(handlers, "thread-b", "plan-b")["ok"]
    assert "plan-a ONLY" in context("thread-a")
    assert "plan-b ONLY" not in context("thread-a")
    assert "plan-b ONLY" in context("thread-b")
    assert "plan-a ONLY" not in context("thread-b")
    assert dict(os.environ) == before


def test_selected_stale_plan_does_not_fall_back(workspace):
    root, _profile, handlers = workspace
    selected = plan(root, "plan-a")
    plan(root, "plan-b")
    assert select(handlers, "thread-a", "plan-a")["ok"]
    (selected / "task_plan.md").unlink()
    (root / ".planning" / ".active_plan").write_text("plan-b\n", encoding="ascii")
    assert "plan-b ONLY" not in context("thread-a")
    status = json.loads(handlers["planning_with_files_status"]({}, session_id="thread-a"))
    assert not status["exists"]
    init = json.loads(handlers["planning_with_files_init"]({"name": "replacement"}, session_id="thread-a"))
    assert not init["ok"]
    assert not list((root / ".planning").glob("*-replacement"))
    assert select(handlers, "thread-a", "plan-b")["ok"]
    assert "plan-b ONLY" in context("thread-a")


def test_successful_named_init_binds_its_calling_session(workspace):
    root, _profile, handlers = workspace
    plan(root, "other")
    result = json.loads(handlers["planning_with_files_init"]({"name": "Own task"}, session_id="init-session"))
    assert result["ok"]
    status = json.loads(handlers["planning_with_files_status"]({}, session_id="init-session"))
    assert status["exists"]
    assert status["plan_id"] == result["plan_id"]


def test_reloaded_module_and_separate_profile_preserve_isolation(workspace, monkeypatch, tmp_path):
    root, profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    assert select(handlers, "thread-a", "plan-a")["ok"]
    assert len(list((profile / "planning-with-files" / "sessions").glob("*.json"))) == 1
    importlib.reload(sessions)
    assert "plan-a ONLY" in context("thread-a")
    other_profile = tmp_path / "other-profile"
    other_profile.mkdir()
    monkeypatch.setattr(sys.modules["hermes_constants"], "get_hermes_home", lambda: other_profile)
    assert "Multiple plans" in context("thread-a")
    assert "plan-a ONLY" not in context("thread-a")


def test_concurrent_sessions_keep_plan_reminder_and_gate_state_separate(workspace):
    root, profile, handlers = workspace
    a = plan(root, "plan-a", "### Phase 1: A\n**Status:** in_progress\n")
    b = plan(root, "plan-b", "### Phase 1: B\n**Status:** complete\n")
    (a / ".mode").write_text("autonomous gate\n", encoding="ascii")
    (a / ".attestation").write_text(hashlib.sha256((a / "task_plan.md").read_bytes()).hexdigest(), encoding="ascii")

    def exercise(index):
        identity = f"session-{index}"
        slug = "plan-a" if index % 2 == 0 else "plan-b"
        assert select(handlers, identity, slug)["ok"]
        for _ in range(8):
            text = context(identity)
            assert f"plan: {slug}" in text
            assert f"plan: {'plan-b' if slug == 'plan-a' else 'plan-a'}" not in text
        return identity, slug

    with ThreadPoolExecutor(max_workers=12) as pool:
        selected = list(pool.map(exercise, range(24)))
    assert len(list((profile / "planning-with-files" / "sessions").glob("*.json"))) == 24
    for identity, slug in selected:
        status = json.loads(handlers["planning_with_files_status"]({}, session_id=identity))
        assert status["plan_id"] == slug
    hooks.post_tool_call(tool_name="write_file", args={"path": "source.py", "content": "done"}, session_id="session-0")
    assert "Update progress.md" not in context("session-1")
    assert "Update progress.md" in context("session-0")
    assert hooks.pre_verify(session_id="session-1") is None
    assert hooks.pre_verify(session_id="session-0")["action"] == "continue"


def test_context_scope_survives_async_switch_and_resets_after_exception(workspace):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    assert select(handlers, "thread-a", "plan-a")["ok"]
    assert select(handlers, "thread-b", "plan-b")["ok"]

    async def inspect(identity, slug):
        with sessions.session_scope({"session_id": identity}):
            await asyncio.sleep(0)
            assert sessions.current_binding().plan_id == slug
        assert sessions.current_binding() is None

    async def exercise():
        await asyncio.gather(inspect("thread-a", "plan-a"), inspect("thread-b", "plan-b"))

    asyncio.run(exercise())
    with pytest.raises(RuntimeError):
        with sessions.session_scope({"session_id": "thread-a"}):
            raise RuntimeError("failure")
    assert sessions.current_binding() is None
    assert sessions.current_session_id() == ""


def test_identity_comes_from_host_metadata_or_actual_contextvar(workspace, monkeypatch):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-process-session")
    host = types.ModuleType("gateway.session_context")
    host._SESSION_ID = ContextVar("test_hermes_id", default=object())
    host.get_session_env = lambda name: os.environ.get(name, "")
    monkeypatch.setitem(sys.modules, "gateway", types.ModuleType("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.session_context", host)
    bad = json.loads(handlers["planning_with_files_select"]({"plan": "plan-a", "session_id": "spoofed"}))
    assert not bad["ok"]
    token = host._SESSION_ID.set("actual-session")
    try:
        assert json.loads(handlers["planning_with_files_select"]({"plan": "plan-a"}))["ok"]
    finally:
        host._SESSION_ID.reset(token)
    assert "plan-a ONLY" in context("actual-session")
    assert "plan-a ONLY" not in context("spoofed")
    assert "plan-a ONLY" not in context("unrelated-process-session")
    assert json.loads(handlers["planning_with_files_select"]({"plan": "plan-b"}, task_id="fallback"))["ok"]
    assert "plan-b ONLY" in context("fallback")


@pytest.mark.parametrize("bad", ["../plan-a", ".hidden", "bad slug", "missing", "plan-a/task_plan.md"])
def test_invalid_selectors_do_not_change_existing_selection(workspace, bad):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    assert select(handlers, "thread-a", "plan-a")["ok"]
    assert not select(handlers, "thread-a", bad)["ok"]
    assert "plan-a ONLY" in context("thread-a")


def test_explicit_clear_restores_unbound_behavior_and_legacy_plan_id(workspace, monkeypatch):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    monkeypatch.setenv("PLAN_ID", "plan-a")
    assert select(handlers, "thread-b", "plan-b")["ok"]
    assert "plan-b ONLY" in context("thread-b")
    assert select(handlers, "thread-b", "")["ok"]
    assert "plan-a ONLY" in context("thread-b")
    monkeypatch.setenv("PLAN_ID", "missing")
    assert "plan-a ONLY" not in context("unbound")
    assert "plan-b ONLY" not in context("unbound")


@pytest.mark.parametrize("shell", [False, True])
def test_cross_root_selection_beats_process_root_for_status_and_completion(workspace, monkeypatch, tmp_path, shell):
    root, _profile, handlers = workspace
    if shell and tools.shutil.which("sh") is None:
        pytest.skip("sh is unavailable")
    plan(root, "plan-a", "### Phase 1: A\n**Status:** pending\n")
    other = tmp_path / "other-project"
    other.mkdir()
    chosen = plan(other, "plan-b", "### Phase 1: B\n**Status:** complete\n")
    monkeypatch.setenv("PWF_PLAN_ROOT", str(root))
    monkeypatch.setenv("PLAN_ID", "plan-a")
    assert select(handlers, "thread-b", str(chosen))["ok"]
    status = json.loads(handlers["planning_with_files_status"]({}, session_id="thread-b"))
    assert status["project_dir"] == str(other)
    assert status["plan_id"] == "plan-b"
    if not shell:
        monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    complete = json.loads(handlers["planning_with_files_check_complete"]({}, session_id="thread-b"))
    assert complete["ok"] and complete["complete"]
    assert complete["plan_id"] == "plan-b"
    assert complete["plan_dir"] == str(chosen)
    assert complete["route"] == ("sh" if shell else "python")
    assert os.environ["PWF_PLAN_ROOT"] == str(root)
    assert os.environ["PLAN_ID"] == "plan-a"


def test_ambiguous_exact_slug_lists_candidates_and_explicit_cwd_disambiguates(workspace, monkeypatch, tmp_path):
    root, _profile, handlers = workspace
    first = plan(root, "same")
    other = tmp_path / "other-project"
    other.mkdir()
    second = plan(other, "same")
    monkeypatch.setenv("PWF_PLAN_ROOT", str(other))
    refusal = select(handlers, "thread-a", "same")
    assert not refusal["ok"]
    assert set(refusal["candidates"]) == {str(first), str(second)}
    assert select(handlers, "thread-a", "same", cwd=str(other))["ok"]
    status = json.loads(handlers["planning_with_files_status"]({}, session_id="thread-a"))
    assert status["project_dir"] == str(other)


def test_conflicting_explicit_cwd_is_rejected_without_creating_files(workspace, tmp_path):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    other = tmp_path / "other-project"
    other.mkdir()
    assert select(handlers, "thread-a", "plan-a")["ok"]
    init = json.loads(handlers["planning_with_files_init"]({"name": "unexpected", "cwd": str(other)}, session_id="thread-a"))
    assert not init["ok"]
    assert str(root) in init["error"]
    assert "planning_with_files_select" in init["error"]
    assert not list(other.iterdir())
    assert not list((root / ".planning").glob("*-unexpected"))
    status = json.loads(handlers["planning_with_files_status"]({"cwd": str(other)}, session_id="thread-a"))
    assert not status["exists"]
    complete = json.loads(handlers["planning_with_files_check_complete"]({"cwd": str(other)}, session_id="thread-a"))
    assert not complete["ok"]


def test_successful_read_binds_only_exact_task_plan_and_only_unbound_session(workspace):
    root, _profile, handlers = workspace
    a, b = plan(root, "plan-a"), plan(root, "plan-b")
    hooks.post_tool_call(tool_name="read_file", args={"path": str(a / "task_plan.md")}, result={"content": "# plan"}, status="ok", session_id="reader")
    assert "plan-a ONLY" in context("reader")
    hooks.post_tool_call(tool_name="read_file", args={"path": str(b / "task_plan.md")}, result=json.dumps({"content": "# reference"}), status="ok", session_id="reader")
    assert "plan-a ONLY" in context("reader")
    nested = b / "examples"
    nested.mkdir()
    (nested / "task_plan.md").write_text("# nested sample\n", encoding="ascii")
    hooks.post_tool_call(tool_name="read_file", args={"path": str(nested / "task_plan.md")}, result={"content": "# nested sample"}, status="ok", session_id="unbound")
    assert "Multiple plans" in context("unbound")


def test_successful_read_normalizes_dot_segments_after_no_link_validation(workspace):
    root, _profile, _handlers = workspace
    a, _b = plan(root, "plan-a"), plan(root, "plan-b")
    path = a / ".." / "plan-b" / "task_plan.md"
    hooks.post_tool_call(tool_name="read_file", args={"path": str(path)}, result={"content": "# plan"}, status="ok", session_id="reader")
    assert "plan-b ONLY" in context("reader")


def test_registered_commands_use_actual_host_session_context(workspace, monkeypatch):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    plan(root, "plan-b")
    host = types.ModuleType("gateway.session_context")
    host._SESSION_ID = ContextVar("test_command_session_id", default="")
    monkeypatch.setitem(sys.modules, "gateway", types.ModuleType("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.session_context", host)
    assert select(handlers, "command-session", "plan-b")["ok"]
    commands = {}

    class Context:
        def register_tool(self, **kw):
            pass

        def register_hook(self, *args, **kw):
            pass

        def register_command(self, name, handler, **kw):
            commands[name] = handler

    plugin.register(Context())
    token = host._SESSION_ID.set("command-session")
    try:
        report = commands["pwf-status"]("")
    finally:
        host._SESSION_ID.reset(token)
    assert "plan-b" in report
    assert "Multiple plans" not in report


@pytest.mark.parametrize("operation", ["write_file", "replace", "v4a", "resolved"])
def test_successful_writes_and_patches_switch_to_unambiguous_plan(workspace, operation):
    root, _profile, handlers = workspace
    plan(root, "plan-a")
    b = plan(root, "plan-b")
    target = b / "progress.md"
    target.write_text("changed\n", encoding="ascii")
    assert select(handlers, "writer", "plan-a")["ok"]
    tool_name = "write_file" if operation in {"write_file", "resolved"} else "patch"
    args = {"path": str(target), "content": "changed"} if tool_name == "write_file" else {"path": str(target), "old_string": "old", "new_string": "new"}
    result = {"bytes_written": 7} if tool_name == "write_file" else {"success": True}
    if operation == "v4a":
        args = {"mode": "patch", "patch": f"*** Begin Patch\n*** Update File: {target}\n@@\n-old\n+new\n*** End Patch"}
    if operation == "resolved":
        args["path"] = "nonexistent-argument-path"
        result["resolved_path"] = str(target)
        result["files_modified"] = [str(target)]
    hooks.post_tool_call(tool_name=tool_name, args=args, result=json.dumps(result), status="ok", session_id="writer")
    assert "plan-b ONLY" in context("writer")
    assert "Update progress.md" not in context("different-session")


@pytest.mark.parametrize("result,status", [
    ({"error": "denied", "resolved_path": "ignored"}, "ok"),
    ({"success": False, "note": "permission denied"}, "ok"),
    ({"content": "would look successful"}, "error"),
    ({"not_found": True, "content": ""}, "ok"),
    ({}, "ok"),
    ("not json", "ok"),
])
def test_unsuccessful_or_unverifiable_file_reads_never_bind(workspace, result, status):
    root, _profile, _handlers = workspace
    a = plan(root, "plan-a")
    plan(root, "plan-b")
    hooks.post_tool_call(tool_name="read_file", args={"path": str(a / "task_plan.md")}, result=result, status=status, session_id="reader")
    assert "Multiple plans" in context("reader")


def test_multi_plan_patch_is_not_a_selection(workspace):
    root, _profile, _handlers = workspace
    a, b = plan(root, "plan-a"), plan(root, "plan-b")
    hooks.post_tool_call(tool_name="patch", args={"patch": "*** Begin Patch"}, result={"success": True, "files_modified": [str(a / "task_plan.md"), str(b / "task_plan.md")]}, status="ok", session_id="writer")
    assert "Multiple plans" in context("writer")


def test_multi_plan_patch_counts_deleted_effects_before_validating_targets(workspace):
    root, _profile, handlers = workspace
    a, b = plan(root, "plan-a"), plan(root, "plan-b")
    plan(root, "plan-c")
    assert select(handlers, "writer", "plan-c")["ok"]
    (a / "task_plan.md").unlink()
    hooks.post_tool_call(tool_name="patch", args={"patch": "*** Begin Patch"}, result={"success": True, "files_modified": [str(a / "task_plan.md"), str(b / "task_plan.md")], "files_deleted": [str(a / "task_plan.md")]}, status="ok", session_id="writer")
    assert "plan-c ONLY" in context("writer")


def test_stale_binding_cannot_be_replaced_by_automatic_read_or_write(workspace):
    root, _profile, handlers = workspace
    a, b = plan(root, "plan-a"), plan(root, "plan-b")
    assert select(handlers, "writer", "plan-a")["ok"]
    (a / "task_plan.md").unlink()
    hooks.post_tool_call(tool_name="read_file", args={"path": str(b / "task_plan.md")}, result={"content": "# reference"}, status="ok", session_id="writer")
    hooks.post_tool_call(tool_name="write_file", args={"path": str(b / "task_plan.md"), "content": "# reference"}, result={"bytes_written": 11, "files_modified": [str(b / "task_plan.md")]}, status="ok", session_id="writer")
    assert "plan-b ONLY" not in context("writer")


def test_corrupted_persistence_fails_closed_until_explicit_clear(workspace):
    root, profile, handlers = workspace
    plan(root, "plan-a")
    assert select(handlers, "reader", "plan-a")["ok"]
    record = next((profile / "planning-with-files" / "sessions").glob("*.json"))
    record.write_text("not json", encoding="ascii")
    assert "plan-a ONLY" not in context("reader")
    assert select(handlers, "reader", "")["ok"]
    assert "plan-a ONLY" in context("reader")


def test_session_attachment_is_admission_at_bound_project(workspace):
    root, _profile, handlers = workspace
    a = plan(root, "plan-a")
    plan(root, "plan-b")
    attachment_dir = root / ".planning" / "sessions"
    attachment_dir.mkdir()
    assert not select(handlers, "detached", "plan-a")["ok"]
    state = importlib.import_module(SPEC.name + ".hook_state")
    sentinel = attachment_dir / f"{state.state_key(root, 'attached')}.attached"
    sentinel.write_text("attached\n", encoding="ascii")
    assert select(handlers, "attached", "plan-a")["ok"]
    assert "plan-a ONLY" in context("attached")
    sentinel.unlink()
    assert "plan-a ONLY" not in context("attached")
    status = json.loads(handlers["planning_with_files_status"]({}, session_id="attached"))
    assert not status["exists"]
    hooks.post_tool_call(tool_name="read_file", args={"path": str(a / "task_plan.md")}, result={"content": "# plan"}, status="ok", session_id="detached")
    assert context("detached") == ""


def test_disabled_hooks_do_not_bind_inject_remind_or_gate(workspace, monkeypatch):
    root, profile, _handlers = workspace
    a = plan(root, "plan-a")
    plan(root, "plan-b")
    monkeypatch.setenv("PLANNING_DISABLED", "1")
    hooks.post_tool_call(tool_name="read_file", args={"path": str(a / "task_plan.md")}, result={"content": "# plan"}, status="ok", session_id="reader")
    assert context("reader") == ""
    assert hooks.pre_verify(session_id="reader") is None
    assert not (profile / "planning-with-files").exists()


def test_bound_attestation_is_still_verified(workspace):
    root, _profile, handlers = workspace
    a = plan(root, "plan-a")
    plan(root, "plan-b")
    (a / ".mode").write_text("autonomous\n", encoding="ascii")
    (a / ".attestation").write_text("0" * 64, encoding="ascii")
    assert select(handlers, "reader", "plan-a")["ok"]
    text = context("reader")
    assert "PLAN TAMPERED" in text
    assert "plan-a ONLY" not in text


def test_linked_plan_or_task_file_is_never_selected(workspace, tmp_path):
    root, _profile, handlers = workspace
    a = plan(root, "plan-a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task_plan.md").write_text("# outside\n", encoding="ascii")
    link = root / ".planning" / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    assert not select(handlers, "reader", str(link))["ok"]
    (a / "task_plan.md").unlink()
    (a / "task_plan.md").symlink_to(outside / "task_plan.md")
    assert not select(handlers, "reader", "plan-a")["ok"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_junction_project_is_never_selected(workspace, tmp_path):
    _root, _profile, handlers = workspace
    outside = tmp_path / "outside"
    outside.mkdir()
    plan(outside, "plan-a")
    junction = tmp_path / "linked-project"
    made = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True, text=True, check=False)
    if made.returncode:
        pytest.skip("junction creation unavailable")
    assert not select(handlers, "reader", str(junction / ".planning" / "plan-a"))["ok"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_junction_storage_parent_is_not_followed_or_written(workspace, tmp_path):
    root, profile, handlers = workspace
    plan(root, "plan-a")
    outside = tmp_path / "outside-state"
    outside.mkdir()
    junction = profile / "planning-with-files"
    made = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True, text=True, check=False)
    if made.returncode:
        pytest.skip("junction creation unavailable")
    assert not select(handlers, "reader", str(root / ".planning" / "plan-a"))["ok"]
    assert not list(outside.iterdir())
    assert "plan-a ONLY" not in context("reader")
