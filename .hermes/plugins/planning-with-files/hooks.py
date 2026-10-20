from __future__ import annotations

import os
import json
import re
import stat
from pathlib import Path
from typing import Any

from .constants import PROGRESS_TAIL_LINES, READ_PREVIEW_LINES
from .context_frame import frame_bytes, read_regular_bytes, select_lines, verified_frame
from .hook_state import add_reminder, pop_reminders, state_key
from .paths import (
    MULTIPLE_PLANS_NOTICE,
    ambiguity_notice,
    attestation_path_for,
    effective_project_root,
    multiple_plans_require_selector,
    normalize_cwd,
    plan_id_for,
    plan_root_is_pinned,
    resolve_plan,
    resolve_plan_dir,
)
from .planning_files import evaluate_gate, normalize_wall_clock
from .session_binding import bind, binding_root, current_binding, current_session_id, scoped_hook
from .tools import validated_plan_path

ATTACH_LEGACY = "legacy"      # no .planning/sessions directory: single-session setup
ATTACH_ATTACHED = "attached"  # sessions directory armed and this session holds a sentinel
ATTACH_DETACHED = "detached"  # sessions directory armed, no sentinel (or unsafe directory)
def _runtime_project_dir(kwargs: dict[str, Any]) -> Path | None:
    """Resolve the active Hermes project, never a cached import-time cwd.

    Honors the per-invocation opt-out (PLANNING_DISABLED=1, issue #195) and
    the absolute plan-root pin (PWF_PLAN_ROOT, issue #212); a broken pin
    fails closed so no unrelated plan is injected.
    """
    if os.environ.get("PLANNING_DISABLED", "") == "1":
        return None
    try:
        from agent.runtime_cwd import resolve_agent_cwd
    except ImportError:
        return None
    else:
        try:
            candidate = resolve_agent_cwd()
        except (OSError, RuntimeError, ValueError):
            return None
    if not isinstance(candidate, Path):
        return None
    try:
        project = normalize_cwd(str(candidate))
        if not project.is_dir():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return effective_project_root(project)


WORKTREES_DIR_NAME = ".worktrees"  # hermes -w and /worktree new place trees at <repo>/.worktrees/<name>


def launch_dir_has_planning_state(launch: Path) -> bool:
    """True when the process launch directory holds a plan, or several plans.

    A plan with a nested-root conflict still counts (``explicit=True`` skips
    only that check), because the question here is whether the directory the
    session was started in carries planning state at all.
    """
    return (
        resolve_plan(launch, explicit=True)[0] is not None
        or multiple_plans_require_selector(launch)
    )


def launch_dir_notice(project_dir: Path, launch: Path) -> str:
    """The once-per-turn line for issue #272; the resolved root and the launch directory differ."""
    return (
        f"[planning-with-files] Hermes resolved {project_dir} as the working directory, "
        f"but the launch directory {launch} holds a plan. Nothing injected. "
        f"Pin the thread with PWF_PLAN_ROOT={launch}."
    )


def _launch_dir_notice(project_dir: Path, kwargs: dict[str, Any]) -> str | None:
    """Issue #272: the resolved root holds no plan while the launch directory does.

    Hermes 0.21.3 rewrites the process-global TERMINAL_CWD during the first
    turn of a CLI session (``agent/relay_runtime.py`` imports ``gateway/run.py``
    lazily; its import-time bridge applies the ``Path.home()`` fallback), so
    ``resolve_agent_cwd`` names the home directory while ``os.getcwd()`` still
    names the directory the session was started in (upstream hermes-agent
    #86411 and #95577). The rewrite runs before the first hook, so the adapter
    cannot recover the root on its own and never switches to the launch
    directory: it says why nothing was injected and names the pin.

    Silent by design: platforms other than the CLI (their process cwd is not a
    launch directory), Hermes worktree sessions (``hermes -w`` points
    TERMINAL_CWD at ``<repo>/.worktrees/<name>`` without a chdir, which is the
    intended split), a launch directory without planning state, and a launch
    directory whose session isolation refuses this session anyway.
    """
    if str(kwargs.get("platform", "")).strip().lower() != "cli":
        return None
    if WORKTREES_DIR_NAME in project_dir.parts:
        return None
    try:
        launch = normalize_cwd(os.getcwd())
        if not launch.is_dir() or launch == project_dir:
            return None
        if _session_attachment(launch, _session_id(kwargs)) == ATTACH_DETACHED:
            return None
        if not launch_dir_has_planning_state(launch):
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return launch_dir_notice(project_dir, launch)


def _session_attachment(project_dir: Path, session_id: str) -> str:
    """Opt-in isolation state once a project creates its sessions directory."""
    sessions_dir = project_dir / ".planning" / "sessions"
    try:
        info = sessions_dir.lstat()
    except FileNotFoundError:
        return ATTACH_LEGACY
    except OSError:
        return ATTACH_DETACHED
    attrs = getattr(info, "st_file_attributes", 0)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or not session_id
    ):
        return ATTACH_DETACHED
    try:
        sentinel = sessions_dir / f"{state_key(project_dir, session_id)}.attached"
        read_regular_bytes(sentinel, max_source_bytes=32)
        return ATTACH_ATTACHED
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return ATTACH_DETACHED


def _resolve_plan(project_dir: Path, *, explicit: bool = False) -> tuple[Path | None, list[str]]:
    """Active plan directory whose task_plan.md is a regular file, plus nested conflicts."""
    plan_dir, conflicts = resolve_plan(project_dir, explicit=explicit)
    if plan_dir is None:
        return None, conflicts
    try:
        read_regular_bytes(plan_dir / "task_plan.md", max_source_bytes=4 * 1024 * 1024)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None, []
    return plan_dir, []


def _session_id(kwargs: dict[str, Any]) -> str:
    """Use Hermes' session identity, with its documented tool task fallback."""
    return current_session_id() or str(kwargs.get("session_id") or kwargs.get("task_id") or "")


def _locate(
    kwargs: dict[str, Any],
) -> tuple[Path | None, Path | None, list[str], str, bool]:
    """Shared front half: project, plan, conflicts, session id, selector refusal.

    PWF_PLAN_ROOT selects the effective project root. PLAN_ID selects a plan
    within it. A session attachment is admission only and never skips either
    same-root binding or nested-root ambiguity checks.
    """
    project_dir = _runtime_project_dir(kwargs)
    if project_dir is None:
        return None, None, [], "", False
    session_id = _session_id(kwargs)
    attachment = _session_attachment(project_dir, session_id)
    if attachment == ATTACH_DETACHED:
        return project_dir, None, [], session_id, False
    if multiple_plans_require_selector(project_dir):
        return project_dir, None, [], session_id, True
    explicit = plan_root_is_pinned()
    plan_dir, conflicts = _resolve_plan(project_dir, explicit=explicit)
    return project_dir, plan_dir, conflicts, session_id, False


def build_user_prompt_context(project_dir: Path, plan_dir: Path | None = None) -> str:
    """Frame the active plan for injection.

    Legacy root plans produce the exact bytes previous releases produced.
    Slug plans (``.planning/<id>/``) add one line naming the resolved plan so
    slug-over-root shadowing stays visible, read their attestation from
    ``<id>/.attestation`` and their mode marker from ``<id>/.mode``.
    """
    if plan_dir is None:
        plan_dir = resolve_plan_dir(project_dir) or project_dir
    task_plan = plan_dir / "task_plan.md"
    parts = ["[planning-with-files] ACTIVE PLAN — current state:"]
    if plan_dir != project_dir:
        parts.append(f"[planning-with-files] plan: {plan_id_for(project_dir, plan_dir)}")
    try:
        parts.append(
            verified_frame(
                "plan",
                task_plan,
                attestation=attestation_path_for(project_dir, plan_dir),
                mode=plan_dir / ".mode",
                head=READ_PREVIEW_LINES,
            )
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return f"[planning-with-files] context blocked: {exc}"
    try:
        progress_bytes = select_lines(
            read_regular_bytes(plan_dir / "progress.md"), tail=PROGRESS_TAIL_LINES
        )
        normalized = normalize_wall_clock(
            progress_bytes.decode("utf-8", errors="replace")
        ).encode("utf-8")
        parts.append(frame_bytes("progress", normalized))
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        pass
    try:
        read_regular_bytes(plan_dir / "findings.md", max_source_bytes=4 * 1024 * 1024)
        parts.append("[planning-with-files] Read findings.md for research context. Continue from the current phase.")
    except (FileNotFoundError, OSError, ValueError):
        pass
    return "\n\n".join(parts)


@scoped_hook
def pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    project_dir, plan_dir, conflicts, session_id, selector_required = _locate(kwargs)
    if project_dir is None:
        return None
    if plan_dir is None:
        # The refusal holds in every hook; the notice is turn-scoped, so only
        # this once-per-turn hook says why (same split as inject-plan.sh).
        if selector_required:
            return {"context": MULTIPLE_PLANS_NOTICE}
        if conflicts:
            return {"context": ambiguity_notice(conflicts)}
        if not plan_root_is_pinned():
            notice = _launch_dir_notice(project_dir, kwargs)
            if notice:
                return {"context": notice}
        return None
    user_message = str(kwargs.get("user_message", ""))
    reminder_messages = pop_reminders(project_dir, session_id)
    context = build_user_prompt_context(project_dir, plan_dir)
    parts: list[str] = []
    if reminder_messages:
        parts.append("\n".join(reminder_messages))
    if context:
        parts.append(context)
    if not user_message.strip() and not kwargs.get("is_first_turn") and not reminder_messages:
        return None
    if not parts:
        return None
    return {"context": "\n\n".join(parts)}


def _successful_file_result(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    if kwargs.get("status", "ok") != "ok" or kwargs.get("error_type") or kwargs.get("error_message"):
        return None
    result = kwargs.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return None
    if not isinstance(result, dict) or result.get("error") or result.get("success") is False or result.get("not_found"):
        return None
    if tool_name == "read_file":
        return result if isinstance(result.get("content"), str) or result.get("status") == "unchanged" else None
    if tool_name == "write_file":
        return result if isinstance(result.get("bytes_written"), int) or result.get("success") is True else None
    if tool_name == "patch":
        return result if result.get("success") is True and not result.get("no_change") else None
    return None


def _auto_bind(tool_name: str, args: dict[str, Any], kwargs: dict[str, Any]) -> None:
    if not current_session_id() or os.environ.get("PLANNING_DISABLED", "") == "1":
        return
    existing = current_binding()
    if existing is not None:
        root = binding_root(existing)
        if tool_name == "read_file" or root is None:
            return
        if validated_plan_path(root / ".planning" / existing.plan_id) is None:
            return  # stale/corrupt bindings need an explicit select or clear
    result = _successful_file_result(tool_name, kwargs)
    if result is None:
        return
    paths = []
    for key in ("files_modified", "files_created", "files_deleted"):
        if isinstance(result.get(key), list):
            paths.extend(result[key])
    if not paths:
        path = result.get("resolved_path") or args.get("path")
        paths = [path] if path else []
        if not paths and tool_name == "patch" and isinstance(args.get("patch"), str):
            # Hermes' V4A parser accepts these anchored headers. Prefer its
            # resolved files_modified list above; never parse arbitrary prose.
            for line in args["patch"].splitlines():
                match = re.match(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", line)
                if match:
                    paths.append(match.group(1).strip())
                move = re.match(r"^\*\*\* Move File: (.+?) -> (.+)$", line)
                if move:
                    paths.extend(move.groups())
    targets: set[tuple[Path, str]] = set()
    # Relative file paths are resolved only against Hermes' runtime cwd, not a
    # previous plan binding or the backend's launch directory.
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        runtime = resolve_agent_cwd()
    except (ImportError, OSError, RuntimeError, ValueError):
        runtime = None
    for value in paths:
        if not isinstance(value, str) or not value:
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            if not isinstance(runtime, Path):
                return
            path = runtime / path
        try:
            from .context_frame import _assert_no_reparse_components

            _assert_no_reparse_components(path.parent)
            if path.exists() or path.is_symlink():
                _assert_no_reparse_components(path)
                if not path.is_file():
                    continue
            elif tool_name != "patch":
                continue
            # A successful patch may have deleted a file. Its plan identity
            # still competes with every other effect before target validation.
            path = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if tool_name == "read_file" and path.name != "task_plan.md":
            continue
        for directory in path.parents:
            if directory.parent.name == ".planning":
                if tool_name == "read_file" and path.parent != directory:
                    break
                targets.add((directory.parent.parent, directory.name))
                break
    if len(targets) != 1:
        return
    project, slug = targets.pop()
    if (validated_plan_path(project / ".planning" / slug) is not None
            and _session_attachment(project, current_session_id()) != ATTACH_DETACHED):
        bind(project, slug)


@scoped_hook
def post_tool_call(**kwargs: Any) -> None:
    tool_name = str(kwargs.get("tool_name", ""))
    args = kwargs.get("args") or {}
    if not isinstance(args, dict):
        return None
    _auto_bind(tool_name, args, kwargs)
    if "result" in kwargs and _successful_file_result(tool_name, kwargs) is None:
        return None
    if tool_name == "write_file":
        if not args.get("path") or "content" not in args:
            return None
    elif tool_name == "patch":
        has_patch_payload = bool(args.get("patch"))
        has_replace_payload = bool(args.get("path")) and "old_string" in args and "new_string" in args
        if not (has_patch_payload or has_replace_payload):
            return None
    else:
        return None
    project_dir, plan_dir, _conflicts, session_id, _selector_required = _locate(kwargs)
    if project_dir is None or plan_dir is None:
        return None
    message = "[planning-with-files] Update progress.md with what you just did. If a phase is now complete, update task_plan.md status."
    add_reminder(project_dir, session_id, message)
    return None


@scoped_hook
def pre_verify(**kwargs: Any) -> dict[str, str] | None:
    """Completion gate on Hermes' verification-loop hook (gated mode only).

    Hermes fires ``pre_verify`` once per turn when the agent edited files and
    is about to finish, and bounds continuations by ``agent.max_verify_nudges``
    (default 3). The gate decision table lives in ``evaluate_gate``; every
    plan without the ``gate`` token stays advisory and this hook returns None,
    so legacy and autonomous plans never hold a turn open.
    """
    project_dir, plan_dir, _conflicts, _session_id, _selector_required = _locate(kwargs)
    if project_dir is None or plan_dir is None:
        return None
    try:
        reason = evaluate_gate(project_dir, plan_dir)
    except (OSError, RuntimeError, ValueError):
        return None
    if not reason:
        return None
    return {"action": "continue", "message": reason}
