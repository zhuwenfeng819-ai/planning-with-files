"""Request-local plan selectors with one atomic, profile-local record per session.

Hermes supplies identity in hook/tool kwargs, or its session ContextVar for
command dispatch. Tool arguments never supply identity. No process environment
is changed, and no shared active-plan pointer identifies a conversation.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

from .context_frame import _assert_no_reparse_components, read_regular_bytes


@dataclass(frozen=True)
class Binding:
    project_dir: Path | None
    plan_id: str


_BINDING: ContextVar[Binding | None] = ContextVar("pwf_hermes_binding", default=None)
_SESSION: ContextVar[str] = ContextVar("pwf_hermes_session", default="")
_INVALID = Binding(None, "")


def session_id(kwargs: dict[str, Any]) -> str:
    """Only host metadata, never the model-controlled ``args`` dictionary."""
    explicit = kwargs.get("session_id") or kwargs.get("task_id")
    if explicit:
        return str(explicit)
    try:
        from gateway.session_context import _SESSION_ID

        # The public get_session_env falls back to process os.environ when
        # never bound. That value can belong to a different conversation.
        value = _SESSION_ID.get()
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
        return ""
    return value if isinstance(value, str) else ""


def current_binding() -> Binding | None:
    return _BINDING.get()


def current_session_id() -> str:
    return _SESSION.get()


def _record_path(identity: str, *, create: bool = False) -> Path:
    # get_hermes_home honors Hermes' active profile. Never guess a user's home
    # or accept a storage directory from model-controlled tool arguments.
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    if not home.is_absolute():
        raise ValueError("Hermes profile directory must be absolute")
    _assert_no_reparse_components(home)
    directory = home / "planning-with-files" / "sessions"
    # Validate existing parents even when the final directory is absent.
    for candidate in (directory.parent, directory):
        if create and not candidate.exists() and not candidate.is_symlink():
            candidate.mkdir(exist_ok=True, mode=0o700)
        if candidate.exists() or candidate.is_symlink():
            _assert_no_reparse_components(candidate)
            if not candidate.is_dir():
                raise ValueError("unsafe Hermes planning state directory")
    digest = hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()
    return directory / f"{digest}.json"


def _load(identity: str) -> Binding | None:
    if not identity:
        return None
    try:
        path = _record_path(identity)
        data = json.loads(read_regular_bytes(path, max_source_bytes=16384))
        if not isinstance(data, dict) or data.get("version") != 1:
            return _INVALID
        root, slug = data.get("project_dir"), data.get("plan_id")
        if not isinstance(root, str) or not isinstance(slug, str):
            return _INVALID
        project = Path(root)
        if not project.is_absolute() or not slug:
            return _INVALID
        # A stale or invalid persisted selection is still binding. The resolver
        # refuses it; loading must never silently remove it and select a peer.
        return Binding(project, slug)
    except FileNotFoundError:
        return None
    except ImportError:
        return None  # a host without Hermes profile APIs has no stored binding
    except (OSError, RuntimeError, ValueError, UnicodeError):
        return _INVALID


@contextmanager
def session_scope(kwargs: dict[str, Any]) -> Iterator[None]:
    identity = session_id(kwargs)
    session_token = _SESSION.set(identity)
    binding_token = _BINDING.set(_load(identity))
    try:
        yield
    finally:
        _BINDING.reset(binding_token)
        _SESSION.reset(session_token)


def scoped_hook(callback):
    @wraps(callback)
    def wrapped(**kwargs):
        with session_scope(kwargs):
            return callback(**kwargs)

    return wrapped


def scoped_tool(callback):
    def wrapped(args, **kwargs):
        with session_scope(kwargs):
            return callback(args)

    return wrapped


def scoped_command(callback):
    @wraps(callback)
    def wrapped(raw_args=""):
        with session_scope({}):
            return callback(raw_args)

    return wrapped


def binding_root(binding: Binding) -> Path | None:
    project = binding.project_dir
    if project is None:
        return None
    try:
        if str(project).startswith(("\\\\", "//")):
            return None
        _assert_no_reparse_components(project)
        if not project.is_dir():
            return None
        return project.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None


def bind(project_dir: Path, plan_id: str) -> str | None:
    """Persist a previously validated target; return an error on any failure."""
    identity = current_session_id()
    if not identity:
        return "Hermes did not supply a session identity; no session selection was saved."
    temp: Path | None = None
    fd: int | None = None
    try:
        path = _record_path(identity, create=True)
        if path.exists() or path.is_symlink():
            read_regular_bytes(path, max_source_bytes=16384)
        temp = path.with_name(f".{path.stem}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            json.dump({"version": 1, "project_dir": str(project_dir), "plan_id": plan_id}, handle)
            handle.write("\n")
        _assert_no_reparse_components(path.parent)
        os.replace(temp, path)
        _BINDING.set(Binding(project_dir, plan_id))
        return None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return f"Could not save Hermes session selection ({exc.__class__.__name__})."
    finally:
        if fd is not None:
            os.close(fd)
        if temp is not None:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def clear() -> str | None:
    identity = current_session_id()
    if not identity:
        return "Hermes did not supply a session identity; no session selection was cleared."
    try:
        path = _record_path(identity)
        if path.exists() or path.is_symlink():
            read_regular_bytes(path, max_source_bytes=16384)
            path.unlink()
        _BINDING.set(None)
        return None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return f"Could not clear Hermes session selection ({exc.__class__.__name__})."
