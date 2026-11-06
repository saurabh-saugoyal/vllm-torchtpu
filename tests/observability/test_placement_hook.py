# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""API-server hook: build_app wrapping and lifespan ordering."""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_torchtpu.observability import placement_hook as hook


def test_lifespan_runs_emitter_before_vllm_lifespan():
    events = []

    @contextlib.asynccontextmanager
    async def vllm_lifespan(app):
        events.append("vllm-startup")
        yield
        events.append("vllm-shutdown")

    async def publish(engine_client, vllm_config):
        events.append(("publish", engine_client, vllm_config))

    wrapped = hook.wrap_build_app(lambda: FastAPI(lifespan=vllm_lifespan))
    assert hook.wrap_build_app(wrapped) is wrapped
    app = wrapped()
    app.state.engine_client, app.state.vllm_config = "client", "config"
    with patch.object(hook, "publish_placement", publish), TestClient(app):
        pass
    assert events == [("publish", "client", "config"), "vllm-startup", "vllm-shutdown"]


def test_registry_callback_retries_until_launcher_imported(monkeypatch):
    for name in hook._BUILD_APP_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(hook.envs, "VLLM_TPU_EMIT_PLACEMENT", True)
    assert hook._patch_api_server_placement_hook() is False

    entry = types.ModuleType(hook._BUILD_APP_MODULES[0])
    original = entry.build_app = lambda: "app"
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    assert hook._patch_api_server_placement_hook() is None
    assert entry.build_app.__wrapped__ is original
    wrapped = entry.build_app
    assert hook._patch_api_server_placement_hook() is False  # already wrapped
    assert entry.build_app is wrapped


def test_registry_callback_disabled_by_env(monkeypatch):
    entry = types.ModuleType(hook._BUILD_APP_MODULES[0])
    entry.build_app = lambda: "app"
    monkeypatch.setitem(sys.modules, entry.__name__, entry)
    monkeypatch.setattr(hook.envs, "VLLM_TPU_EMIT_PLACEMENT", False)
    assert hook._patch_api_server_placement_hook() is None
    assert not hasattr(entry.build_app, "__wrapped__")


def test_lifespan_hook_uses_app_state():
    seen = SimpleNamespace()

    async def publish(engine_client, vllm_config):
        seen.args = (engine_client, vllm_config)

    app = FastAPI()
    hook.install_lifespan_hook(app)
    app.state.engine_client, app.state.vllm_config = object(), object()
    with patch.object(hook, "publish_placement", publish), TestClient(app):
        pass
    assert seen.args == (app.state.engine_client, app.state.vllm_config)
