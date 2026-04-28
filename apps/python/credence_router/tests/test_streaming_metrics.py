# Role: body
"""Tests that streaming metrics are recorded even when the client disconnects.

The proxy uses async generators to stream SSE responses. When clients close the
connection after data: [DONE], the generator is abandoned (GeneratorExit). These
tests verify that observation data and request metrics survive that scenario.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from credence_router.routing_domain import Observation


# ---------------------------------------------------------------------------
# forward_streaming: obs_out populated on aclose (simulated client disconnect)
# ---------------------------------------------------------------------------


def test_forward_streaming_obs_out_on_aclose():
    """forward_streaming stores observation in obs_out even when generator is
    abandoned via aclose() (simulates client disconnect)."""

    async def _run():
        from credence_router.tools.llm.provider import forward_streaming

        obs_holder: dict = {}

        fake_chunks = [
            b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]

        class FakeResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in fake_chunks:
                    yield chunk

            async def aread(self):
                return b""

            async def aclose(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, *args, **kwargs):
                return FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("credence_router.tools.llm.provider.httpx.AsyncClient", return_value=FakeClient()):
            gen = forward_streaming(
                b'{"model":"auto","messages":[{"role":"user","content":"hi"}]}',
                "claude-haiku-4-5",
                obs_out=obs_holder,
            )

            chunk, obs = await gen.__anext__()
            assert obs is None
            await gen.aclose()

        assert "observation" in obs_holder
        obs = obs_holder["observation"]
        assert isinstance(obs, Observation)

    asyncio.run(_run())


def test_forward_streaming_obs_out_on_normal_completion():
    """forward_streaming stores observation in obs_out on normal completion too."""

    async def _run():
        from credence_router.tools.llm.provider import forward_streaming

        obs_holder: dict = {}

        fake_chunks = [
            b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3}}\n\n',
            b"data: [DONE]\n\n",
        ]

        class FakeResponse:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in fake_chunks:
                    yield chunk

            async def aread(self):
                return b""

            async def aclose(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, *args, **kwargs):
                return FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("credence_router.tools.llm.provider.httpx.AsyncClient", return_value=FakeClient()):
            gen = forward_streaming(
                b'{"model":"auto","messages":[{"role":"user","content":"hi"}]}',
                "claude-haiku-4-5",
                obs_out=obs_holder,
            )
            async for _chunk, _obs in gen:
                pass

        assert "observation" in obs_holder
        obs = obs_holder["observation"]
        assert obs.input_tokens == 8
        assert obs.output_tokens == 3
        assert obs.completed
        assert obs.response_text == "world"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# server generate() + BackgroundTask: _request_log populated after response
# ---------------------------------------------------------------------------


def test_request_log_populated_via_background_task():
    """BackgroundTask appends to _request_log after response completes."""
    import credence_router.server as srv

    old_log = srv._request_log.copy()
    old_domain = srv._llm_domain
    old_state_path = srv._llm_state_path
    old_brain = srv._brain
    srv._request_log.clear()
    srv._llm_state_path = None

    fake_obs = Observation(
        completed=True,
        input_tokens=42,
        output_tokens=17,
        cost_usd=0.001,
        ttft_seconds=0.5,
        total_seconds=2.0,
    )

    try:
        from starlette.testclient import TestClient
        from credence_router.server import app

        mock_domain = type("MockDomain", (), {
            "route": lambda self, msg: type("D", (), {"provider_name": "claude-haiku-4-5"})(),
            "queue_outcome": lambda self, obs: None,
            "provider_names": ["claude-haiku-4-5"],
            "load_state": lambda self, path: None,
        })()

        async def fake_forward(body, model, obs_out=None):
            if obs_out is not None:
                obs_out["observation"] = fake_obs
            yield b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n", None
            yield b"data: [DONE]\n\n", None
            yield b"", fake_obs

        with (
            patch("credence_router.server._init_llm_domain"),
            patch("credence_router.tools.llm.provider.forward_streaming", fake_forward),
            patch.dict("os.environ", {"CREDENCE_FORCE_MODEL": "claude-haiku-4-5"}, clear=False),
        ):
            srv._llm_domain = mock_domain
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "auto", "messages": [{"role": "user", "content": "test"}]},
                )
                assert resp.status_code == 200

        assert len(srv._request_log) == 1
        record = srv._request_log[0]
        assert record["input_tokens"] == 42
        assert record["output_tokens"] == 17
        assert record["cost_usd"] == 0.001
        assert record["model_selected"] == "claude-haiku-4-5"

    finally:
        srv._request_log.clear()
        srv._request_log.extend(old_log)
        srv._llm_domain = old_domain
        srv._llm_state_path = old_state_path
        srv._brain = old_brain
