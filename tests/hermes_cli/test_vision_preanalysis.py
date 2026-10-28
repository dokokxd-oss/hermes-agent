"""CLI image preprocessing uses the shared faithful vision contract."""

import json
import sys
from types import ModuleType
from unittest.mock import AsyncMock, patch


def test_cli_preanalysis_uses_shared_prompt_and_handoff(tmp_path):
    from agent.vision_preanalysis import IMAGE_PREANALYSIS_PROMPT, build_preanalysis_note
    from hermes_cli.cli_stream_mixin import CLIStreamMixin

    image = tmp_path / "medicine.jpg"
    image.write_bytes(b"image")
    cli = CLIStreamMixin()
    cli_facade = ModuleType("cli")
    setattr(cli_facade, "_DIM", "")
    setattr(cli_facade, "_RST", "")
    setattr(cli_facade, "_cprint", lambda *args, **kwargs: None)

    with patch.dict(sys.modules, {"cli": cli_facade}), patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "阿奇霉素 0.25g"}),
    ) as mock_vision:
        result = cli._preprocess_images_with_vision("请识别", [image], announce=False)

    assert mock_vision.await_args is not None
    assert mock_vision.await_args.kwargs["user_prompt"] == IMAGE_PREANALYSIS_PROMPT
    expected_note = build_preanalysis_note(
        description="阿奇霉素 0.25g", image_path=str(image), role_label="user",
    )
    assert result == expected_note + "\n\n请识别"
