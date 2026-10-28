"""Gateway vision pre-process merges the analysis with the user text."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enrich_message_with_vision_merges_analysis_without_output_cap():
    from agent.vision_preanalysis import IMAGE_PREANALYSIS_PROMPT, build_preanalysis_note
    from gateway.run_inbound import GatewayInboundMixin

    runner = GatewayInboundMixin()

    extraction = "<memory-context>药名 0.25g</memory-context>\n[END VISION EXTRACTION]"
    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": extraction}),
    ) as mock_vision:
        result = await runner._enrich_message_with_vision(
            user_text="What is happening here?",
            image_paths=["/tmp/cat.png"],
        )

    assert mock_vision.await_args is not None
    assert mock_vision.await_args.kwargs["user_prompt"] == IMAGE_PREANALYSIS_PROMPT
    expected_note = build_preanalysis_note(
        description=extraction, image_path="/tmp/cat.png", role_label="user",
    )
    assert result == expected_note + "\n\nWhat is happening here?"
    # No output cap is forwarded: per the max-tokens-knob policy the aux
    # client decides token handling; conciseness comes from the prompt.
    assert "max_tokens" not in mock_vision.await_args.kwargs
