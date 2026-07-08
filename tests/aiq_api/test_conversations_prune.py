# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiq_api.routes.conversations import _clean_conversation_payload


def test_clean_conversation_payload_strips_heavy_message_fields():
    payload = {
        "id": "session_1",
        "userId": "default-user",
        "title": "Research",
        "messages": [
            {
                "id": "msg_1",
                "role": "assistant",
                "content": "answer",
                "reportContent": "very large report",
                "citations": [{"id": "c1", "url": "https://example.com"}],
                "deepResearchTodos": [{"id": "t1", "content": "todo"}],
                "thinkingSteps": [
                    {
                        "id": "ts_deep",
                        "isDeepResearch": True,
                        "displayName": "Search",
                        "content": "huge payload",
                    },
                    {
                        "id": "ts_shallow",
                        "displayName": "Plan",
                        "content": "also huge",
                        "rawPayload": '{"big": true}',
                    },
                ],
            }
        ],
    }

    cleaned = _clean_conversation_payload(payload)

    assert cleaned is not None
    message = cleaned["messages"][0]
    assert "reportContent" not in message
    assert "citations" not in message
    assert "deepResearchTodos" not in message
    assert message["content"] == "answer"
    assert len(message["thinkingSteps"]) == 1
    assert message["thinkingSteps"][0]["id"] == "ts_shallow"
    assert message["thinkingSteps"][0]["content"] == ""