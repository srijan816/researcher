# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI routes for unified AI-Q API."""

__all__ = [
    "add_collection_routes",
    "add_document_routes",
    "register_job_routes",
    "register_prompt_routes",
    "register_queue_routes",
]


def __getattr__(name: str):
    """Lazily import route registration helpers.

    Some route modules pull in the full AI-Q runtime dependency stack. Keeping
    these imports lazy lets lightweight routes and tests load without requiring
    optional database or agent packages until they are actually used.
    """
    if name == "add_collection_routes":
        from .collections import add_collection_routes

        return add_collection_routes
    if name == "add_document_routes":
        from .documents import add_document_routes

        return add_document_routes
    if name == "register_job_routes":
        from .jobs import register_job_routes

        return register_job_routes
    if name == "register_prompt_routes":
        from .prompts import register_prompt_routes

        return register_prompt_routes
    if name == "register_queue_routes":
        from .queue import register_queue_routes

        return register_queue_routes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
