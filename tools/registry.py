"""Single place that defines the agent's tool set (used by interactive.py and evaluate.py)."""

from framework.agent import Tool
from tools.db_tools import DESCRIBE_SCHEMA, PROFILE_COLUMN, RUN_SQL
from tools.guide_tools import READ_GUIDE, SEARCH_GUIDES
from tools.submit_answer import SUBMIT_ANSWER


def all_tools() -> dict[str, Tool]:
    tools = [SEARCH_GUIDES, READ_GUIDE, DESCRIBE_SCHEMA, PROFILE_COLUMN, RUN_SQL, SUBMIT_ANSWER]
    return {t.name: t for t in tools}
