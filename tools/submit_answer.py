"""Tool for submitting a final SQL query answer to the evaluation pipeline."""

from framework.agent import ANSWER_SUBMITTED_PREFIX, Tool
from framework.database import execute_query


def submit_answer(query: str) -> str:
    """Submit a SQL query as the final answer to the evaluation.

    The query is executed first; if it fails or returns no rows, the submission is NOT
    accepted and the error is returned to the model so it can fix the query (an
    empty result is accepted on a second, identical submission).

    Returns:
        ANSWER_SUBMITTED_PREFIX + query on success (signals the agent loop to stop).
    """
    result = execute_query(query)
    if not result.is_success:
        return (
            f"Submission rejected - the query failed to execute: {result.error_message}\n"
            "Fix the query and call submit_answer again."
        )
    if result.is_empty and query.strip() not in _empty_warned:
        _empty_warned.add(query.strip())
        return (
            "Submission not yet accepted - the query returned 0 rows. Double-check filters "
            "and value spellings with run_sql/profile_column. If you are sure an empty "
            "result is correct, submit the identical query again."
        )
    # IMPORTANT: Do not add any text after the query - the extraction logic in the evals
    # takes everything after the prefix as the query.
    return f"{ANSWER_SUBMITTED_PREFIX}{query}"


_empty_warned: set[str] = set()


SUBMIT_ANSWER: Tool = Tool(
    name="submit_answer",
    description=(
        "Submit your final SQL query as the answer to the current question. The query is "
        "executed; if it errors you will get the error back and must fix it. Call this "
        "only after you have run the query with run_sql and checked the result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The final DuckDB SQL query. Table names must be schema-qualified "
                    "(e.g., 'SELECT * FROM schema.table')."
                ),
            },
        },
        "required": ["query"],
    },
    function=submit_answer,
)
