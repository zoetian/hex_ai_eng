"""Database exploration tools: discover schemas/tables/columns and run exploratory SQL.

Everything here is discovered from the live database catalog, so the same tools work
against any DuckDB warehouse (no hand-maintained schema map).
"""

import threading

import duckdb
import polars as pl

from framework.agent import Tool
from framework.database import DATABASE_PATH

QUERY_TIMEOUT_S = 60.0
MAX_PREVIEW_ROWS = 30
MAX_CELL_CHARS = 80


def _run(query: str, params: list | None = None) -> pl.DataFrame:
    """Run a query read-only with a timeout; raises on error."""
    conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
    timer = threading.Timer(QUERY_TIMEOUT_S, conn.interrupt)
    timer.start()
    try:
        result = conn.execute(query, params or [])
        return pl.DataFrame(result.fetch_arrow_table())
    finally:
        timer.cancel()
        conn.close()


def format_df(df: pl.DataFrame, max_rows: int = MAX_PREVIEW_ROWS) -> str:
    """Render a dataframe compactly as pipe-separated text for the LLM."""
    header = f"{df.height} rows x {df.width} cols"
    if df.height > max_rows:
        header += f" (showing first {max_rows})"
    cols = [f"{c} ({t})" for c, t in zip(df.columns, df.dtypes, strict=True)]
    lines = [header, " | ".join(cols)]
    for row in df.head(max_rows).iter_rows():
        cells = []
        for v in row:
            s = "NULL" if v is None else repr(v) if isinstance(v, str) else str(v)
            cells.append(s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "...")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def schema_names() -> list[str]:
    df = _run("SELECT DISTINCT schema_name FROM duckdb_tables() WHERE NOT internal")
    return sorted(df["schema_name"].to_list())


def list_schemas_overview() -> str:
    """One line per schema: name and its table names (used in the system prompt)."""
    df = _run(
        """
        SELECT schema_name, string_agg(table_name, ', ' ORDER BY table_name) AS tables
        FROM duckdb_tables() WHERE NOT internal
        GROUP BY schema_name ORDER BY schema_name
        """
    )
    lines = []
    for schema, tables in df.iter_rows():
        if len(tables) > 300:
            tables = tables[:300] + ", ..."
        lines.append(f"- {schema}: {tables}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: describe_schema
# ---------------------------------------------------------------------------


def describe_schema(schema: str, tables: list[str] | None = None) -> str:
    """Describe all (or selected) tables of a schema: row counts, columns, types."""
    tdf = _run(
        """
        SELECT table_name, estimated_size FROM duckdb_tables()
        WHERE lower(schema_name) = lower(?) ORDER BY table_name
        """,
        [schema],
    )
    if tdf.is_empty():
        return f"Schema '{schema}' not found. Check the schema list in the system prompt."
    cdf = _run(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM duckdb_columns() WHERE lower(schema_name) = lower(?)
        ORDER BY table_name, column_index
        """,
        [schema],
    )
    wanted = {t.lower() for t in tables} if tables else None
    out = []
    for table_name, size in tdf.iter_rows():
        if wanted and table_name.lower() not in wanted:
            continue
        cols = cdf.filter(pl.col("table_name") == table_name)
        col_str = ", ".join(
            f"{c} {t}" + ("" if n else " NOT NULL") for _, c, t, n in cols.iter_rows()
        )
        out.append(f"{schema}.{table_name} (~{size} rows): {col_str}")
    return "\n".join(out) if out else f"No matching tables in '{schema}'."


DESCRIBE_SCHEMA = Tool(
    name="describe_schema",
    description=(
        "List the tables in a schema with approximate row counts and every column with its "
        "type. Optionally restrict to some tables. Use this before writing SQL."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema": {"type": "string", "description": "Schema name, e.g. 'financial'."},
            "tables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of table names.",
            },
        },
        "required": ["schema"],
    },
    function=describe_schema,
)


# ---------------------------------------------------------------------------
# Tool: profile_column
# ---------------------------------------------------------------------------


def profile_column(table: str, column: str, limit: int = 25) -> str:
    """Null count, distinct count, min/max and most frequent values of a column."""
    col = f'"{column}"'
    stats = _run(
        f"""
        SELECT count(*) AS total_rows, count({col}) AS non_null,
               count(DISTINCT {col}) AS distinct_values,
               min({col})::VARCHAR AS min, max({col})::VARCHAR AS max
        FROM {table}
        """
    )
    top = _run(
        f"""
        SELECT {col} AS value, count(*) AS n FROM {table}
        GROUP BY 1 ORDER BY n DESC, 1 LIMIT {int(limit)}
        """
    )
    return f"{format_df(stats)}\n\nMost frequent values:\n{format_df(top, max_rows=limit)}"


PROFILE_COLUMN = Tool(
    name="profile_column",
    description=(
        "Profile one column: total rows, NULL count, distinct count, min/max and the most "
        "frequent values with counts. Use it to learn exact codes/spellings/casing, "
        "sentinel or test values, and NULL behavior before filtering on a column."
    ),
    parameters={
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "Schema-qualified table, e.g. 'financial.loan'.",
            },
            "column": {"type": "string", "description": "Column name."},
            "limit": {"type": "integer", "description": "How many top values (default 25)."},
        },
        "required": ["table", "column"],
    },
    function=profile_column,
)


# ---------------------------------------------------------------------------
# Tool: run_sql
# ---------------------------------------------------------------------------


def run_sql(query: str) -> str:
    """Execute an exploratory read-only query and return a preview of the result."""
    try:
        return format_df(_run(query))
    except Exception as e:
        return f"SQL error: {e}"


RUN_SQL = Tool(
    name="run_sql",
    description=(
        "Execute a read-only DuckDB SQL query and see the first rows of the result plus the "
        "total row count. Use it to explore data and to test your final query before "
        "submitting. Always schema-qualify tables (schema.table)."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "DuckDB SQL query."}},
        "required": ["query"],
    },
    function=run_sql,
)
