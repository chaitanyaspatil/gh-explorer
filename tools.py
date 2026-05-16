"""LangChain tools the agent can call.

Each tool wraps a GitHub endpoint via github_api.gh_get and trims the
response to fields the agent actually needs.

On success a tool returns a plain data dict (the trimmed fields).
On failure it returns {"error": "<short code or message>"} so the agent
can observe and adapt without exceptions crossing the tool boundary.
"""

from __future__ import annotations

import base64
import functools
from datetime import datetime, timedelta, timezone
import requests

from langchain_core.tools import tool
from github_api import gh_get


# Length caps to keep tool outputs from blowing context.
README_CHAR_CAP = 5000
ISSUE_BODY_CHAR_CAP = 1500
COMMENT_BODY_CHAR_CAP = 500
RELEASE_BODY_CHAR_CAP = 1500
ITEM_BODY_EXCERPT_CHAR_CAP = 300
MAX_COMMENTS_PER_ISSUE = 10
MAX_TIMELINE_EVENTS = 20


def _truncate(text: str | None, cap: int) -> str:
    if not text:
        return ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"... [truncated, {len(text) - cap} chars omitted]"


# HTTP statuses where the agent can plausibly recover by changing arguments
# (bad query, missing resource, unprocessable input)
AGENT_FIXABLE_STATUSES = (400, 404, 422)


def gh_safe(fn):
    """Decorator: catch agent-fixable HTTP errors and return them as data.

    Wrap any tool whose body calls gh_get(). On HTTPError with a status in
    AGENT_FIXABLE_STATUSES, the wrapper returns {"error": str(e)} so the
    agent observes the failure as a tool result and can adapt. Other
    HTTPErrors and any other exceptions propagate.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            if e.response.status_code in AGENT_FIXABLE_STATUSES:
                return {"error": str(e)}
            raise
    return wrapper


# ---------- Tool 1: read_readme ----------

@tool
@gh_safe
def read_readme(repo: str) -> dict:
    """Fetch the README of a GitHub repo.

    Args:
        repo: GitHub repo in "owner/name" form.

    Returns:
        On success: {content, html_url}. content is truncated to ~5000 chars.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    payload = gh_get(f"/repos/{repo}/readme")
    encoded = payload.get("content", "")
    decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
    return {
        "content": _truncate(decoded, README_CHAR_CAP),
        "html_url": payload.get("html_url"),
    }


# ---------- Tool 2: recent_commits ----------

@tool
@gh_safe
def recent_commits(repo: str, days: int = 14) -> dict:
    """List commits to the default branch within the last N days.

    Returns up to 30 commits.

    Args:
        repo: GitHub repo in "owner/name" form.
        days: Look-back window in days. Default 14.

    Returns:
        On success: {count, commits: [{sha, date, author, message_first_line, html_url}]}.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    # fetch commits
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    data = gh_get(f"/repos/{repo}/commits", params={"since": since, "per_page": 30})
    
    # loop through commits and extract extract important data for each
    commits = []
    for c in data:
        commit = c.get("commit", {})
        msg = commit.get("message", "")
        commits.append({
            "sha": c.get("sha", "")[:10],
            "date": commit.get("author", {}).get("date"),
            "author": commit.get("author", {}).get("name"),
            "message_first_line": msg.split("\n", 1)[0][:200],
            "html_url": c.get("html_url"),
        })
    return {"count": len(commits), "commits": commits}


# ---------- Tool 3: list_releases ----------

@tool
@gh_safe
def list_releases(repo: str, n: int = 5) -> dict:
    """List the most recent N releases.

    Args:
        repo: GitHub repo in "owner/name" form.
        n: Max releases to return. Default 5.

    Returns:
        On success: {count, releases: [{tag_name, name, published_at, body_excerpt, html_url}]}.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    data = gh_get(f"/repos/{repo}/releases", params={"per_page": n})
    releases = [{
        "tag_name": r.get("tag_name"),
        "name": r.get("name"),
        "published_at": r.get("published_at"),
        "body_excerpt": _truncate(r.get("body"), RELEASE_BODY_CHAR_CAP),
        "html_url": r.get("html_url"),
    } for r in data]
    return {"count": len(releases), "releases": releases}


# ---------- Tool 4: search_issues ----------

@tool
@gh_safe
def search_issues(
    repo: str,
    query: str,
    state: str = "open",
    sort: str = "updated",
    limit: int = 15,
) -> dict:
    """Search issues in a repo. Returns total_count and a trimmed item list.

    The query is the q-syntax for GitHub issue search, MINUS the repo qualifier
    (which is added automatically). Q-syntax examples:
      - 'streaming label:"partner: anthropic"'
      - 'label:"good first issue"'
      - 'sort:comments-desc'

    Use total_count for cheap counts (no need for a separate count tool).
    Set limit=1 if you only need the count.

    Args:
        repo: GitHub repo in "owner/name" form.
        query: Issue search query (without repo: qualifier).
        state: 'open' (default), 'closed', or 'all'. Folded into query as is:state.
        sort: 'updated' (default), 'created', 'comments', or 'reactions'.
        limit: Max items to return. Default 15.

    Returns:
        On success: {total_count, items: [...]}.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    # create query
    q_parts = [f"repo:{repo}", "is:issue"]
    if state in ("open", "closed"):
        q_parts.append(f"is:{state}")
    q_parts.append(query)
    full_query = " ".join(q_parts)

    # get all issues corresponding to query
    payload = gh_get("/search/issues", params={
        "q": full_query, "sort": sort, "order": "desc", "per_page": limit,
    })

    # format each issue
    items = []
    for it in payload.get("items", []):
        items.append({
            "number": it.get("number"),
            "title": it.get("title"),
            "state": it.get("state"),
            "labels": [l.get("name") for l in it.get("labels", [])],
            "comments": it.get("comments"),
            "created_at": it.get("created_at"),
            "updated_at": it.get("updated_at"),
            "html_url": it.get("html_url"),
            "body_excerpt": _truncate(it.get("body"), ITEM_BODY_EXCERPT_CHAR_CAP),
        })
    return {"total_count": payload.get("total_count", 0), "items": items}


# ---------- Tool 5: get_issue ----------

@tool
@gh_safe
def get_issue(repo: str, number: int) -> dict:
    """Fetch a single issue with its body, comments, and timeline events.

    Args:
        repo: GitHub repo in "owner/name" form.
        number: Issue number.

    Returns:
        On success: {number, title, state, labels, body, assignees,
                     comments_count, comments: [...], timeline_events: [...], html_url}.
            Comments capped at 10. Timeline filtered to assigned/unassigned/
            cross-referenced/referenced/closed events; capped at 20.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    issue = gh_get(f"/repos/{repo}/issues/{number}")

    # get issue comments
    comments = []
    for c in gh_get(
        f"/repos/{repo}/issues/{number}/comments",
        params={"per_page": MAX_COMMENTS_PER_ISSUE},
    ):
        comments.append({
            "user": c.get("user", {}).get("login"),
            "created_at": c.get("created_at"),
            "body_excerpt": _truncate(c.get("body"), COMMENT_BODY_CHAR_CAP),
        })

    # get issue timeline
    relevant_events = {"assigned", "unassigned", "cross-referenced", "referenced", "closed"}
    timeline = []
    for event in gh_get(
        f"/repos/{repo}/issues/{number}/timeline",
        params={"per_page": MAX_TIMELINE_EVENTS},
    ):
        event_type = event.get("event")
        if event_type not in relevant_events:
            continue

        entry = {"event": event_type, "created_at": event.get("created_at")}

        if event_type in {"assigned", "unassigned"}:
            entry["assignee"] = (event.get("assignee") or {}).get("login")
        
        if event_type == "cross-referenced":
            source = event.get("source", {}).get("issue", {})
            entry["from_issue_or_pr"] = source.get("number")
            entry["from_html_url"] = source.get("html_url")
            entry["is_pull_request"] = "pull_request" in source
        
        timeline.append(entry)

    # create issue summary and return
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "state": issue.get("state"),
        "labels": [l.get("name") for l in issue.get("labels", [])],
        "assignees": [a.get("login") for a in issue.get("assignees", [])],
        "comments_count": issue.get("comments"),
        "body": _truncate(issue.get("body"), ISSUE_BODY_CHAR_CAP),
        "comments": comments,
        "timeline_events": timeline,
        "html_url": issue.get("html_url"),
    }


# ---------- Tool 6: list_labels ----------

@tool
@gh_safe
def list_labels(repo: str) -> dict:
    """List all labels defined on the repo.

    Each label includes its name, description, and color. Does NOT include
    per-label issue counts — call search_issues with a label filter
    (limit=1) to get a count.

    Args:
        repo: GitHub repo in "owner/name" form.

    Returns:
        On success: {count, labels: [{name, description, color}]}.
        On failure: {error} if the agent can fix it (bad repo name etc.);
        otherwise the exception propagates.
    """
    data = gh_get(f"/repos/{repo}/labels", params={"per_page": 100})
    labels = [{
        "name": l.get("name"),
        "description": l.get("description"),
        "color": l.get("color"),
    } for l in data]
    return {"count": len(labels), "labels": labels}


# Convenience: the list of all tools, for agent.py to import.
ALL_TOOLS = [
    read_readme,
    recent_commits,
    list_releases,
    search_issues,
    get_issue,
    list_labels,
]
