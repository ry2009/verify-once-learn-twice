from __future__ import annotations


def base_prompt(prompt: str) -> str:
    return (
        "Write only Python statements for the function body that continues this definition.\n"
        "Do not include markdown, prose, comments, or a full def line.\n\n"
        f"{prompt}"
    )


def feedback_prompt(prompt: str, feedback: str) -> str:
    return (
        "Fix the function body so unit tests pass.\n"
        "Return only Python statements for the body.\n"
        "Do not include markdown, prose, comments, or a full def line.\n\n"
        f"Task:\n{prompt}\n\n"
        f"Test failure:\n{feedback}\n\n"
        "Corrected function body:"
    )


def judge_prompt(prompt: str, feedback: str, candidate: str) -> str:
    return (
        "You are a strict judge. Decide whether the candidate code satisfies the feedback.\n"
        "Return exactly one token: PASS or FAIL.\n\n"
        f"Task:\n{prompt}\n\n"
        f"Feedback:\n{feedback}\n\n"
        f"Candidate completion:\n{candidate}\n"
    )
