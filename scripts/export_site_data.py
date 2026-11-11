"""Print ``src/data/code-labs.ts`` for the course site (https://github.com/xwzy/dmxsf.com).

The site draws its "companion code" blocks and the /code/ page from a mirror of ``dmxsf.course``.
Never edit the mirror by hand; change ``src/dmxsf/course.py`` and regenerate it:

    uv run python scripts/export_site_data.py > /path/to/dmxsf.com/src/data/code-labs.ts
"""

from __future__ import annotations

import json

from dmxsf import course


def q(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def main() -> None:
    covered = course.by_key()
    out = [
        "/**",
        " * The companion code (https://github.com/dmxsf/dmxsf): which lab belongs to which wiki page.",
        " *",
        " * GENERATED from src/dmxsf/course.py of that repository by scripts/export_site_data.py.",
        " * Do not edit by hand: change course.py there and regenerate this file.",
        " *",
        " * - `key` is `<section>/<slug>`, the lab's directory under labs/.",
        " * - `wiki` lists every wiki key the lab covers (the primary page first); each page has exactly one lab.",
        " * - `seconds` is about how long a full run takes on a laptop CPU.",
        " */",
        "export interface CodeLab {",
        "  key: string;",
        "  wiki: string[];",
        "  seconds: number;",
        "  /** Optional extra (`uv sync --extra <extra>`) that the lab's --hf path needs. */",
        "  extra?: string;",
        "  zh: { title: string; summary: string };",
        "  en: { title: string; summary: string };",
        "}",
        "",
        "export interface CodeRoute {",
        "  id: string;",
        "  /** For each week of the roadmap, the keys of the labs to run (empty: read and write up). */",
        "  weeks: string[][];",
        "}",
        "",
        "export const CODE_LABS: CodeLab[] = [",
    ]
    for lab in course.LABS:
        zh_title, zh_summary = lab.text("zh")
        title, summary = lab.text("en")
        out += [
            "  {",
            f"    key: {q(lab.key)},",
            f"    wiki: [{', '.join(q(k) for k in lab.wiki)}],",
            f"    seconds: {lab.seconds},",
            *([f"    extra: {q(lab.extra)},"] if lab.extra else []),
            f"    zh: {{ title: {q(zh_title)}, summary: {q(zh_summary)} }},",
            f"    en: {{ title: {q(title)}, summary: {q(summary)} }},",
            "  },",
        ]
    out += ["];", "", "export const CODE_ROUTES: CodeRoute[] = ["]
    for route, (_, weeks) in course.ROUTES.items():
        out.append(f"  {{ id: {q(route)}, weeks: [")
        for pages in weeks:
            keys = list(dict.fromkeys(covered[page].key for page in pages))
            out.append(f"    [{', '.join(q(k) for k in keys)}],")
        out.append("  ] },")
    out += ["];", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
