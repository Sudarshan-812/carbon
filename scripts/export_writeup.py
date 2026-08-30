"""Export docs/writeup.md to PDF (Prompt.md - Phase 14).

Tries ``pandoc`` first. If it is not installed, renders a self-contained
``docs/writeup.html`` with a tiny Markdown subset converter (headings, tables,
fenced code, lists, ``**bold**`` / ``*em*`` / ``code``) and prints how to turn
that into a PDF from any browser (Ctrl+P → Save as PDF).
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "writeup.md"
PDF = ROOT / "docs" / "writeup.pdf"
HTML = ROOT / "docs" / "writeup.html"

_CSS = """
@page { size: A4; margin: 18mm; }
body { font: 11pt/1.45 -apple-system, "Segoe UI", Roboto, sans-serif; color: #1a1a1a; max-width: 46em; margin: 2em auto; padding: 0 1em; }
h1 { font-size: 1.7em; border-bottom: 2px solid #333; padding-bottom: .2em; }
h2 { font-size: 1.3em; margin-top: 1.6em; border-bottom: 1px solid #ccc; padding-bottom: .15em; }
h3 { font-size: 1.05em; margin-top: 1.2em; }
code { background: #f2f2f2; padding: .1em .3em; border-radius: 3px; font-size: .9em; }
pre { background: #f6f8fa; padding: .9em; border-radius: 5px; overflow-x: auto; font-size: .82em; line-height: 1.3; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: .92em; }
th, td { border: 1px solid #ccc; padding: .35em .6em; text-align: left; }
th { background: #f2f2f2; }
ul { padding-left: 1.3em; }
"""

_INLINE = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<em>\1</em>"),
)


def _inline(text: str) -> str:
    out = html.escape(text)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _md_to_html(md: str) -> str:
    body: list[str] = []
    lines = md.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if line.startswith("```"):                       # fenced code
            block = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                block.append(html.escape(lines[i]))
                i += 1
            i += 1
            body.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        if line.startswith("#"):                          # heading
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
            i += 1
            continue

        if line.lstrip().startswith("|") and "|" in line[1:]:   # table
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, *rest = rows
            if rest and all(set(c) <= set("-: ") for c in rest[0]):
                rest = rest[1:]
            out = ["<table><thead><tr>"]
            out += [f"<th>{_inline(c)}</th>" for c in header]
            out.append("</tr></thead><tbody>")
            for r in rest:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            body.append("".join(out))
            continue

        if line.lstrip().startswith(("- ", "* ")):        # unordered list
            items = []
            while i < n and lines[i].lstrip().startswith(("- ", "* ")):
                items.append(f"<li>{_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue

        if line.strip():                                  # paragraph
            para = [line]
            i += 1
            while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", "```")) \
                    and not lines[i].lstrip().startswith(("- ", "* ")):
                para.append(lines[i])
                i += 1
            body.append(f"<p>{_inline(' '.join(para))}</p>")
            continue

        i += 1

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(MD.stem)}</title><style>{_CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")


def main() -> int:
    if not MD.is_file():
        print(f"missing {MD}", file=sys.stderr)
        return 2

    pandoc = shutil.which("pandoc")
    if pandoc:
        # pandoc -> PDF needs a PDF engine; try the common ones, else fall
        # through to the self-contained HTML.
        engines = ["weasyprint", "wkhtmltopdf", "typst", "pdflatex", "xelatex"]
        avail = next((e for e in engines if shutil.which(e)), None)
        cmd = [pandoc, str(MD), "-o", str(PDF)]
        if avail:
            cmd.append(f"--pdf-engine={avail}")
        if subprocess.run(cmd, check=False).returncode == 0 and PDF.is_file():
            print(f"wrote {PDF}  (via pandoc{f' + {avail}' if avail else ''})")
            return 0
        print("pandoc found but no working PDF engine — writing HTML instead.")

    HTML.write_text(_md_to_html(MD.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {HTML}")
    print("To get the PDF, either:")
    print(f"  1. open {HTML.name} in a browser → Ctrl+P → 'Save as PDF', or")
    print("  2. install pandoc (winget install --id JohnMacFarlane.Pandoc -e) and re-run, or")
    print("  3. open docs/writeup.md in VS Code with the 'Markdown PDF' extension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
