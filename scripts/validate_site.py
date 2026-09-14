from html.parser import HTMLParser
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REQUIRED = [SITE / "index.html", SITE / "styles.css", SITE / "app.js"]

class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.refs = []
        self.labels = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if tag == "label" or "aria-label" in attrs or "aria-labelledby" in attrs:
            self.labels += 1
        for key in ("href", "src"):
            value = attrs.get(key, "")
            if value.startswith("./"):
                self.refs.append(value[2:].split("#")[0].split("?")[0])

def main():
    errors = []
    for path in REQUIRED:
        if not path.is_file() or path.stat().st_size < 100:
            errors.append(f"missing or empty: {path.relative_to(ROOT)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    html = (SITE / "index.html").read_text(encoding="utf-8")
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    js = (SITE / "app.js").read_text(encoding="utf-8")
    doc = Document(); doc.feed(html)
    for tag in ("main", "nav", "title"):
        if tag not in doc.tags: errors.append(f"missing <{tag}>")
    if doc.labels < 3: errors.append("interactive content lacks accessible labels")
    for ref in doc.refs:
        if ref and not (SITE / ref).is_file(): errors.append(f"broken local reference: {ref}")
    checks = [("viewport", html), (":focus-visible", css), ("prefers-reduced-motion", css), ("@media (max-width: 720px)", css), ("aria-live=\"polite\"", html)]
    for needle, source in checks:
        if needle not in source: errors.append(f"missing required contract: {needle}")
    if re.search(r"TBD|TODO|lorem ipsum", html + css + js, re.I): errors.append("placeholder copy found")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("site contract: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
