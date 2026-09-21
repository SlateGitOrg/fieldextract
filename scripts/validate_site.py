from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
required=['site/index.html','site/styles.css','site/app.mjs','site/core.mjs','site/core.test.mjs']
missing=[name for name in required if not (ROOT/name).is_file()]
if missing: raise SystemExit('Missing: '+', '.join(missing))
html=(ROOT/'site/index.html').read_text(encoding='utf-8')
for marker in ['<form id="workspace-form"','type="submit"','aria-live="polite"','app.mjs']:
    if marker not in html: raise SystemExit('Missing HTML marker: '+marker)
print('Site validation passed')
