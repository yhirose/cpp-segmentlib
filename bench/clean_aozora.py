import sys, re, html

def clean(raw_bytes):
    text = raw_bytes.decode('shift_jis', errors='ignore')
    # Take from the main_text div start to the bibliographical info (or end).
    start = text.find('<div class="main_text">')
    if start != -1:
        text = text[start:]
    end = text.find('<div class="bibliographical_information">')
    if end != -1:
        text = text[:end]
    body = text
    body = re.sub(r'<rp>.*?</rp>', '', body, flags=re.S)
    body = re.sub(r'<rt>.*?</rt>', '', body, flags=re.S)
    body = re.sub(r'［＃.*?］', '', body, flags=re.S)
    body = re.sub(r'<br\s*/?>', '\n', body)
    body = re.sub(r'<[^>]+>', '', body)
    body = html.unescape(body)
    lines = []
    for line in body.split('\n'):
        line = line.strip().replace('　', '')
        if not line:
            continue
        for p in re.split(r'(?<=[。！？])', line):
            p = p.strip()
            if p:
                lines.append(p)
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    data = open(sys.argv[1], 'rb').read()
    sys.stdout.write(clean(data))
