# Builds gesture-detector.html from template.html, results.json and sample images.
# Run train_numpy.py first to (re)create results.json.
import base64, io, json, os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')


def uri(f):
    b = io.BytesIO()
    Image.open(os.path.join(R, f)).save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


res = json.load(open(os.path.join(HERE, 'results.json')))
grid = {}
for s in ['A', 'D', 'H', 'K']:
    for g in ['down', 'up', 'hold', 'stop']:
        fs = sorted(x for x in os.listdir(os.path.join(R, 'gestures', s)) if f'_{g}_' in x)
        if fs:
            grid[f'{s}_{g}'] = uri(f'gestures/{s}/{fs[0]}')
for w in res['wrong']:
    w['img'] = uri(w['file'])
res['grid'] = grid

page = open(os.path.join(HERE, 'template.html')).read().replace('__DATA__', json.dumps(res))
open(os.path.join(HERE, 'gesture-detector.html'), 'w').write(page)
print('wrote gesture-detector.html')
