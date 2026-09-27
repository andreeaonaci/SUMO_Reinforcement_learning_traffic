"""For each synthesized / native program: which controlled links are NEVER green in
any selectable phase (i.e. traffic on them can never be served).  Any Python."""
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
GREEN = set('Gg')


def programs(path):
    out = {}
    for logic in ET.parse(path).getroot().iter('tlLogic'):
        states = [p.get('state') for p in logic.findall('phase')]
        greens = [s for s in states if any(c in GREEN for c in s) and not any(c in 'yY' for c in s)]
        out[logic.get('id')] = greens
    return out


for key in ['arterial4x4', 'cologne3', 'ingolstadt7', 'grid4x4']:
    sc = HERE / 'scenarios' / key
    net = ET.parse(sc / 'synth.sumocfg').getroot().find('./input/net-file').get('value')
    for arm, progs in [('native', programs(net)), ('synth', programs(sc / 'synth.tll.xml'))]:
        bad = {}
        for tls, greens in progs.items():
            n = len(greens[0])
            never = [i for i in range(n) if not any(g[i] in GREEN for g in greens)]
            if never:
                bad[tls] = never
        print(f'{key:<12} {arm:<6} TLS with never-green links: {len(bad)}/{len(progs)}  {bad}')
