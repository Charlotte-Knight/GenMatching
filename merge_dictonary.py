import json
import sys

ins = sys.argv[1:-1]
print(ins)
out = sys.argv[-1]

in_dicts = {}
for f in ins:
  with open(f, "r") as f:
    in_dicts.update(json.load(f))

print(f"write to {out}")
with open(out, "w") as f:
  json.dump(in_dicts, f)
