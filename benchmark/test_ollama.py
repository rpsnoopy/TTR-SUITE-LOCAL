import requests, json
r = requests.get("http://localhost:11434/api/tags")
models = json.loads(r.text).get("models", [])
if models:
    for m in models:
        print(m["name"])
else:
    print("(vuoto)")
