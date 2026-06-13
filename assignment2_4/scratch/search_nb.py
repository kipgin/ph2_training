import json
import glob

notebooks = glob.glob("*.ipynb")
for nb_path in notebooks:
    print(f"=== {nb_path} ===")
    with open(nb_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    if "comfy" in source.lower() or "ngrok" in source.lower() or "safetensors" in source.lower() or "export" in source.lower():
                        # print first 3 lines of the source
                        lines = source.split("\n")
                        print("Match:")
                        for line in lines[:5]:
                            print(f"  {line}")
                        print("...")
        except Exception as e:
            print(f"Error reading {nb_path}: {e}")
