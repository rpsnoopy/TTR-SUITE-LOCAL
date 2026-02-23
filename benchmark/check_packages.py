try:
    import pandas, requests, datasets, anthropic, tqdm, openpyxl, huggingface_hub
    print("All packages OK")
except ImportError as e:
    print(f"Missing: {e}")
