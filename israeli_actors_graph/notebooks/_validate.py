import json

with open('06_link_prediction_improved.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('✅ Notebook is valid JSON!')
print(f'Total cells: {len(nb["cells"])}')
print(f'Code cells: {sum(1 for c in nb["cells"] if c["cell_type"] == "code")}')
print(f'Markdown cells: {sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")}')
print(f'Format version: {nb["nbformat"]}.{nb["nbformat_minor"]}')

# Check first code cell
first_code = next(c for c in nb["cells"] if c["cell_type"] == "code")
print(f'\n📝 First code cell (preview):')
print(''.join(first_code["source"][:5]))
