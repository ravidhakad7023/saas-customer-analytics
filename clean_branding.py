import os
import glob
import re

directories = ['notebooks', 'src', 'dashboard', 'tests', 'queries', '']
extensions = ['*.py', '*.ipynb', '*.md', '*.sql']

files = []
for d in directories:
    for ext in extensions:
        pattern = os.path.join(d, ext) if d else ext
        files.extend(glob.glob(pattern))

replacements = {
    r"PulseMetrics\s*/\s*RevenueIQ": "PipelineIQ",
    r"PulseMetrics": "PipelineIQ",
    r"RevenueIQ": "PipelineIQ",
    r"pulsemetrics\.io": "example.com",
    r"adan-data": "your-username",
    r"\*\*Author:\*\* \[Your Name\] \| ": "",
    r"Author: \[Your Name\]": "",
    r"\[Your Name\]": ""
}

for filepath in set(files):
    # skip this script itself and sqlite db
    if 'clean_branding.py' in filepath or filepath.endswith('.db'): continue
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for pattern, replacement in replacements.items():
            new_content = re.sub(pattern, replacement, new_content, flags=re.IGNORECASE)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Could not process {filepath}: {e}")
