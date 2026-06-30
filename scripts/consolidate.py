#!/usr/bin/env python3
"""Consolidate Badger AI into ONE super-advanced library."""
import os, re, shutil

BASE = "/home/dillun/Desktop/Badger_Ai"

# Step 1: Merge blocks_v2.py → blocks.py
print("Step 1: Merging blocks_v2.py → blocks.py...")
with open(f"{BASE}/src/models/blocks_v2.py") as f:
    v2_content = f.read()
# Strip header docstring
v2_content = re.sub(r'^""".*?"""\s*', '', v2_content, flags=re.DOTALL)
# Remove duplicate imports
for pat in [r'from \.blocks import .*?\n', r'^import torch\n', r'^import torch\.nn as nn\n',
            r'^import torch\.nn\.functional as F\n', r'^import math\n']:
    v2_content = re.sub(pat, '', v2_content, flags=re.MULTILINE)
# Add section header
v2_content = ('\n\n# =============================================================================\n'
              '# Advanced Blocks — PConv, CIB, RepC2f, A², DCNv4, GELAN, DyHead\n'
              '# =============================================================================\n\n' + v2_content.strip() + '\n')
with open(f"{BASE}/src/models/blocks.py", "a") as f:
    f.write(v2_content)
print("  ✓ blocks.py merged")

# Step 2: Merge badger_v2.py → badger.py
print("Step 2: Merging badger_v2.py → badger.py...")
with open(f"{BASE}/src/models/badger_v2.py") as f:
    v2b = f.read()
v2b = re.sub(r'^""".*?"""\s*', '', v2b, flags=re.DOTALL)
# Fix imports: .blocks_v2 → .blocks, .head_v2 → .head
v2b = v2b.replace('from .blocks_v2 import', 'from .blocks import')
v2b = v2b.replace('from .head_v2 import', 'from .head import')
for pat in [r'from \.blocks import .*?\n', r'^import torch\n', r'^import torch\.nn as nn\n',
            r'^import torch\.nn\.functional as F\n']:
    v2b = re.sub(pat, '', v2b, flags=re.MULTILINE)
v2b = ('\n\n# =============================================================================\n'
       '# Badger SOTA — Next-gen (PConv/RepC2f + BiFPN + DualHead)\n'
       '# =============================================================================\n\n' + v2b.strip() + '\n')
with open(f"{BASE}/src/models/badger.py", "a") as f:
    f.write(v2b)
print("  ✓ badger.py merged")

# Step 3: Merge head_v2.py → head.py
print("Step 3: Merging head_v2.py → head.py...")
with open(f"{BASE}/src/models/head_v2.py") as f:
    v2h = f.read()
v2h = re.sub(r'^""".*?"""\s*', '', v2h, flags=re.DOTALL)
for pat in [r'from \.blocks import .*?\n', r'^import torch\n', r'^import torch\.nn as nn\n',
            r'^import torch\.nn\.functional as F\n']:
    v2h = re.sub(pat, '', v2h, flags=re.MULTILINE)
v2h = ('\n\n# =============================================================================\n'
       '# NMS-Free Dual Head (YOLOv10/YOLO26)\n'
       '# =============================================================================\n\n' + v2h.strip() + '\n')
with open(f"{BASE}/src/models/head.py", "a") as f:
    f.write(v2h)
print("  ✓ head.py merged")

# Step 4: Extract unique sota_losses content → advanced_losses.py
print("Step 4: Merging sota_losses.py → advanced_losses.py...")
with open(f"{BASE}/src/losses/sota_losses.py") as f:
    sota = f.read()
focal_match = re.search(r'(# =+\n# 3\. Focal-EIoU.*?)(?=\n# =+\n# 4\.|\Z)', sota, re.DOTALL)
compute_match = re.search(r'(# =+\n# 4\. Unified Box Loss.*)', sota, re.DOTALL)
appended = '\n'
if focal_match:
    appended += focal_match.group(1) + '\n\n'
if compute_match:
    text = compute_match.group(1)
    text = text.replace('from .advanced_losses import siou_loss', '# siou_loss: defined above')
    appended += text + '\n'
with open(f"{BASE}/src/losses/advanced_losses.py", "a") as f:
    f.write(appended)
print("  ✓ advanced_losses.py merged")

# Step 5: Delete _v2 duplicates
print("Step 5: Deleting duplicate _v2 files...")
for fn in ['blocks_v2.py', 'badger_v2.py', 'head_v2.py']:
    fp = f"{BASE}/src/models/{fn}"
    if os.path.exists(fp):
        os.remove(fp)
        print(f"  ✓ Deleted src/models/{fn}")
fp = f"{BASE}/src/losses/sota_losses.py"
if os.path.exists(fp):
    os.remove(fp)
    print(f"  ✓ Deleted src/losses/sota_losses.py")

print("\n✓✓✓ CONSOLIDATION COMPLETE — single super-advanced library ✓✓✓")
