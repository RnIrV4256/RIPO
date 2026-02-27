# DUPE‑MedSAM (scaffold)

1. Install deps: `pip install -r requirements.txt`
2. Put your CLIP and SAM checkpoints where your loader expects.
3. Run the scripts **in the following order**:
1). `new_generate.py`
2). `stable_select.py`
3). `box_generate_v2.py`
4). `score_select_v2.py`
5). `segment.py`
6). `main_autoloop.py`
Steps 1-5 can be implemented step-by-step, or step 6 can be run directly to implement iterative loop optimization.
