# main.py  —— 适配你的 5 个脚本，Windows 双击可运行
import argparse, json, subprocess, shutil, re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2

# ---------------- 基础工具 ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path) -> List[Path]:
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def read_json(p: Path) -> Optional[Dict]:
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return None

def write_json(p: Path, obj: Dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def bin_mask(p: Path):
    if not p.exists(): return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    return (m > 127).astype(np.uint8)

def dice_coeff(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    return float((2.0*inter) / (pred.sum() + gt.sum() + eps))

def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[.;,]\s*$", "", t)
    return t

# ---------------- 结构/内容反馈：3×3→9 ----------------
def structural_variants(base_prompt: str, max_tokens: int=22) -> List[str]:
    def cleanup(s: str) -> str:
        s = normalize_text(s)
        s = re.sub(r"\b(mild|slight|probable|possible|likely|suspicious|suggestive of)\b", "", s, flags=re.I)
        s = re.sub(r"\b(the|a|an)\b", "", s, flags=re.I)
        s = re.sub(r"\s{2,}", " ", s).strip(",; ").strip()
        if not re.search(r"\blung[s]?\b", s, flags=re.I):
            s = "Lungs: " + s
        toks = s.split()
        if len(toks) > max_tokens:
            s = " ".join(toks[:max_tokens])
        if not s.endswith("."): s += "."
        return s
    v1 = cleanup(base_prompt)
    v2 = cleanup(f"{base_prompt}. Describe distribution, extent and symmetry.")
    v3 = cleanup(f"{base_prompt}. Without pleural effusion.")
    out=[]
    for v in [v1,v2,v3]:
        if v not in out: out.append(v)
    return out[:3]

def mask_stats(mask) -> Dict:
    H, W = mask.shape
    area = int(mask.sum())
    frac = area/(H*W+1e-6)
    left = int(mask[:,:W//2].sum()); right = int(mask[:,W//2:].sum())
    side = "bilateral" if min(left,right)/(max(left,right)+1e-6)>=0.6 else ("left" if left>right else "right")
    border = np.zeros_like(mask, np.uint8)
    t = int(0.1*min(H,W)); border[:t,:]=1; border[-t:,:]=1; border[:,:t]=1; border[:,-t:]=1
    peripheral_frac = (mask & border).sum()/(area+1e-6)
    thirds = np.array_split(mask, 3, axis=0)
    thirds_frac = [blk.sum()/(area+1e-6) for blk in thirds]
    vertical = ["upper","middle","lower"][int(np.argmax(thirds_frac))]
    return {"H":H,"W":W,"area":area,"frac":frac,"side":side,"peripheral_frac":peripheral_frac,"vertical":vertical}

def content_variants(base_prompt: str, pred_stats: Dict, gt_stats: Optional[Dict], dice: float) -> List[str]:
    want = (gt_stats["side"] if gt_stats else pred_stats["side"])
    side_phrase = "Bilateral involvement" if want=="bilateral" else f"{want.capitalize()} lung involvement"
    periph = (gt_stats["peripheral_frac"] if gt_stats else pred_stats["peripheral_frac"])
    dist = "predominantly peripheral" if periph>=0.35 else "multifocal patchy"
    frac_ref = (gt_stats["frac"] if gt_stats else pred_stats["frac"])
    extent = "diffuse" if frac_ref>=0.08 else ("extensive" if frac_ref>=0.04 else "focal")
    vertical = (gt_stats["vertical"] if gt_stats else pred_stats["vertical"])

    c1 = f"Lungs: {side_phrase} with {dist} {extent} ground-glass opacities, {vertical} predominance. No pleural effusion."
    c2 = f"Lungs: {side_phrase}, {dist} distribution with patchy ground-glass and consolidations, {vertical} zone dominant. Without pleural effusion."
    c3 = f"Lungs: {side_phrase}. {extent.capitalize()} opacities with subpleural predominance, {vertical} predominant. No pleural effusion."
    if dice>=0 and dice<0.6 and want=="bilateral":
        c1 = re.sub(r"\b(left|right) lung involvement\b", "Bilateral involvement", c1, flags=re.I)
        c2 = re.sub(r"\b(left|right) lung involvement\b", "Bilateral involvement", c2, flags=re.I)
    outs=[]
    for s in [c1,c2,c3]:
        s = structural_variants(s, max_tokens=24)[0]
        if s not in outs: outs.append(s)
    return outs[:3]

def cross_3x3_to_9(struct3: List[str], content3: List[str]) -> List[str]:
    outs = []
    for s in struct3:
        for c in content3:
            body = normalize_text(c)
            merged = body if s.lower().startswith("lungs") else normalize_text(f"{s} {body}")
            if not merged.endswith("."): merged += "."
            if merged not in outs: outs.append(merged)
    return outs[:9]

# ---------------- 子脚本包装（按各脚本 argparse 适配） ----------------
def run_step1(step1_py: Path, images_root: Path, out_dir: Path, jsonl_path: Path):
    """
    scripts/batch_generate_prompts.py
      --input_dir --output_dir --jsonl [--api_key --model --overwrite --limit]
    """
    subprocess.run([
        "python", str(step1_py),
        "--input_dir", str(images_root),
        "--output_dir", str(out_dir),
        "--jsonl", str(jsonl_path),
        "--overwrite"
    ], check=True)  # 参数名参考源码 :contentReference[oaicite:5]{index=5}

def run_step2(step2_py: Path, in_jsonl: Path, out_jsonl: Path, out_txt_root: Path):
    """
    prompts/stable_select.py
      --input_jsonl --output_jsonl --output_txt_root [--api_key --model]
    """
    subprocess.run([
        "python", str(step2_py),
        "--input_jsonl", str(in_jsonl),
        "--output_jsonl", str(out_jsonl),
        "--output_txt_root", str(out_txt_root)
    ], check=True)  # 参数名参考源码 :contentReference[oaicite:6]{index=6}

def run_step3(step3_py: Path, images_root: Path, prompts_root: Path, out_dir: Path, device="cuda"):
    subprocess.run([
        "python", str(step3_py),
        "--images_root", str(images_root),
        "--prompts_txt_root", str(prompts_root),
        "--out_dir", str(out_dir),
        "--device", device
    ], check=True)  # 源码参数名 :contentReference[oaicite:7]{index=7}

def run_step4(step4_py: Path, images_root: Path, prompts_root: Path, boxes_root: Path, out_root: Path, device="cuda"):
    subprocess.run([
        "python", str(step4_py),
        "--images_root", str(images_root),
        "--prompts_txt_root", str(prompts_root),
        "--boxes_root", str(boxes_root),
        "--out_root", str(out_root),
        "--device", device
    ], check=True)  # 源码参数名 :contentReference[oaicite:8]{index=8}

def run_step5(step5_py: Path, images_root: Path, select_root: Path, out_root: Path,
              model_type: str, checkpoint: Path, device="cuda",
              do_split_lr=True, keep_per_side=1):
    cmd = [
        "python", str(step5_py),
        "--images_root", str(images_root),
        "--select_root", str(select_root),
        "--out_root", str(out_root),
        "--model_type", str(model_type),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--min_area", "2000",
        "--keep_per_side", str(keep_per_side),
        "--smooth_kernel", "7"
    ]
    if do_split_lr: cmd.append("--do_split_lr")
    subprocess.run(cmd, check=True)  # 源码参数名 :contentReference[oaicite:9]{index=9}

# ---------------- 主循环 ----------------
def main():
    ap = argparse.ArgumentParser("AutoLoop: Step1-5 迭代 + Dice早停（final seg）")
    # 数据
    ap.add_argument("--images_root",  default=r"E:\data\QaTa-COV19-v2\100train")
    ap.add_argument("--gt_root",      default=r"E:\data\QaTa-COV19-v2\100gt")
    # 脚本路径（你的 5 个文件）
    ap.add_argument("--step1_script", default=r"E:\DUPE-MedSAM\src\prompts\generator.py")
    ap.add_argument("--step2_script", default=r"E:\DUPE-MedSAM\src\prompts\stable_select.py")
    ap.add_argument("--step3_script", default=r"E:\DUPE-MedSAM\src\visual\box_generate.py")
    ap.add_argument("--step4_script", default=r"E:\DUPE-MedSAM\src\scoring\score_select.py")
    ap.add_argument("--step5_script", default=r"E:\DUPE-MedSAM\src\sam\infer.py")
    # SAM
    ap.add_argument("--sam_model_type", default="vit_b")
    ap.add_argument("--sam_checkpoint", default=r"E:\DUPE-MedSAM\segment-anything\sam_vit_b_01ec64.pth")
    ap.add_argument("--device",         default="cuda")
    # 循环 & 早停
    ap.add_argument("--work_root",      default=r"E:\data\QaTa-COV19-v2\AutoLoop9")
    ap.add_argument("--max_rounds",     type=int, default=6)
    ap.add_argument("--min_delta",      type=float, default=0.005)
    ap.add_argument("--patience",       type=int, default=1)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    gt_root     = Path(args.gt_root)
    work_root   = Path(args.work_root); ensure_dir(work_root)

    # —— R0：先跑 Step1/Step2 —— #
    raw_prompts_R0 = work_root / "R0_RawPrompts"
    stable_R0      = work_root / "R0_StablePromptsTxt"
    r0_index_jsonl = work_root / "R0_prompts_index.jsonl"
    r0_stable_jsonl= work_root / "R0_prompts_stable_n6.jsonl"
    ensure_dir(raw_prompts_R0); ensure_dir(stable_R0)

    print("[INIT] Step1 → Step2")
    run_step1(Path(args.step1_script), images_root, raw_prompts_R0, r0_index_jsonl)   # :contentReference[oaicite:10]{index=10}
    run_step2(Path(args.step2_script), r0_index_jsonl, r0_stable_jsonl, stable_R0)    # :contentReference[oaicite:11]{index=11}

    stems = [p.stem for p in list_images(images_root)]
    dice_hist = []; no_improve = 0

    for R in range(args.max_rounds):
        print(f"\n========== ROUND {R} ==========")
        R_root = work_root / f"R{R}"
        prompts_root = R_root / "PromptsTxt"   # 本轮 9 条/图
        boxes_root   = R_root / "Step3_Boxes"
        select_root  = R_root / "Step4_Scored"
        step5_root   = R_root / "Step5_SAM"
        for d in [prompts_root, boxes_root, select_root, step5_root]:
            ensure_dir(d)

        if R == 0:
            # 用 R0 选出的 Top-3 启动（脚本会从前3行读取）
            shutil.copytree(stable_R0, prompts_root, dirs_exist_ok=True)
        else:
            # 根据上一轮 final seg + best prompt 生成 9 条
            prev_select = work_root / f"R{R-1}" / "Step4_Scored"
            prev_step5  = work_root / f"R{R-1}" / "Step5_SAM"
            for stem in stems:
                sel = read_json(prev_select / f"{stem}_select.json")
                fin = read_json(prev_step5 / f"{stem}_final.json")
                if not sel or not fin: continue
                base_prompt = sel.get("best_prompt_text", "")
                m = bin_mask(Path(fin["final_mask"]))
                if m is None: continue
                pred_s = mask_stats(m)
                gt_mask = bin_mask(gt_root / f"{stem}.png")
                gt_s = mask_stats(gt_mask) if gt_mask is not None else None
                dsc = dice_coeff(m, gt_mask) if gt_mask is not None else -1.0

                struct3  = structural_variants(base_prompt, max_tokens=22)
                content3 = content_variants(base_prompt, pred_s, gt_s, dsc)
                prompts9 = cross_3x3_to_9(struct3, content3)
                (prompts_root / f"{stem}.txt").write_text("\n".join(prompts9), encoding="utf-8")

        # Step3 → 生成候选框
        run_step3(Path(args.step3_script), images_root, prompts_root, boxes_root, device=args.device)   # :contentReference[oaicite:12]{index=12}
        # Step4 → 选“最优 prompt + 3 框”
        run_step4(Path(args.step4_script), images_root, prompts_root, boxes_root, select_root, device=args.device)  # :contentReference[oaicite:13]{index=13}
        # Step5 → SAM + 多连通后处理
        run_step5(Path(args.step5_script), images_root, select_root, step5_root,
                  model_type=args.sam_model_type, checkpoint=Path(args.sam_checkpoint),
                  device=args.device, do_split_lr=True, keep_per_side=1)  # :contentReference[oaicite:14]{index=14}

        # 评估：final seg Dice
        dices=[]
        for stem in stems:
            info = read_json(step5_root / f"{stem}_final.json")
            if not info: continue
            pred = bin_mask(Path(info["final_mask"]))
            gt   = bin_mask(gt_root / f"{stem}.png")
            if pred is None or gt is None: continue
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            dices.append(dice_coeff(pred, gt))
        mean_dice = float(np.mean(dices)) if dices else -1.0
        dice_hist.append(mean_dice)
        write_json(R_root / "metrics.json", {"mean_dice": mean_dice, "n": len(dices)})
        print(f"[ROUND {R}] mean Dice(final seg) = {mean_dice:.4f}  (n={len(dices)})")

        # 早停（有 GT 才判断）
        if mean_dice >= 0 and len(dice_hist) >= 2:
            delta = dice_hist[-1] - dice_hist[-2]
            print(f"[ROUND {R}] ΔDice = {delta:.4f}")
            no_improve = no_improve + 1 if delta < args.min_delta else 0
            if no_improve > args.patience:
                print("[STOP] Dice 未显著提升，结束迭代。")
                break

    write_json(work_root / "dice_history.json", {"history": [{"round":i,"mean_dice":float(v)} for i,v in enumerate(dice_hist)]})
    try: input("\n任务完成，按回车退出...")
    except: pass

if __name__ == "__main__":
    main()
