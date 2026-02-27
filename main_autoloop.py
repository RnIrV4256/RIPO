# main_autoloop.py
# R0→R1→… 循环：Step1(new_generate)→Step2(稳定筛Top-3)→Step3(候选框)→Step4(扰动评分)→Step5(SAM+按左右侧拼接)
# 每轮后：读取 Step5 final_mask，与 GT(E:\data\QaTa-COV19-v2\100gt) 计算 Dice，维护每图最佳；支持 per-image 早停
# 最终：将每图最佳掩膜复制到 Final/masks，并输出 results.csv

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
import csv

from typing import Optional

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

# ---------------- 基础 I/O ----------------
def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(p: Path):
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_json(p: Path, obj):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- Dice 计算 ----------------
def _read_mask01(path: Path, target_hw=None):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    if target_hw and (m.shape != target_hw):
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)

def dice_coeff(pred01: np.ndarray, gt01: np.ndarray, eps: float=1e-6) -> float:
    inter = float((pred01 & gt01).sum())
    s = float(pred01.sum() + gt01.sum())
    return 2.0 * inter / (s + eps) if s > 0 else 1.0 if inter == 0 else 0.0

def find_final_mask(seg_dir: Path, stem: str):
    """
    优先读 step5 写的汇总 json 里的路径；否则按约定路径兜底：
      1) seg_dir / f"{stem}_final.json" → get ["final_mask"]
      2) seg_dir / stem / f"{stem}_final_mask.png"
      3) seg_dir / f"{stem}_final_mask.png"
    """
    j = read_json(seg_dir / f"{stem}_final.json")
    if j and "final_mask" in j:
        p = Path(j["final_mask"])
        if p.exists(): return p
    p2 = seg_dir / stem / f"{stem}_final_mask.png"
    if p2.exists(): return p2
    p3 = seg_dir / f"{stem}_final_mask.png"
    if p3.exists(): return p3
    return None

# ---------------- 构建回灌 ctx（供 R1+ 受控生成）----------------
def build_ctx_for_round(prev_round_root: Path, ctx_dir: Path):
    sel_dir = prev_round_root / "Step4_Scored"
    seg_dir = prev_round_root / "Step5_SAM"
    ensure_dir(ctx_dir)

    for sel_path in sorted(sel_dir.glob("*_select.json")):
        stem = sel_path.stem.replace("_select","")
        sel = read_json(sel_path) or {}
        best_prompt = sel.get("best_prompt_text","").strip()

        left_area = right_area = 0
        sum_json = seg_dir / f"{stem}_final.json"
        if sum_json.exists():
            sj = read_json(sum_json) or {}
            stats = sj.get("stitch_stats", {})
            left_area  = int(stats.get("left_area", 0))
            right_area = int(stats.get("right_area",0))
        else:
            left_png  = seg_dir / stem / f"{stem}_left_mask.png"
            right_png = seg_dir / stem / f"{stem}_right_mask.png"
            if left_png.exists():
                m = cv2.imread(str(left_png), cv2.IMREAD_GRAYSCALE); left_area = int((m>127).sum())
            if right_png.exists():
                m = cv2.imread(str(right_png), cv2.IMREAD_GRAYSCALE); right_area = int((m>127).sum())
        lr_ratio = float(left_area) / float(right_area+1e-6) if (left_area or right_area) else 0.0

        ctx = {
            "best_struct_prompt": best_prompt,
            "seg_stats": {
                "left_area": left_area,
                "right_area": right_area,
                "lr_ratio": lr_ratio
            },
            "target_len": max(8, min(32, len(best_prompt.split()))) if best_prompt else 16
        }
        write_json(ctx_dir / f"{stem}.json", ctx)

# ---------------- 调度外部步骤 ----------------
def run_step1_new_generate(python_exe: str, script: Path,
                           images_root: Path, out_prompts_dir: Path,
                           ctx_dir: Optional[Path]):
    """
    new_generator.py 的参数是:
      --input_dir / --output_dir / --jsonl / [--ctx_dir]
    """
    out_prompts_dir = Path(out_prompts_dir)
    out_prompts_dir.mkdir(parents=True, exist_ok=True)

    # 每一轮把索引写在同级，便于 Step2 读取
    jsonl_path = out_prompts_dir.parent / "prompts_index.jsonl"

    cmd = [python_exe, str(script),
           "--input_dir",  str(images_root),
           "--output_dir", str(out_prompts_dir),
           "--jsonl",      str(jsonl_path)]
    if ctx_dir is not None:
        cmd += ["--ctx_dir", str(ctx_dir)]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_step2_stability_filter(python_exe: str, script: Path,
                               images_root: Path, gen_dir: Path, out_top3_dir: Path,
                               api_key: Optional[str]=None, model: Optional[str]=None):
    rk_root = Path(gen_dir).parent
    input_jsonl = rk_root / "prompts_index.jsonl"
    output_jsonl = rk_root / "stable_select.jsonl"

    out_top3_dir = Path(out_top3_dir);
    out_top3_dir.mkdir(parents=True, exist_ok=True)

    cmd = [python_exe, str(script),
           "--input_jsonl", str(input_jsonl),
           "--output_jsonl", str(output_jsonl),
           "--output_txt_root", str(out_top3_dir)]
    if api_key:
        cmd += ["--api_key", api_key]
    if model:
        cmd += ["--model", model]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_step3_boxes(python_exe: str, script: Path,
                    images_root: Path, prompts_txt_dir: Path, out_boxes_dir: Path):
    ensure_dir(out_boxes_dir)
    cmd = [python_exe, str(script),
           "--images_root", str(images_root),
           "--prompts_txt_root", str(prompts_txt_dir),
           "--out_dir", str(out_boxes_dir)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_step4_select(python_exe: str, script: Path,
                     images_root: Path, prompts_txt_dir: Path, boxes_dir: Path, out_scored_dir: Path):
    ensure_dir(out_scored_dir)
    cmd = [python_exe, str(script),
           "--images_root", str(images_root),
           "--prompts_txt_root", str(prompts_txt_dir),
           "--boxes_root", str(boxes_dir),
           "--out_root", str(out_scored_dir)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_step5_seg(python_exe: str, script: Path,
                  images_root: Path, select_root: Path, out_root: Path):
    ensure_dir(out_root)
    cmd = [python_exe, str(script),
           "--images_root", str(images_root),
           "--select_root", str(select_root),
           "--out_root", str(out_root)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ---------------- 主控循环（含早停与最终汇总） ----------------
def main():
    ap = argparse.ArgumentParser("AutoLoop 主控：R0→R1→…（回灌 + 早停 + 最优汇总 + Dice）")
    # 根路径与数据
    ap.add_argument("--work_root",   default=r"E:\data\QaTa-COV19-v2\AutoLoop5")
    ap.add_argument("--images_root", default=r"E:\data\QaTa-COV19-v2\100train")
    ap.add_argument("--gt_root",     default=r"E:\data\QaTa-COV19-v2\100gt")
    ap.add_argument("--python_exe",  default="python")
    # 子脚本
    ap.add_argument("--step1_script", default=r"E:\DUPE-MedSAM\src\prompts\new_generator.py")
    ap.add_argument("--step2_script", default=r"E:\DUPE-MedSAM\src\prompts\stable_select.py")
    ap.add_argument("--step3_script", default=r"E:\DUPE-MedSAM\src\visual\box_generate_v2.py")
    ap.add_argument("--step4_script", default=r"E:\DUPE-MedSAM\src\scoring\score_select_v2.py")
    ap.add_argument("--step5_script", default=r"E:\DUPE-MedSAM\src\sam\segment.py")  # 左右侧拼接版
    # 轮次与早停
    ap.add_argument("--max_rounds",   type=int, default=10)
    ap.add_argument("--patience_img", type=int, default=2)  # 连续多少轮无提升→早停
    args = ap.parse_args()

    work_root   = Path(args.work_root)
    images_root = Path(args.images_root)
    gt_root     = Path(args.gt_root)
    ensure_dir(work_root)

    # 构建图像清单
    img_paths = list_images(images_root)
    stems = [p.stem for p in img_paths]
    print(f"[INFO] total images: {len(stems)}")

    # R0：若无现成 Top-3，则先产一份
    r0 = work_root / "R0"
    ensure_dir(r0)
    r0_prompts = r0 / "PromptsTxt"
    if not any(r0_prompts.glob("*.txt")):
        run_step1_new_generate(args.python_exe, Path(args.step1_script),
                               images_root, out_prompts_dir=r0/"GeneratedPrompts", ctx_dir=None)
        run_step2_stability_filter(args.python_exe, Path(args.step2_script),
                                   images_root, gen_dir=r0/"GeneratedPrompts", out_top3_dir=r0_prompts)

    # 早停状态 & 最佳记录（持久化）
    status_path = work_root / "best_status.json"
    if status_path.exists():
        best_status = read_json(status_path) or {}
    else:
        best_status = {}
    # 初始化
    for s in stems:
        if s not in best_status:
            best_status[s] = {
                "best_dice": -1.0,
                "best_round": -1,
                "best_mask": "",
                "no_improve_rounds": 0,
                "frozen": False
            }

    # 轮次循环
    for R in range(args.max_rounds):
        Rk = work_root / f"R{R}"
        ensure_dir(Rk)
        print(f"\n========== ROUND R{R} ==========")

        # 本轮路径
        prompts_dir = Rk / "PromptsTxt"
        boxes_dir   = Rk / "Step3_Boxes"
        scored_dir  = Rk / "Step4_Scored"
        seg_dir     = Rk / "Step5_SAM"
        ensure_dir(prompts_dir); ensure_dir(boxes_dir); ensure_dir(scored_dir); ensure_dir(seg_dir)

        # R1+：构建回灌 ctx → 受控生成 6 条 → 稳定筛 Top-3
        if R == 0:
            print("[INFO] R0 使用已有 PromptsTxt")
        else:
            ctx_dir = Rk / "ctx"
            build_ctx_for_round(work_root / f"R{R-1}", ctx_dir)
            run_step1_new_generate(args.python_exe, Path(args.step1_script),
                                   images_root, out_prompts_dir=Rk/"GeneratedPrompts", ctx_dir=ctx_dir)
            run_step2_stability_filter(args.python_exe, Path(args.step2_script),
                                       images_root, gen_dir=Rk/"GeneratedPrompts", out_top3_dir=prompts_dir)

        # Step3→4→5
        run_step3_boxes(args.python_exe, Path(args.step3_script),
                        images_root, prompts_txt_dir=prompts_dir, out_boxes_dir=boxes_dir)
        run_step4_select(args.python_exe, Path(args.step4_script),
                         images_root, prompts_txt_dir=prompts_dir, boxes_dir=boxes_dir, out_scored_dir=scored_dir)
        run_step5_seg(args.python_exe, Path(args.step5_script),
                      images_root, select_root=scored_dir, out_root=seg_dir)

        # 本轮评估：与 GT 计算 Dice，更新最佳 & 早停计数
        round_csv = Rk / "metrics_round.csv"
        with open(round_csv, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["stem", "dice", "best_dice_so_far", "frozen"])
            for stem in stems:
                # 若已经早停，我们仍然读取 Dice（用于观测），但不再影响最佳
                pred_path = find_final_mask(seg_dir, stem)
                if pred_path is None:
                    dice = -1.0
                else:
                    gt_path = gt_root / f"mask_{stem}.png"
                    if not gt_path.exists():
                        # 没 GT 就跳过评估（dice=-1）
                        dice = -1.0
                    else:
                        gt01   = _read_mask01(gt_path)
                        pred01 = _read_mask01(pred_path, target_hw=gt01.shape if gt01 is not None else None)
                        dice = dice_coeff(pred01, gt01) if (gt01 is not None and pred01 is not None) else -1.0

                rec = best_status[stem]
                frozen = rec.get("frozen", False)

                if dice >= 0:
                    if dice > rec["best_dice"] + 1e-8:
                        # 有提升：更新最佳，清空 no_improve
                        rec["best_dice"] = float(dice)
                        rec["best_round"] = int(R)
                        rec["best_mask"]  = str(pred_path) if pred_path else ""
                        rec["no_improve_rounds"] = 0
                    else:
                        # 无提升：累加 no_improve，达到耐心阈值 → 冻结
                        rec["no_improve_rounds"] = int(rec.get("no_improve_rounds",0)) + 1
                        if not frozen and rec["no_improve_rounds"] >= args.patience_img:
                            rec["frozen"] = True
                # 记录到本轮 csv
                w.writerow([stem, f"{dice:.4f}" if dice>=0 else "",
                            f"{rec['best_dice']:.4f}" if rec["best_dice"]>=0 else "",
                            str(rec.get("frozen", False))])

        # 保存状态快照
        write_json(status_path, best_status)
        print(f"[INFO] round R{R} evaluated. Metrics saved to {round_csv}")

    # 所有轮次结束：把每图最佳掩膜复制到 Final，并汇总 CSV
    final_dir = ensure_dir(work_root / "Final" / "masks")
    results_csv = work_root / "Final" / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["stem", "best_dice", "best_round", "best_mask_relpath"])
        total_dice = 0.0
        count = 0
        for stem in stems:
            rec = best_status.get(stem, {})
            best_mask = rec.get("best_mask", "")
            best_dice = rec.get("best_dice", -1.0)
            best_round = rec.get("best_round", -1)
            if best_mask and Path(best_mask).exists():
                dst = final_dir / f"mask_{stem}.png"
                shutil.copy2(best_mask, dst)
                best_rel = str(dst.relative_to(work_root))
            else:
                best_rel = ""

            if best_dice >= 0:
                total_dice += best_dice
                count += 1

            w.writerow([stem, f"{best_dice:.4f}" if best_dice >= 0 else "",
                        best_round if best_round >= 0 else "", best_rel])

    # 计算并输出平均Dice
    avg_dice = total_dice / count if count > 0 else 0.0
    print(f"\n[DONE] 全部轮次完成。最佳掩膜已复制到：{final_dir}")
    print(f"[DONE] 评估表：{results_csv}")
    print(f"[STATS] 平均 Dice: {avg_dice:.4f}")


if __name__ == "__main__":
    main()
