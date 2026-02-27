
# scripts/batch_generate_prompts.py
# R0: 无 ctx → 通用生成；R1+: 有 ctx → 结构+内容反馈的“最小编辑”受控生成

import os, re, json, time, argparse, sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from getpass import getpass

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# ---------------- 基础配置 ----------------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

# R0 的通用指令（无 ctx 时使用）
PROMPT_INSTRUCTION = (
    "You are a board-certified radiologist. Generate EXACTLY 6 concise, distinct, "
    "English prompts describing ONLY pulmonary parenchymal or pleural ABNORMALITIES "
    "on chest radiographs of a COVID-19 suspected patient. "
    "Describe positive findings ONLY (no normal statements, no negations such as 'no', 'absent', "
    "'within normal limits', 'not prominent'). "
    "Do NOT mention heart, diaphragm, mediastinum, trachea, bones, vascular markings, or any unrelated structures. "
    "Each prompt must be one short sentence, clinical and neutral, no numbering, no patient identifiers. "
    "For example (style only, do not copy or restrict to this): "
    "Bilateral pulmonary infection, two infected areas, lower left lung and lower right lung. "
    "\n\nReturn ONLY a JSON array of 6 strings, e.g.:\n"
    '["Sentence 1","Sentence 2","Sentence 3","Sentence 4","Sentence 5","Sentence 6"]'
)

# ---------------- 文本清洗与约束 ----------------
STOP_SENTENCES = [
    r"\bdescribe distribution, extent and symmetry\.?",
]
STOP_WORDS = {
    "the","a","an","mild","slight","probable","possible",
    "likely","suspicious","suggestive","of"
}

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.{2,}$", ".", s)
    if not s.endswith("."): s += "."
    return s

def strip_filler(s: str) -> str:
    s = normalize_text(s)
    for pat in STOP_SENTENCES:
        s = re.sub(pat, "", s, flags=re.I)
    for w in STOP_WORDS:
        s = re.sub(rf"\b{re.escape(w)}\b", "", s, flags=re.I)
    s = re.sub(r"(^|\.\s*)Lungs:\s*", r"\1", s)  # 去冗余前缀
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s.endswith("."): s += "."
    return s

def limit_len(s: str, lo: int, hi: int) -> str:
    toks = s.split()
    if len(toks) > hi:
        s = " ".join(toks[:hi]).rstrip(" .") + "."
    if len(toks) < lo:
        s = s.rstrip(".") + "."
    return s

def post_clean_and_constrain(
    prompts: List[str],
    lo: int, hi: int,
    seg_facts: Dict[str, Any]
) -> List[str]:
    """对 LLM 输出做二次收敛：清洗 + 长度约束 + 关键词注入/替换（侧性、病灶数、位置）。"""
    outs, seen = [], set()

    side = str(seg_facts.get("side", "")).lower()            # 'unilateral' / 'bilateral' / 'unknown'
    lesions = seg_facts.get("lesion_count", None)            # int or None
    locs = seg_facts.get("locations", [])                    # ['left','right','upper','middle','lower',...]

    def loc_phrase(locs: List[str]) -> Optional[str]:
        if not locs: return None
        lr = []
        if "left" in locs:  lr.append("left lung")
        if "right" in locs: lr.append("right lung")
        z = []
        if "upper" in locs:  z.append("upper")
        if "middle" in locs: z.append("middle")
        if "lower" in locs:  z.append("lower")
        tail = (" " + "-".join(z) + " zone") if z else ""
        if lr: return ", ".join(lr) + tail
        if z:  return "-".join(z) + " zone"
        return None

    loc_str = loc_phrase(locs)

    for p in prompts:
        q = strip_filler(p)

        # 侧性替换：优先贴合分割事实
        if side == "unilateral":
            q = re.sub(r"\bbi-?lateral\b", "unilateral", q, flags=re.I)
        elif side == "bilateral":
            q = re.sub(r"\buni-?lateral\b", "bilateral", q, flags=re.I)

        # 病灶数注入：若有 lesion/area/focus 词根则替换或补齐
        if lesions is not None:
            if re.search(r"\b(lesion|area|focus|foci|opacity|opacities)\b", q, flags=re.I):
                q = re.sub(r"\b\d+\s+(lesion|area|focus|foci)\b", f"{lesions} \\1", q, flags=re.I)
                if not re.search(r"\b\d+\s+(lesion|area|focus|foci)\b", q, flags=re.I):
                    q = re.sub(r"(unilateral|bilateral)(\s+)", rf"\1\2{lesions} lesions, ", q, count=1, flags=re.I)

        # 位置注入：句中无左右信息时，在侧性后拼接
        if loc_str and not re.search(r"\b(left|right)\b", q, flags=re.I):
            q = re.sub(r"\b(unilateral|bilateral)\b", rf"\1, {loc_str}", q, count=1, flags=re.I)

        q = limit_len(q, lo, hi)
        if q not in seen and len(q.split()) >= max(6, lo-2):
            outs.append(q); seen.add(q)

    return outs[:6]

# ---------------- ctx 支持：R1+ 结构+内容反馈 ----------------
def read_ctx(ctx_dir: Optional[str], stem: str) -> Optional[dict]:
    if not ctx_dir: return None
    p = Path(ctx_dir) / f"{stem}.json"
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def build_guided_instruction(best_struct: str, target_len: int, seg_facts: dict) -> str:
    lo, hi = max(8, target_len-2), min(40, target_len+2)
    side      = seg_facts.get("side", "unknown")
    lesions   = seg_facts.get("lesion_count", "unknown")
    locations = ", ".join(seg_facts.get("locations", [])) or "unspecified"
    return (
        "You are a board-certified radiologist. "
        "Generate EXACTLY 6 English prompts that are MINIMAL edits of the base sentence. "
        "Keep sentence structure and style close to the base; use synonyms/light rewording only. "
        "Do not invent new findings. No numbering or bullets. "
        # ★ 加限制：不要引入肺外结构
        "Do not mention heart, mediastinum, diaphragm, bones, or any structures outside the lungs/pleura.\n\n"
        f"Target length: about {target_len} words (allowed range {lo}-{hi}).\n"
        "Clinical facts to prefer if wording choices matter (do not add new content):\n"
        f"- Laterality: {side}\n"
        f"- Lesion count (approx): {lesions}\n"
        f"- Locations: {locations}\n\n"
        f"Base sentence:\n\"{best_struct}\"\n\n"
        "Return ONLY a JSON array of 6 strings, e.g.:\n"
        '["Sentence 1","Sentence 2","Sentence 3","Sentence 4","Sentence 5","Sentence 6"]'
    )


# ---------------- 与 OpenAI 通讯 ----------------
def parse_json_array(text: str) -> List[str]:
    # 1) 去掉 ```json / ``` 代码块围栏
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped, flags=re.I)

    # 2) 优先从文本中抓取第一段 JSON 数组
    m = re.search(r"\[[\s\S]*\]", stripped)
    candidate = m.group(0) if m else stripped

    try:
        arr = json.loads(candidate)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr]
    except Exception:
        pass

    # 3) 兜底：把每行非空当一句（去掉项目符号/编号）
    lines = [re.sub(r"^\s*[-*\d\.\)]\s*", "", ln).strip()
             for ln in stripped.splitlines() if ln.strip()]
    return lines

def call_gpt_with_instruction(client: OpenAI, model: str, instruction: str, want_n: int = 6) -> List[str]:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": instruction}],
                temperature=0.2,  # 受控生成更“克制”
            )
            content = resp.choices[0].message.content.strip()
            arr = parse_json_array(content)
            arr = [x.strip() for x in arr if x and isinstance(x, str)]
            if len(arr) >= want_n:
                return arr[:want_n]
            raise ValueError(f"Output not valid: {content[:200]}...")
        except (RateLimitError, APITimeoutError):
            if attempt == MAX_RETRIES: raise
            time.sleep(backoff); backoff *= 2
        except APIError as e:
            if getattr(e, "status", None) in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(backoff); backoff *= 2
            else:
                raise
    raise RuntimeError("Exceeded max retries")

# ---------------- IO 帮手 ----------------
def save_txt(prompts: List[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(p.strip() for p in prompts), encoding="utf-8")

def append_jsonl(record: dict, jsonl_path: Path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def should_skip(out_txt: Path, overwrite: bool) -> bool:
    return out_txt.exists() and not overwrite

# ---------------- 主处理 ----------------
def process_folder(
    input_dir: Path,
    output_dir: Path,
    jsonl_path: Path,
    client: OpenAI,
    model: str,
    overwrite: bool = False,
    limit: Optional[int] = None,
    ctx_dir: Optional[str] = None,   # 新增：R1+ 传 ctx_dir 即启用受控
):
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(sorted(input_dir.rglob(f"*{ext}")))
    if not images:
        print(f"[WARN] No images found under {input_dir}")
        return

    if limit is not None:
        images = images[:limit]

    total = len(images)
    print(f"[INFO] Found {total} images.")

    for idx, img_path in enumerate(images, start=1):
        stem = img_path.stem
        rel  = img_path.relative_to(input_dir)
        out_txt = output_dir / rel.with_suffix(".txt")

        if should_skip(out_txt, overwrite):
            if idx % 50 == 0 or idx == total:
                print(f"[SKIP] {idx}/{total}: {rel}")
            continue

        try:
            # R1+: 若有 ctx → 受控生成；否则（R0）走通用指令
            ctx = read_ctx(ctx_dir, stem)
            if ctx and ctx.get("best_struct_prompt"):
                target_len = int(ctx.get("target_len", 20))
                instr = build_guided_instruction(
                    best_struct = ctx["best_struct_prompt"],
                    target_len  = target_len,
                    seg_facts   = ctx.get("seg_facts", {})
                )
                raw = call_gpt_with_instruction(client, model, instr, want_n=6)
                prompts = post_clean_and_constrain(
                    raw, lo=max(8, target_len-2), hi=min(40, target_len+2),
                    seg_facts=ctx.get("seg_facts", {})
                )
            else:
                # R0：通用指令
                instr = PROMPT_INSTRUCTION
                raw = call_gpt_with_instruction(client, model, instr, want_n=6)
                prompts = [strip_filler(p) for p in raw][:6]

            save_txt(prompts, out_txt)
            append_jsonl({"image": str(rel).replace("\\", "/"), "prompts": prompts}, jsonl_path)

            if idx % 25 == 0 or idx == total:
                mode = "GUIDED" if ctx and ctx.get("best_struct_prompt") else "GEN"
                print(f"[DONE-{mode}] {idx}/{total}: {rel}")

        except Exception as e:
            print(f"[ERROR] {rel}: {e}")

# ---------------- CLI ----------------
def build_parser():
    parser = argparse.ArgumentParser(description="R0: general; R1+: guided (structure+seg facts) prompt generator.")
    # 双击可跑的默认值（按需改）
    parser.add_argument("--input_dir",  type=str, default=r"E:\data\QaTa-COV19-v2\100train", help="影像根目录")
    parser.add_argument("--output_dir", type=str, default=r"E:\data\QaTa-COV19-v2\PromptTxts", help="逐图 TXT 输出目录")
    parser.add_argument("--jsonl",      type=str, default=r"E:\data\QaTa-COV19-v2\prompts_index.jsonl", help="汇总 JSONL")
    parser.add_argument("--ctx_dir",    type=str, default=r"E:\data\QaTa-COV19-v2\ctx", help="可选：Rk/ctx，含 {stem}.json；存在则启用受控生成")
    parser.add_argument("--api_key",    type=str, default="..", help="OpenAI API Key（建议用环境变量 OPENAI_API_KEY）")
    parser.add_argument("--model",      type=str, default="gpt-4o", help="OpenAI 模型")
    parser.add_argument("--overwrite",  action="store_true", help="已存在是否覆盖")
    parser.add_argument("--limit",      type=int, default=None, help="仅处理前 N 张（调试）")
    return parser

def pause_exit(code: int):
    try:
        input("\n任务结束。按回车关闭窗口...")
    except Exception:
        pass
    sys.exit(code)

def main():
    parser = build_parser()
    args = parser.parse_args()  # 无参数即用默认值

    # API Key：环境变量优先。没设置就让用户输入一次（便于双击）
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = getpass("请输入 OpenAI API Key（输入不可见）: ").strip()
        except Exception:
            api_key = input("请输入 OpenAI API Key: ").strip()
        if not api_key:
            print("[FATAL] 缺少 API Key。")
            return pause_exit(1)

    print("===== 配置 =====")
    print("input_dir :", args.input_dir)
    print("output_dir:", args.output_dir)
    print("jsonl_path:", args.jsonl)
    print("ctx_dir   :", args.ctx_dir or "(None → R0 general)")
    print("model     :", args.model)
    print("overwrite :", args.overwrite)
    print("limit     :", args.limit)
    print("===============")

    try:
        client = OpenAI(api_key=api_key)
        process_folder(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            jsonl_path=Path(args.jsonl),
            client=client,
            model=args.model,
            overwrite=args.overwrite,
            limit=args.limit,
            ctx_dir=args.ctx_dir,   # 关键：R0 无 ctx；R1+ 传 ctx
        )
    except Exception as e:
        print(f"[FATAL] {e}")
        return pause_exit(1)


if __name__ == "__main__":
    main()
