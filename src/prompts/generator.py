# scripts/batch_generate_prompts.py
import os, re, json, time, argparse, sys
from pathlib import Path
from typing import List, Optional
from getpass import getpass
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# -------- 基本配置（放在 argparse 的 default 里） --------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

PROMPT_INSTRUCTION = (
    "You are a board-certified radiologist. Generate EXACTLY 6 concise, distinct, "
    "English prompts suitable for describing chest radiograph findings in a COVID-19 "
    "suspected patient (QaTa-COVID19 style). Each prompt should be one short sentence, "
    "clinical and neutral, without numbering or bullets, no patient identifiers.\n\n"
    "Return ONLY a JSON array of 6 strings, e.g.:\n"
    '["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5", "Sentence 6"]'
)

def clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*[-*\d\.\)]\s*", "", s)
    return s

def call_gpt(client: OpenAI, model: str) -> List[str]:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT_INSTRUCTION}],
                temperature=0.7,
            )
            content = resp.choices[0].message.content.strip()

            # 首选 JSON 解析
            try:
                data = json.loads(content)
                if isinstance(data, list) and len(data) == 6 and all(isinstance(x, str) for x in data):
                    return [x.strip() for x in data]
            except json.JSONDecodeError:
                pass

            # 兜底：逐行抽取
            lines = [clean_line(line) for line in content.splitlines() if clean_line(line)]
            if len(lines) >= 6:
                return lines[:6]

            raise ValueError(f"Output not valid: {content[:200]}...")
        except (RateLimitError, APITimeoutError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff); backoff *= 2
        except APIError as e:
            if getattr(e, "status", None) in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(backoff); backoff *= 2
            else:
                raise
    raise RuntimeError("Exceeded max retries")

def save_txt(prompts: List[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.strip() + "\n")

def append_jsonl(record: dict, jsonl_path: Path):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def should_skip(out_txt: Path, overwrite: bool) -> bool:
    return out_txt.exists() and not overwrite

def process_folder(
    input_dir: Path,
    output_dir: Path,
    jsonl_path: Path,
    client: OpenAI,
    model: str,
    overwrite: bool = False,
    limit: Optional[int] = None,
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
        rel = img_path.relative_to(input_dir)
        out_txt = output_dir / rel.with_suffix(".txt")

        if should_skip(out_txt, overwrite):
            if idx % 50 == 0 or idx == total:
                print(f"[SKIP] {idx}/{total}: {rel}")
            continue

        try:
            prompts = call_gpt(client, model)
            save_txt(prompts, out_txt)
            append_jsonl({"image": str(rel).replace("\\", "/"), "prompts": prompts}, jsonl_path)
            if idx % 25 == 0 or idx == total:
                print(f"[DONE] {idx}/{total}: {rel}")
        except Exception as e:
            print(f"[ERROR] {rel}: {e}")

def build_parser():
    parser = argparse.ArgumentParser(description="Batch-generate 6 prompts per chest X-ray using OpenAI GPT.")

    # —— 在这里把“默认值”写死到 args 里（双击运行时会自动使用）——
    parser.add_argument("--input_dir",  type=str, default=r"E:\data\QaTa-COV19-v2\100train",
                        help="包含影像的根目录")
    parser.add_argument("--output_dir", type=str, default=r"E:\data\QaTa-COV19-v2\PromptsTxt",
                        help="保存逐图 TXT 的目录")
    parser.add_argument("--jsonl",  type=str, default=r"E:\data\QaTa-COV19-v2\prompts_index.jsonl",
                        help="汇总 JSONL 文件路径")
    parser.add_argument("--api_key",   type=str, default="..",
                        help="OpenAI API Key")
    parser.add_argument("--model",     type=str, default="gpt-4o",
                        help="OpenAI 模型名称")
    parser.add_argument("--overwrite", action="store_true",
                        help="若已存在 TXT 是否覆盖（默认不覆盖）")
    parser.add_argument("--limit",     type=int, default=None,
                        help="仅处理前 N 张（调试用），None 表示全部")
    return parser

def pause_exit(code: int):
    try:
        input("\n任务结束。按回车关闭窗口...")
    except Exception:
        pass
    sys.exit(code)

def main():
    # 双击运行时没有命令行参数，这里直接使用 defaults；若之后你想覆盖，命令行传参即可
    parser = build_parser()
    args = parser.parse_args()  # 无参数 → 使用 default

    # API Key：环境变量优先；没有就交互式输入一次
    api_key = args.api_key
    if not api_key:
        print("未检测到 OPENAI_API_KEY 环境变量。")
        try:
            api_key = getpass("请输入 OpenAI API Key（输入不可见）: ").strip()
        except Exception:
            api_key = input("请输入 OpenAI API Key: ").strip()
        if not api_key:
            print("[FATAL] 缺少 API Key。")
            return pause_exit(1)

    # 打印配置，便于核对
    print("===== 配置 =====")
    print("input_dir :", args.input_dir)
    print("output_dir:", args.output_dir)
    print("jsonl_path:", args.jsonl)
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
        )
    except Exception as e:
        print(f"[FATAL] {e}")
        return pause_exit(1)

    pause_exit(0)

if __name__ == "__main__":
    main()
