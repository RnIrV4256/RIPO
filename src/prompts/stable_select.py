import os, sys, math, time, argparse, json
from pathlib import Path
from typing import List
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

# ============ 配置 ============
EMBED_MODEL = "text-embedding-3-large"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

def cosine(u: List[float], v: List[float]) -> float:
    num = sum(a*b for a,b in zip(u,v))
    du = math.sqrt(sum(a*a for a in u))
    dv = math.sqrt(sum(b*b for b in v))
    return 0.0 if du==0 or dv==0 else num/(du*dv)

def embed_with_retry(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except (RateLimitError, APITimeoutError):
            if attempt == MAX_RETRIES: raise
            time.sleep(backoff); backoff *= 2
        except APIError as e:
            if getattr(e,"status",None) in (500,502,503,504) and attempt<MAX_RETRIES:
                time.sleep(backoff); backoff *= 2
            else:
                raise
    raise RuntimeError("embedding retries exceeded")

def set_stability(embeds: List[List[float]]):
    n = len(embeds)
    sims = []
    worst = (0,1,1.0)
    for i in range(n):
        for j in range(i+1,n):
            s = cosine(embeds[i],embeds[j])
            sims.append((i,j,s))
            if s<worst[2]:
                worst = (i,j,s)
    mu = sum(s for _,_,s in sims)/len(sims)
    var = sum((s-mu)**2 for _,_,s in sims)/len(sims)
    sigma = math.sqrt(var)
    instability = 1-mu
    s_list = [0.0]*n; counts=[0]*n
    for i,j,s in sims:
        s_list[i]+=s; s_list[j]+=s
        counts[i]+=1; counts[j]+=1
    s_list=[ s_list[i]/counts[i] if counts[i]>0 else 0.0 for i in range(n)]
    return mu,sigma,instability,worst,s_list

def select_top3(prompts: List[str], scores: List[float]) -> List[int]:
    return sorted(range(len(prompts)), key=lambda i:scores[i], reverse=True)[:3]

def process_file(client: OpenAI, in_jsonl: Path, out_jsonl: Path, out_txt_root: Path, model: str):
    count=0
    with open(in_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            item=json.loads(line)
            prompts=item.get("prompts",[])
            if len(prompts)!=6:
                continue
            embeds=embed_with_retry(client,prompts,model)
            mu,sigma,instab,worst,s_list=set_stability(embeds)
            top3=select_top3(prompts,s_list)
            selected=[prompts[i] for i in top3]
            rec={
                "image": item.get("image"),
                "mu":mu,"sigma":sigma,"instability":instab,
                "worst_pair":worst,
                "scores":s_list,
                "top3_idx":top3,
                "selected":selected
            }
            # 保存 JSONL
            with open(out_jsonl,"a",encoding="utf-8") as wf:
                wf.write(json.dumps(rec,ensure_ascii=False)+"\n")
            # 保存逐图 TXT
            if item.get("image"):
                out_txt=out_txt_root/Path(item["image"]).with_suffix(".txt")
                out_txt.parent.mkdir(parents=True,exist_ok=True)
                with open(out_txt,"w",encoding="utf-8") as tf:
                    for s in selected:
                        tf.write(s.strip()+"\n")
            count+=1
    print(f"[DONE] processed {count} items")

def main():
    parser=argparse.ArgumentParser()
    # 在 default 里直接写死路径 ↓↓↓
    parser.add_argument("--input_jsonl", default=r"E:\data\QaTa-COV19-v2\prompts_index.jsonl",
                        help="第一步生成的6条prompts JSONL")
    parser.add_argument("--output_jsonl", default=r"E:\data\QaTa-COV19-v2\prompts_stable_n6.jsonl",
                        help="输出的稳定性结果 JSONL")
    parser.add_argument("--output_txt_root", default=r"E:\data\QaTa-COV19-v2\StablePromptsTxt",
                        help="输出的Top3逐图 TXT 文件夹")
    parser.add_argument("--api_key", default="..",
                        help="OpenAI API Key（建议用环境变量 OPENAI_API_KEY）")
    parser.add_argument("--model", default=EMBED_MODEL,
                        help="使用的embedding模型")
    args=parser.parse_args()

    if not args.api_key:
        raise SystemExit("缺少 API Key，请设置 OPENAI_API_KEY 环境变量或在 args 里改 default")

    client=OpenAI(api_key=args.api_key)
    process_file(client, Path(args.input_jsonl), Path(args.output_jsonl), Path(args.output_txt_root), args.model)
    print("finish")


if __name__=="__main__":
    main()