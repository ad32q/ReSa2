from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from tevatron.preprocessor import MarcoPassageTrainPreProcessor as TrainPreProcessor
import math
import random

def load_ranking(rank_file, relevance, n_sample, depth, tau=0.5):
    """从排序文件中加载数据，并按概率选择负样本
    
    Args:
        rank_file (str): 排序文件路径，格式为[query, _, paragraph, _, score, _]
        relevance (dict): 每个查询对应的正样本段落集合 {q: {positive_paragraphs}}
        n_sample (int): 需要采样的负样本数量
        depth (int): 考虑负样本的最大排名深度
        tau (float): 概率计算中的温度系数，默认值1/4
    
    Yields:
        tuple: (当前查询, 正样本列表, 采样的负样本列表)
    """
    with open(rank_file) as rf:
        lines = iter(rf)
        
        # 初始化第一个查询
        try:
            parts = next(lines).strip().split()
            curr_q, curr_p = parts[0], parts[2]
            curr_score = float(parts[4])
            
            pos_scores = [curr_score] if curr_p in relevance.get(curr_q, set()) else []
            negatives = [] if curr_p in relevance.get(curr_q, set()) else [(curr_p, curr_score)]
        except StopIteration:
            return  # 空文件直接返回

        def _sample_negatives(pos_scores, negatives):
            """根据正负样本得分生成采样结果"""
            # 无正样本时回退随机采样
            if not pos_scores:
                candidates = [p for p, _ in negatives[:depth]]
                return random.sample(candidates, min(n_sample, len(candidates)))
            
            # 计算概率权重
            pos_max = max(pos_scores)  # 使用最高正样本得分作为基准
            candidates = negatives[:depth]
            if not candidates:
                return []
            
            # 计算指数权重
            weights = [math.exp(-(s - pos_max)**2 * tau) for _, s in candidates]
            total = sum(weights)
            
            # 处理全零权重情况
            if total <= 1e-8:
                return random.sample([p for p, _ in candidates], min(n_sample, len(candidates)))
            
            # 概率归一化
            probs = [w / total for w in weights]
            
            # 无放回加权采样
            sampled = []
            remain_indices = list(range(len(candidates)))
            remain_probs = probs.copy()
            
            for _ in range(min(n_sample, len(candidates))):
                # 轮盘赌选择
                r = random.uniform(0, sum(remain_probs))
                accum = 0
                for i in range(len(remain_probs)):
                    accum += remain_probs[i]
                    if accum >= r:
                        chosen_idx = remain_indices[i]
                        sampled.append(candidates[chosen_idx][0])
                        # 移除已选样本
                        del remain_indices[i]
                        del remain_probs[i]
                        # 重新归一化剩余概率
                        if remain_probs:
                            sum_remain = sum(remain_probs)
                            remain_probs = [p/sum_remain for p in remain_probs]
                        break
            return sampled

        # 主循环处理后续行
        while True:
            try:
                parts = next(lines).strip().split()
                q, p = parts[0], parts[2]
                score = float(parts[4])
                
                # 遇到新查询时生成结果
                if q != curr_q:
                    sampled = _sample_negatives(pos_scores, negatives)
                    yield curr_q, relevance.get(curr_q, []), sampled
                    
                    # 重置状态
                    curr_q = q
                    pos_scores = [score] if p in relevance.get(q, set()) else []
                    negatives = [] if p in relevance.get(q, set()) else [(p, score)]
                else:
                    # 累积正负样本得分
                    if p in relevance.get(q, set()):
                        pos_scores.append(score)
                    else:
                        negatives.append((p, score))
                        
            except StopIteration:
                # 处理最后一个查询
                sampled = _sample_negatives(pos_scores, negatives)
                yield curr_q, relevance.get(curr_q, []), sampled
                return

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--nn_file', required=True, help='TREC run format, returned by DR_OQ model.')
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--query_variation', required=False, default='None')

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=7)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
    query_variation_file=args.query_variation
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

pbar = tqdm(load_ranking(args.nn_file, qrel, args.n_sample, args.depth))
with Pool() as p:
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.json'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()