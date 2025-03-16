import sys
import math
from scipy import stats


def calculate_metrics(qrels_file_path, output_results_path):
    # 读取qrel文件（支持多级相关性）
    qrel = {}
    with open(qrels_file_path, 'r', encoding='utf8') as f:
        for line in f:
            topicid, _, docid, rel = line.strip().split()
            rel_score = int(rel)
            if rel_score < 0:  # 通常负值表示不相关
                continue
            if topicid not in qrel:
                qrel[topicid] = {}
            qrel[topicid][docid] = rel_score

    # 读取结果文件并排序
    rankings = {}
    with open(output_results_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            try:
                qid = parts[0]
                docid = parts[2]
                rank = int(parts[3])
            except IndexError:
                continue
            
            if qid not in rankings:
                rankings[qid] = []
            rankings[qid].append((docid, rank))
    
    # 按rank升序排序（rank=1是第一名）
    for qid in rankings:
        rankings[qid].sort(key=lambda x: x[1])

    # 初始化指标
    query_num = 0
    skipped_queries = 0
    mrr_ten_score = 0.0
    recall_all_score = 0.0
    recall_50_score = 0.0
    recall_100_score = 0.0
    ndcg_10_score = 0.0
    
    qid_metrics = {
        'mrr': {},
        'recall': {},
        'ndcg': {}
    }

    # 处理每个查询
    for query in rankings:
        # 跳过不在qrel中的查询
        if query not in qrel:
            skipped_queries += 1
            continue
            
        relevant_info = qrel[query]
        total_relevant = len(relevant_info)
        
        # 跳过没有相关文档的查询
        if total_relevant == 0:
            skipped_queries += 1
            continue

        query_num += 1
        
        # MRR@10计算
        mrr_10 = 0.0
        for pos, (doc, _) in enumerate(rankings[query][:10], 1):
            if doc in relevant_info:
                mrr_10 = 1.0 / pos
                break
        mrr_ten_score += mrr_10
        qid_metrics['mrr'][query] = mrr_10

        # Recall计算
        hit_all = 0
        hit_50 = 0
        hit_100 = 0
        
        for doc, rank in rankings[query]:
            if doc in relevant_info:
                hit_all += 1
                if rank <= 50:
                    hit_50 += 1
                if rank <= 100:
                    hit_100 += 1
        
        recall_all = hit_all / total_relevant
        recall_50 = hit_50 / total_relevant
        recall_100 = hit_100 / total_relevant
        
        recall_all_score += recall_all
        recall_50_score += recall_50
        recall_100_score += recall_100
        
        qid_metrics['recall'][query] = {
            'all': recall_all,
            '50': recall_50,
            '100': recall_100
        }

        # nDCG@10计算
        dcg = 0.0
        for pos, (doc, _) in enumerate(rankings[query][:10], 1):
            rel = relevant_info.get(doc, 0)
            dcg += rel / math.log2(pos + 1)  # 折扣因子从第2位开始
        
        # 计算理想DCG
        ideal_scores = sorted(relevant_info.values(), reverse=True)[:10]
        idcg = sum(rel / math.log2(i+1) for i, rel in enumerate(ideal_scores, 1))
        
        ndcg_10 = dcg / idcg if idcg > 0 else 0.0
        ndcg_10_score += ndcg_10
        qid_metrics['ndcg'][query] = ndcg_10

    # 计算平均值
    avg_mrr = mrr_ten_score / query_num if query_num > 0 else 0
    avg_recall_all = recall_all_score / query_num if query_num > 0 else 0
    avg_recall_50 = recall_50_score / query_num if query_num > 0 else 0
    avg_recall_100 = recall_100_score / query_num if query_num > 0 else 0
    avg_ndcg_10 = ndcg_10_score / query_num if query_num > 0 else 0

    return {
        'mrr': avg_mrr,
        'recall_all': avg_recall_all,
        'recall_50': avg_recall_50,
        'recall_100': avg_recall_100,
        'ndcg_10': avg_ndcg_10,
        'query_num': query_num,
        'skipped': skipped_queries
    }


def evaluate_single_file(qrels_path, result_path):
    metrics = calculate_metrics(qrels_path, result_path)
    
    print("\nEvaluation Results:")
    print(f"- MRR@10: {metrics['mrr']:.4f}")
    print(f"- Recall@All: {metrics['recall_all']:.4f}")
    print(f"- Recall@50: {metrics['recall_50']:.4f}") 
    print(f"- Recall@100: {metrics['recall_100']:.4f}")
    print(f"- nDCG@10: {metrics['ndcg_10']:.4f}")
    print(f"- Evaluated Queries: {metrics['query_num']}")
    print(f"- Skipped Queries: {metrics['skipped']}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        evaluate_single_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python eval.py <qrels.txt> <results.run>")
        sys.exit(1)