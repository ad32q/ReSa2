import random
import math
import numpy as np
import argparse

def load_ranking(rank_file, relevance, n_sample, depth, tau=0.25):
    """Load data from ranking file and select negative samples with probability

    Args:
        rank_file (str): Path to ranking file, format [query, _, paragraph, _, score, _]
        relevance (dict): Mapping of query to set of positive paragraphs {q: {positive_paragraphs}}
        n_sample (int): Number of negative samples to sample
        depth (int): Maximum rank depth to consider for negative samples
        tau (float): Temperature coefficient for probability calculation, default 0.25

    Yields:
        tuple: (current query, list of positive samples, list of sampled negative samples)
    """
    with open(rank_file) as rf:
        lines = iter(rf)

        # Initialize first query
        try:
            parts = next(lines).strip().split()
            curr_q, curr_p = parts[0], parts[2]
            curr_score = float(parts[4])

            pos_scores = [curr_score] if int(curr_p) in relevance.get(curr_q, set()) else []
            negatives = [] if int(curr_p) in relevance.get(curr_q, set()) else [(int(curr_p), curr_score)]
        except StopIteration:
            return  # Return directly if empty file

        def _sample_negatives(pos_scores, negatives):
            """Generate negative samples based on scores"""
            # Fallback to random sampling if no positive samples
            if not pos_scores:
                candidates = [p for p, _ in negatives[:depth]]
                return random.sample(candidates, min(n_sample, len(candidates)))

            # Calculate probability weights
            pos_max = max(pos_scores)  # Use highest positive score as baseline
            candidates = negatives[:depth]
            if not candidates:
                return []

            # Calculate exponential weights
            scores = np.array([s for _, s in candidates])
            weights = np.exp(-(scores - pos_max) ** 2 * tau)
            total = weights.sum()

            # Handle zero-weight cases
            if total <= 1e-8:
                candidates = [p for p, _ in candidates]
                return random.sample(candidates, min(n_sample, len(candidates)))

            # Probability normalization
            probs = weights / total

            # Weighted sampling without replacement
            indices = np.arange(len(candidates))
            sampled_indices = np.random.choice(indices, size=min(n_sample, len(candidates)), replace=False, p=probs)
            sampled = [candidates[i][0] for i in sampled_indices]
            return sampled

        # Main loop processing subsequent lines
        while True:
            try:
                parts = next(lines).strip().split()
                q, p = parts[0], parts[2]
                score = float(parts[4])

                # Generate result when encountering new query
                if q != curr_q:
                    sampled = _sample_negatives(pos_scores, negatives)
                    yield curr_q, list(relevance.get(curr_q, [])), sampled

                    # Reset state
                    curr_q = q
                    pos_scores = [score] if int(p) in relevance.get(q, set()) else []
                    negatives = [] if int(p) in relevance.get(q, set()) else [(int(p), score)]
                else:
                    # Accumulate scores
                    if int(p) in relevance.get(q, set()):
                        pos_scores.append(score)
                    else:
                        negatives.append((int(p), score))

            except StopIteration:
                # Process final query
                sampled = _sample_negatives(pos_scores, negatives)
                yield curr_q, list(relevance.get(curr_q, [])), sampled
                return

def load_qrels(qrels_file):
    """
    Load qrels file
    :param qrels_file: Path to qrels file
    :return: Mapping of query ID to set of relevant document IDs
    """
    print(f"Loading qrels file: {qrels_file}")
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(int(docid))
    print(f"Qrels file {qrels_file} loaded successfully, containing {len(qrels)} queries")
    return qrels

def main():
    parser = argparse.ArgumentParser(description='Process ranking data for negative sampling')
    parser.add_argument('--rank_file', required=True, help='Path to input ranking file')
    parser.add_argument('--qrels_file', required=True, help='Path to qrels file')
    parser.add_argument('--output_file', required=True, help='Path to output file')
    args = parser.parse_args()

    relevance = load_qrels(args.qrels_file)
    n_sample = 500  # Number of negative samples to sample
    depth = 1000    # Maximum rank depth for negative sampling

    with open(args.output_file, 'w') as out_f:
        for query, positive_samples, negative_samples in load_ranking(args.rank_file, relevance, n_sample, depth):
            # Write negative samples
            for rank, neg_sample in enumerate(negative_samples, start=1):
                line = f"{query}\tQ0\t{neg_sample}\t{rank}\tnegative\tQ1\n"
                out_f.write(line)

if __name__ == "__main__":
    main()