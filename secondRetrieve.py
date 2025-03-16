import numpy as np
import torch
import faiss
import argparse

def load_and_preprocess_vectors(file_path):
    """
    Load and preprocess vector file
    :param file_path: Path to vector file
    :return: Mapping of IDs to vectors
    """
    try:
        loaded_data = torch.load(file_path)
        vectors = loaded_data[0].numpy()
        ids = loaded_data[1]
        id_to_vector = {id_: vector for id_, vector in zip(ids, vectors)}
        return id_to_vector
    except FileNotFoundError:
        print(f"File {file_path} not found, please check path.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_ranking(rank_file):
    """
    Read ranking file line by line, yielding query-candidate pairs
    (Only keeps first 500 candidates per query)
    """
    with open(rank_file) as rf:
        current_qid = None
        count = 0

        for line in rf:
            q_0, _, p_0, _, _, _ = line.strip().split()

            if current_qid is None or current_qid != q_0:
                current_qid = q_0
                count = 0

            if count < 500:
                yield q_0, p_0
                count += 1
            else:
                continue

def load_corpus(corpus_file):
    """
    Load vectors from corpus file
    :param corpus_file: Path to corpus file
    :return: Numpy array of vectors
    """
    print(f"Loading corpus file: {corpus_file}")
    try:
        loaded_data = torch.load(corpus_file)
        vectors = loaded_data[0].numpy()
        print(f"Corpus file {corpus_file} loaded successfully, vector array length: {len(vectors)}")
        return vectors
    except FileNotFoundError:
        print(f"File {corpus_file} not found, please check path.")
        return None
    except Exception as e:
        print(f"Error loading {corpus_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='FAISS-based document reranking')
    parser.add_argument('--query_vectors', required=True, help='Path to query vectors file')
    parser.add_argument('--rank_file', required=True, help='Path to input ranking file')
    parser.add_argument('--corpus_file', required=True, help='Path to corpus vectors file')
    parser.add_argument('--output_file', required=True, help='Path to output reranked file')
    args = parser.parse_args()

    # Load vectors
    id_to_vector_1 = load_and_preprocess_vectors(args.query_vectors)
    id_to_vector_w = load_corpus(args.corpus_file)

    if id_to_vector_w is None:
        return

    # Create FAISS index
    d = id_to_vector_w.shape[1]  # Vector dimension
    index = faiss.IndexFlatIP(d)  # Inner product similarity
    index.add(id_to_vector_w)

    # Move index to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    # Reranking storage
    re_ranked_dict = {}
    current_q_id = None
    candidate_indices = []

    with open(args.output_file, 'w') as out_f:
        for q_id, p_id in load_ranking(args.rank_file):
            if current_q_id is None:
                current_q_id = q_id
            elif q_id != current_q_id:
                # Process current query's candidates
                if candidate_indices and current_q_id in id_to_vector_1:
                    vector_q = id_to_vector_1[current_q_id].reshape(1, -1).astype('float32')
                    
                    # Batch processing
                    batch_size = 100
                    sorted_indices = []
                    for i in range(0, len(candidate_indices), batch_size):
                        batch_indices = candidate_indices[i:i+batch_size]
                        candidate_vectors = id_to_vector_w[batch_indices]

                        # Create temporary index
                        temp_index = faiss.IndexFlatIP(d)
                        temp_index.add(candidate_vectors)
                        temp_gpu_index = faiss.index_cpu_to_gpu(res, 0, temp_index)

                        # Search
                        D, I = temp_gpu_index.search(vector_q, len(candidate_vectors))
                        batch_sorted_indices = [batch_indices[i] for i in I[0]]
                        sorted_indices.extend(batch_sorted_indices)

                        # Cleanup
                        del temp_gpu_index
                        del temp_index
                        torch.cuda.empty_cache()

                    # Write results
                    for rank, p_id in enumerate(sorted_indices, start=1):
                        line = f"{current_q_id}\tQ0\t{p_id}\t{rank}\tmy_rerank\tQ1\n"
                        out_f.write(line)

                # Reset for new query
                candidate_indices = []
                current_q_id = q_id

            try:
                p_id = int(p_id)
                if p_id < len(id_to_vector_w):
                    candidate_indices.append(p_id)
                else:
                    print(f"ID {p_id} exceeds corpus array bounds, skipping.")
            except ValueError:
                print(f"Invalid ID format: {p_id}, skipping.")

        # Process final query
        if candidate_indices and current_q_id in id_to_vector_1:
            vector_q = id_to_vector_1[current_q_id].reshape(1, -1).astype('float32')
            batch_size = 100
            sorted_indices = []
            for i in range(0, len(candidate_indices), batch_size):
                batch_indices = candidate_indices[i:i+batch_size]
                candidate_vectors = id_to_vector_w[batch_indices]

                temp_index = faiss.IndexFlatIP(d)
                temp_index.add(candidate_vectors)
                temp_gpu_index = faiss.index_cpu_to_gpu(res, 0, temp_index)

                D, I = temp_gpu_index.search(vector_q, len(candidate_vectors))
                batch_sorted_indices = [batch_indices[i] for i in I[0]]
                sorted_indices.extend(batch_sorted_indices)

                del temp_gpu_index
                del temp_index
                torch.cuda.empty_cache()

            for rank, p_id in enumerate(sorted_indices, start=1):
                line = f"{current_q_id}\tQ0\t{p_id}\t{rank}\tmy_rerank\tQ1\n"
                out_f.write(line)

if __name__ == "__main__":
    main()