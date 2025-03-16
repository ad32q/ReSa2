
import numpy as np
import faiss
import re
import logging
import time
import threading

# 配置日志记录，设置日志级别为DEBUG以获取更详细的信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def index_cpu_to_gpu_with_timeout(res, device, index, timeout=10):
    result = []
    def target():
        try:
            gpu_index = faiss.index_cpu_to_gpu(res, device, index)
            result.append(gpu_index)
        except Exception as e:
            import traceback
            logger.error(f"Error migrating index to GPU: {e}\n{traceback.format_exc()}")

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.error("Index migration to GPU timed out.")
        return None
    if result:
        return result[0]
    return None

def index_from_factory(init_reps: np.ndarray, index_param: str):
    start_time = time.time()
    logger.debug(f"Entering index_from_factory with index_param: {index_param}")
    if '-' in index_param:
        index_param = re.sub('-', ',', index_param)
        logger.debug(f"Modified index_param to: {index_param}")
    dim, measure = init_reps.shape[1], faiss.METRIC_INNER_PRODUCT
    logger.debug(f"Dimension: {dim}, Measure: {measure}")
    index = faiss.index_factory(dim, index_param, measure)
    logger.debug("CPU index created.")

    try:
        res = faiss.StandardGpuResources()
        logger.debug("GPU resources created.")
        gpu_index = index_cpu_to_gpu_with_timeout(res, 0, index)
        if gpu_index is None:
            raise Exception("Failed to migrate index to GPU.")
        logger.debug("Index migrated to GPU.")
    except Exception as e:
        logger.error(f"Error migrating index to GPU: {e}")
        raise

    if not gpu_index.is_trained:
        logger.debug("Starting training...")
        try:
            gpu_index.train(init_reps)
            logger.debug("Training completed.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    logger.info(f"Index trained: {gpu_index.is_trained}")
    logger.debug("Starting to add vectors to the index...")
    try:
        gpu_index.add(init_reps)
        logger.debug("Vectors added to the index.")
    except Exception as e:
        logger.error(f"Error adding vectors to the index: {e}")
        raise
    end_time = time.time()
    logger.debug(f"index_from_factory took {end_time - start_time} seconds.")
    return gpu_index

class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray, index_type: str, index_file: str, construct: bool):
        start_time = time.time()
        logger.debug(f"Entering __init__ with construct: {construct}")
        if construct:
            logger.debug("Constructing a new index.")
            try:
                self.index = index_from_factory(init_reps, index_type)
                logger.debug("New index constructed.")
            except Exception as e:
                logger.error(f"Error constructing index: {e}")
                raise
        else:
            logger.debug("Reading index from file.")
            try:
                self.index = faiss.read_index(index_file)
                logger.debug("Index read from file.")
                res = faiss.StandardGpuResources()
                logger.debug("GPU resources created for existing index.")
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.debug("Existing index migrated to GPU.")
            except Exception as e:
                logger.error(f"Error reading or migrating existing index: {e}")
                raise
        end_time = time.time()
        logger.debug(f"__init__ took {end_time - start_time} seconds.")

    def search(self, q_reps: np.ndarray, k: int):
        start_time = time.time()
        logger.debug(f"Searching with {q_reps.shape[0]} queries, k={k}")
        try:
            result = self.index.search(q_reps, k)
            logger.debug("Search completed.")
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
        end_time = time.time()
        logger.debug(f"search took {end_time - start_time} seconds.")
        return result

    def add(self, p_reps: np.ndarray):
        start_time = time.time()
        logger.debug(f"Adding {p_reps.shape[0]} vectors to the index.")
        try:
            self.index.add(p_reps)
            logger.debug("Vectors added.")
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise
        end_time = time.time()
        logger.debug(f"add took {end_time - start_time} seconds.")

    def save_index(self, save_dir: str):
        start_time = time.time()
        logger.debug(f"Saving index to {save_dir}")
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            logger.debug("Index migrated back to CPU.")
            faiss.write_index(cpu_index, save_dir)
            logger.debug("Index saved.")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
        end_time = time.time()
        logger.debug(f"save_index took {end_time - start_time} seconds.")

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        start_time = time.time()
        num_query = q_reps.shape[0]
        logger.debug(f"Starting batch search with {num_query} queries, k={k}, batch_size={batch_size}")
        all_scores = []
        all_indices = []
        for start_idx in range(0, num_query, batch_size):
            logger.info(f"Searching idx: {start_idx}")
            try:
                nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
                all_scores.append(nn_scores)
                all_indices.append(nn_indices)
            except Exception as e:
                logger.error(f"Error in batch search at index {start_idx}: {e}")
                raise
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        logger.debug("Batch search completed.")
        end_time = time.time()
        logger.debug(f"batch_search took {end_time - start_time} seconds.")
        return all_scores, all_indices