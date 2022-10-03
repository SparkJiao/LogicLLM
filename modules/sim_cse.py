import logging
from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from simcse import SimCSE
from torch import Tensor
from tqdm import tqdm
from typing import Tuple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ModifiedSimCSE(SimCSE):
    def __init__(self, model_name_or_path: str,
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10,
                 pooler=None,
                 shard: bool = False):
        super().__init__(model_name_or_path, device, num_cells, num_cells_in_search, pooler)

        self.shard = shard

    def encode(self, sentence: Union[str, List[str]],
               device: str = None,
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = False,
               batch_size: int = 64,
               max_length: int = 128,
               save_path: str = None) -> Union[ndarray, Tensor]:
        """
        Modifications:
            1. Add `save_path` to avoid building index repeatedly.
        """

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch), dynamic_ncols=True, disable=(total_batch < 10)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if save_path is not None:
            torch.save(embeddings, save_path)
            logger.info(f"Save embeddings to {save_path}.")

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def build_index(self, sentences_or_file_path: Union[str, List[str]],
                    use_faiss: bool = None,
                    faiss_fast: bool = False,
                    device: str = None,
                    batch_size: int = 64,
                    embeddings: ndarray = None):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                logger.warning(
                    "Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % sentences_or_file_path)
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        if embeddings is not None:
            assert embeddings.shape[0] == len(sentences_or_file_path)
        else:
            logger.info("Encoding embeddings for sentences...")
            embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True,
                                     return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)))
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(40 * 1024 * 1024 * 1024)
                    n_gpu = torch.cuda.device_count()
                    if n_gpu > 1:
                        if self.shard:
                            # There seems an unfixed bug here.
                            # See: https://github.com/facebookresearch/faiss/issues/2064
                            # And the documentation is here: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
                            option = faiss.GpuMultipleClonerOptions()
                            option.shard = True
                            index = faiss.index_cpu_to_gpu_multiple(res, 0, index, options=option)
                        else:
                            index = faiss.index_cpu_to_all_gpus(index)
                    else:
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else:
                logger.info("Use CPU-version faiss")

            if faiss_fast:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def search(self, queries: Union[str, List[str]],
               query_vecs: ndarray = None,
               device: str = None,
               threshold: float = 0.6,
               top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:

        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results

            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            if query_vecs is None:
                query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)

            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])
