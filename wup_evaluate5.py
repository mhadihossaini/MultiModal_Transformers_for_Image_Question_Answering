from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import wordnet
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import wordnet

class wupScoreCalculator:
    def __init__(self, answer_space: List[str]):
        self.answer_space = answer_space

    def wup_measure(self, a, b, similarity_threshold=0.925):
        def get_semantic_field(word):
            weight = 1.0
            semantic_field = wordnet.synsets(word, pos=wordnet.NOUN)
            return semantic_field, weight

        def get_stem_word(word):
            weight = 1.0
            return word, weight

        global_weight = 1.0
        a, global_weight_a = get_stem_word(a)
        b, global_weight_b = get_stem_word(b)
        global_weight = min(global_weight_a, global_weight_b)

        if a == b:
            return 1.0 * global_weight

        if not a or not b:
            return 0

        interp_a, weight_a = get_semantic_field(a)
        interp_b, weight_b = get_semantic_field(b)

        if not interp_a or not interp_b:
            return 0

        global_max = 0.0
        for x in interp_a:
            for y in interp_b:
                local_score = x.wup_similarity(y)
                if local_score > global_max:
                    global_max = local_score

        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score = global_max * weight_a * weight_b * interp_weight * global_weight
        return final_score

    def batch_wup_measure(self, labels: np.ndarray, preds: np.ndarray, answer_space) -> float:
        wup_scores = [self.wup_measure(self.answer_space[label], self.answer_space[pred]) for label, pred in zip(labels, preds)]
        return np.mean(wup_scores)

