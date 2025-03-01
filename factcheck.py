# factcheck.py

import torch
import torch.nn as nn
from typing import List, Dict, Set
import numpy as np
import spacy
import gc
from collections import Counter
import math


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = nn.functional.softmax(logits, dim=1).squeeze()

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        entailment_prob = probs[0].item()
        del inputs, outputs, logits
        gc.collect()

        # return something
        return entailment_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, jaccard_threshold=0.1, cosine_threshold=0.15, jaccard_weight=0.3, stopwords=None):
        self.jaccard_threshold = jaccard_threshold
        self.cosine_threshold = cosine_threshold
        self.jaccard_weight = jaccard_weight
        self.stopwords = stopwords or set(["the", "is", "was", "in", "a", "an", "to", "of", "for"])
        self.nlp = spacy.load('en_core_web_sm')
    
    def preprocess(self, text: str) -> Set[str]:
        doc = self.nlp(text.lower())
        return {token.lemma_ for token in doc if token.lemma_ not in self.stopwords and token.is_alpha}

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0
    
    def split_into_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text) 
        return [sent.text for sent in doc.sents]
    
    def calculate_tf_idf(self, documents: List[List[str]]) -> List[Dict[str, float]]:
        doc_count = len(documents)
        df = Counter(word for doc in documents for word in set(doc))
        
        tf_idf_vectors = []
        for doc in documents:
            tf = Counter(doc)
            tf_idf = {}
            for word, count in tf.items():
                term_freq = count / len(doc) #TF
                inv_doc_freq = math.log(doc_count / (1 + df[word]))  # IDF
                tf_idf[word] = term_freq * inv_doc_freq  # TF-IDF
            tf_idf_vectors.append(tf_idf)
        
        return tf_idf_vectors

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[word] * vec2[word] for word in intersection)
        
        magnitude1 = math.sqrt(sum(value**2 for value in vec1.values()))
        magnitude2 = math.sqrt(sum(value**2 for value in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def predict(self, fact: str, passages: List[Dict[str, str]]) -> str:
        fact_words = self.preprocess(fact)
        fact_set = set(fact_words)
        max_combined_similarity = 0.0
        
        sentences = []
        for passage in passages:
            sentences.extend(self.split_into_sentences(passage['text']))

        processed_sentences = [self.preprocess(sentence) for sentence in sentences]
        
        tf_idf_vectors = self.calculate_tf_idf([fact_words] + processed_sentences)
        fact_tf_idf = tf_idf_vectors[0]
        sentence_tf_idfs = tf_idf_vectors[1:]

        for sentence, sentence_tf_idf in zip(sentences, sentence_tf_idfs):
            sentence_words = set(self.preprocess(sentence))
            jaccard_sim = self.jaccard_similarity(fact_set, sentence_words)
            cosine_sim = self.cosine_similarity(fact_tf_idf, sentence_tf_idf)
            combined_similarity = (jaccard_sim * self.jaccard_weight) + (cosine_sim * (1 - self.jaccard_weight))
            max_combined_similarity = max(max_combined_similarity, combined_similarity)

        return "S" if max_combined_similarity >= max(self.jaccard_threshold, self.cosine_threshold) else "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        # raise Exception("Implement me")
        for passage in passages:
            sentences = passage["text"].split(". ")
            
            for sentence in sentences:
                entailment_score = self.ent_model.check_entailment(sentence, fact)
                if entailment_score > 0.5:
                    return "S"
        
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self, threshold=0.1):
        self.nlp = spacy.load('en_core_web_sm')
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        # raise Exception("Implement me")
        fact_dependencies = self.get_dependencies(fact)
        max_similarity = 0.0
        for passage in passages:
            sentences = passage["text"].split(". ")
            for sentence in sentences:
                passage_dependencies = self.get_dependencies(sentence)
                
                similarity = self.jaccard_similarity(fact_dependencies, passage_dependencies)
                
                max_similarity = max(max_similarity, similarity)

        return "S" if max_similarity >= self.threshold else "NS"

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """

        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / union