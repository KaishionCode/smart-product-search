from dataclasses import dataclass
from typing import List, Optional, Dict
import re
import math

@dataclass
class Query:
    raw: str
    normalized: str
    words: List[str]
    @classmethod
    def from_raw(cls, text: str):
        if not text:
            return cls("", "", [])
        norm = re.sub(r"\s+", " ", text.strip().lower())
        return cls(text, norm, norm.split())
    @property
    def word_count(self): 
        return len(self.words)
    @property
    def is_short(self): 
        return self.word_count < 12
    @property
    def threshold(self):
        if self.word_count == 1: return 0.2
        if self.word_count == 2: return 0.25
        return 0.3
    @property
    def fusion_weights(self):
        return (0.5, 0.5) if self.word_count >= 5 else (0.3, 0.7)
    @property
    def candidate_limit(self): 
        return 100 if self.is_short else 300
    @property
    def probes(self):
        return min(25, max(5, self.word_count * 2)) if self.is_short else min(20, max(5, self.word_count))
@dataclass
class SearchResult:
    stt: int
    ten_hang: str
    thong_so: str
    gia_tham_dinh: Optional[float]
    final_score: float
    
    def to_dict(self): 
        return self.__dict__
@dataclass
class RawData:
    stt: int
    ten_hang: str
    thong_so: str
    gia_tham_dinh: Optional[float]
    vector_score: float = 0.0
    bm25_score: float = 0.0
class ScoreCalc:
    """Các hàm tính toán điểm số - đặt ở đây vì liên quan đến models"""
    @staticmethod
    def softmax(scores):
        if not scores: return []
        max_s = max(scores)
        exp = [math.exp(s - max_s) for s in scores]
        total = sum(exp)
        return [e/total if total else 0 for e in exp] 
    @staticmethod
    def match_boost(cnt): 
        return 1 + min(cnt, 3) * 0.3
    @staticmethod
    def idf_boost(cnt): 
        return math.log(1 + cnt) * 0.2
    @staticmethod
    def phrase_boost(q, ten): 
        return 0.5 if q.lower() in ten.lower() else 0.0
    @staticmethod
    def penalty(ten): 
        return min(len(ten) * 0.0005, 0.1)
