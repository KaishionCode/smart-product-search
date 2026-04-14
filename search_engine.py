# search_engine.py
import re
import math
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from data import DatabasePool, Cache, EmbeddingModel
from models import Query, SearchResult, RawData, ScoreCalc

logger = logging.getLogger(__name__)
# ========== REPOSITORY ==========
class SearchRepo:
    @staticmethod
    def short_query(vec, thresh, limit, probes):
        with DatabasePool.connection() as conn:
            with DatabasePool.cursor(conn) as cur:
                cur.execute("BEGIN; SET LOCAL ivfflat.probes = %s;", (probes,))
                cur.execute("""
                    SELECT *, 1 - (vt_ten_hang <=> %s::vector) AS sim
                    FROM thong_tin_hang_hoa
                    WHERE vt_ten_hang IS NOT NULL 
                      AND 1 - (vt_ten_hang <=> %s::vector) > %s
                    ORDER BY vt_ten_hang <=> %s::vector
                    LIMIT %s
                """, (vec, vec, thresh, vec, limit))
                rows = cur.fetchall()
                conn.commit()
                return [RawData(r['stt'], r['ten_hang'], r.get('thong_so',''), 
                               r.get('gia_tham_dinh'), r['sim']) for r in rows]
    
    @staticmethod
    def vector_query(vec, thresh, limit, probes):
        with DatabasePool.connection() as conn:
            with DatabasePool.cursor(conn) as cur:
                cur.execute("BEGIN; SET LOCAL ivfflat.probes = %s;", (probes,))
                cur.execute("""
                    WITH c AS (
                        SELECT stt, ten_hang, thong_so, gia_tham_dinh,
                               (vector <=> %s::vector) as dist
                        FROM thong_tin_hang_hoa 
                        WHERE vector IS NOT NULL 
                        ORDER BY dist LIMIT %s
                    )
                    SELECT *, 1 - dist as vector_score 
                    FROM c 
                    WHERE 1 - dist > %s 
                    LIMIT 50
                """, (vec, limit, thresh))
                rows = cur.fetchall()
                conn.commit()
                return [RawData(r['stt'], r['ten_hang'], r.get('thong_so',''),
                               r.get('gia_tham_dinh'), r['vector_score']) for r in rows]
    
    @staticmethod
    def bm25_query(text):
        with DatabasePool.connection() as conn:
            with DatabasePool.cursor(conn) as cur:
                cur.execute("BEGIN;")
                cur.execute("""
                    SELECT stt, ten_hang, thong_so, gia_tham_dinh,
                           ts_rank(search_vector, plainto_tsquery('english', %s)) as bm25_score
                    FROM thong_tin_hang_hoa
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY bm25_score DESC
                    LIMIT 50
                """, (text, text))
                rows = cur.fetchall()
                conn.commit()
                return [RawData(r['stt'], r['ten_hang'], r.get('thong_so',''),
                               r.get('gia_tham_dinh'), bm25_score=r['bm25_score']) for r in rows]

# ========== STRATEGIES ==========
class SearchStrategy(ABC):
    @abstractmethod
    def search(self, q: Query) -> List[SearchResult]: pass

class ShortStrategy(SearchStrategy):
    def search(self, q: Query) -> List[SearchResult]:
        vec = EmbeddingModel.encode(q.normalized)
        data = SearchRepo.short_query(vec, q.threshold, q.candidate_limit, q.probes)
        ranked = []
        for r in data:
            score = r.vector_score * 0.4 + self._boost(q.words, r.ten_hang)
            score -= ScoreCalc.penalty(r.ten_hang)
            ranked.append(SearchResult(r.stt, r.ten_hang, r.thong_so, r.gia_tham_dinh, score))
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)[:15]
    def _boost(self, words, ten):
        if not words: return 0
        boost, ten_lower = 0, ten.lower()
        weights = list(range(len(words), 0, -1))
        sum_w = sum(weights)
        for i, w in enumerate(words):
            weight = (weights[i]/sum_w) * 0.6
            if re.search(rf"\b{re.escape(w)}\b", ten_lower): 
                boost += weight * 1.5
            elif ten_lower.startswith(w): 
                boost += weight * 1.2
            elif w in ten_lower: 
                boost += weight
        return boost
class LongStrategy(SearchStrategy):
    def search(self, q: Query) -> List[SearchResult]:
        vec = EmbeddingModel.encode(q.normalized)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fv = ex.submit(SearchRepo.vector_query, vec, q.threshold, q.candidate_limit, q.probes)
            fb = ex.submit(SearchRepo.bm25_query, q.normalized)
            v_data, b_data = fv.result(), fb.result()
        if v_data and max(r.vector_score for r in v_data) > 0.85:
            b_data = []
        merged = {r.stt: r for r in v_data}
        for r in b_data:
            if r.stt in merged:
                merged[r.stt].bm25_score = max(merged[r.stt].bm25_score, r.bm25_score)
            else:
                merged[r.stt] = r
        items = sorted(merged.values(), key=lambda x: x.vector_score + x.bm25_score, reverse=True)[:50]
        v_norm = ScoreCalc.softmax([i.vector_score for i in items])
        b_norm = ScoreCalc.softmax([i.bm25_score for i in items])
        vw, bw = q.fusion_weights
        ranked = []
        for i, item in enumerate(items):
            fusion = v_norm[i] * vw + b_norm[i] * bw if v_norm else 0
            match_cnt = sum(1 for w in q.words if w in item.ten_hang.lower())
            score = fusion * ScoreCalc.match_boost(match_cnt)
            score += ScoreCalc.phrase_boost(q.normalized, item.ten_hang)
            score += ScoreCalc.idf_boost(match_cnt)
            score -= ScoreCalc.penalty(item.ten_hang)
            ranked.append(SearchResult(item.stt, item.ten_hang, item.thong_so, item.gia_tham_dinh, score))
        return sorted(ranked, key=lambda x: x.final_score, reverse=True)[:15]
    
class StrategyFactory:
    _map = {True: ShortStrategy(), False: LongStrategy()}  
    @classmethod
    def get(cls, q): 
        return cls._map[q.is_short]
# ========== SEARCH ENGINE ==========
class SearchEngine:
    def search(self, raw: str) -> List[SearchResult]:
        if not raw: return []
        q = Query.from_raw(raw)
        key = f"search:{q.normalized}"
        cached = Cache.get(key)
        if cached:
            logger.info(f"✅ Cache: '{raw}'")
            return cached
        start = time.time()
        results = StrategyFactory.get(q).search(q)
        logger.info(f"🔍 '{raw}' | {time.time()-start:.3f}s | {len(results)} results")
        Cache.set(key, results)
        return results
# ========== AUTOCOMPLETE ==========
class AutocompleteService:
    @staticmethod
    def suggest(kw: str) -> List[Dict]:
        if not kw or len(kw) < 2:
            return []
        
        try:
            with DatabasePool.connection() as conn:
                with DatabasePool.cursor(conn) as cur:
                    kw_lower = kw.lower()
                    sql = """
                    SELECT DISTINCT ten_hang
                    FROM thong_tin_hang_hoa
                    WHERE LOWER(ten_hang) LIKE %s
                    LIMIT 20
                    """
                    cur.execute(sql, (f"%{kw_lower}%",))
                    rows = cur.fetchall()
                    conn.commit() 
                    if not rows:
                        return []
                    suggestions = []
                    for r in rows:
                        ten = r['ten_hang']
                        ten_lower = ten.lower()
                        pos = ten_lower.find(kw_lower)
                        if ten_lower.startswith(kw_lower):
                            priority = 0
                        else:
                            priority = pos if pos >= 0 else 999
                        suggestions.append({
                            'ten_hang': ten,
                            'priority': priority
                        })
                    suggestions.sort(key=lambda x: x['priority'])
                    return [{'ten_hang': s['ten_hang']} for s in suggestions[:10]]     
        except Exception as e:
            logger.error(f"❌ Autocomplete error: {e}")
            return []
