from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize


class SemanticScoreChecker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def compute_score(self, chunk, answer, verbose=True):
    
        claims = self.split_sentences(answer)
        chunk_sentences = self.split_sentences(chunk)

        if len(claims) == 0 or len(chunk_sentences) == 0:
            return 0.0, []

        # Encode chunk sentences once
        chunk_embeddings = self.model.encode(chunk_sentences, convert_to_tensor=True, show_progress_bar=False)

        results = []
        supported_count = 0

        for claim in claims:
            claim_embedding = self.model.encode(claim, convert_to_tensor=True, show_progress_bar=False)

            # Compare with all chunk sentences
            similarities = util.cos_sim(claim_embedding, chunk_embeddings)

            max_score = similarities.max().item()
            best_match_idx = similarities.argmax().item()
            best_match_sentence = chunk_sentences[best_match_idx]

            is_supported = max_score >= self.threshold

            if is_supported:
                supported_count += 1

            results.append({
                "claim": claim,
                "max_similarity": round(max_score, 4),
                "supported": is_supported,
                "best_match": best_match_sentence
            })

        
        semantic_score = supported_count / len(claims)

        return {
            "results": results,
            "semantic_score": semantic_score
        }

