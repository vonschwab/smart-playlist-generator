"""
Genre Similarity Calculator V2 - Data Science Approach
Uses multiple similarity metrics for robust genre comparison
"""
import yaml
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class GenreSimilarityV2:
    """
    Advanced genre similarity calculator using multiple metrics:
    1. Jaccard similarity (set overlap)
    2. Weighted Jaccard (with genre relationship weights)
    3. Cosine similarity (genre vector representation)
    4. Average pairwise similarity (mean of all genre-to-genre comparisons)
    """

    def __init__(self, similarity_file: str = "data/genre_similarity.yaml"):
        """
        Initialize genre similarity calculator

        Args:
            similarity_file: Path to genre similarity YAML file
        """
        self.similarity_matrix = {}
        self._load_similarity_matrix(similarity_file)

        # Build vocabulary for vectorization
        self._build_genre_vocabulary()

        logger.info(f"Loaded genre similarity matrix with {len(self.similarity_matrix)} genres")
        logger.info(f"Genre vocabulary size: {len(self.genre_vocab)}")

    def _load_similarity_matrix(self, filepath: str):
        """Load genre similarity matrix from YAML file"""
        try:
            with open(filepath, 'r') as f:
                self.similarity_matrix = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Genre similarity file not found: {filepath}")
            self.similarity_matrix = {}
        except Exception as e:
            logger.error(f"Error loading genre similarity matrix: {e}")
            self.similarity_matrix = {}

    def _build_genre_vocabulary(self):
        """Build vocabulary of all genres for vectorization"""
        genres = set(self.similarity_matrix.keys())

        # Add all genres that appear in similarity relationships
        for genre, similar in self.similarity_matrix.items():
            genres.update(similar.keys())

        # Create index mapping
        self.genre_vocab = {genre: idx for idx, genre in enumerate(sorted(genres))}
        self.vocab_size = len(self.genre_vocab)

    def _lookup_similarity(self, genre1: str, genre2: str) -> float:
        """
        Look up similarity between two genres

        Checks both directions: genre1->genre2 and genre2->genre1
        """
        if genre1 == genre2:
            return 1.0

        # Check genre1 -> genre2
        if genre1 in self.similarity_matrix:
            if genre2 in self.similarity_matrix[genre1]:
                return self.similarity_matrix[genre1][genre2]

        # Check genre2 -> genre1 (reverse direction)
        if genre2 in self.similarity_matrix:
            if genre1 in self.similarity_matrix[genre2]:
                return self.similarity_matrix[genre2][genre1]

        return 0.0

    def jaccard_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate Jaccard similarity (set intersection / set union)

        Measures pure overlap between genre sets, ignoring relationships.
        Good for finding exact genre matches.

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Jaccard similarity score (0.0-1.0)
        """
        if not genres1 or not genres2:
            return 0.0

        set1 = set(g.lower().strip() for g in genres1)
        set2 = set(g.lower().strip() for g in genres2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def weighted_jaccard_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate weighted Jaccard similarity using genre relationship matrix

        Instead of binary membership, uses similarity scores from the matrix.
        Better than pure Jaccard because it accounts for related genres.

        Formula:
            Sum(min(weight(g1), weight(g2))) / Sum(max(weight(g1), weight(g2)))

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Weighted Jaccard similarity score (0.0-1.0)
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        # Get all unique genres
        all_genres = set(genres1) | set(genres2)

        min_sum = 0.0
        max_sum = 0.0

        for genre in all_genres:
            # Calculate maximum similarity of this genre to each set
            weight1 = self._genre_set_membership(genre, genres1)
            weight2 = self._genre_set_membership(genre, genres2)

            min_sum += min(weight1, weight2)
            max_sum += max(weight1, weight2)

        if max_sum == 0:
            return 0.0

        return min_sum / max_sum

    def _genre_set_membership(self, genre: str, genre_set: List[str]) -> float:
        """
        Calculate how much a genre "belongs" to a set of genres

        Returns 1.0 if exact match, or maximum similarity to any genre in set
        """
        if genre in genre_set:
            return 1.0

        max_sim = 0.0
        for g in genre_set:
            sim = self._lookup_similarity(genre, g)
            max_sim = max(max_sim, sim)

        return max_sim

    def cosine_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate cosine similarity using genre vectors

        Represents each genre list as a vector in high-dimensional space,
        then calculates the angle between them.

        Good for comparing overall genre profiles, less sensitive to set size.

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        # Create genre vectors using relationship-aware weights
        vec1 = self._create_genre_vector(genres1)
        vec2 = self._create_genre_vector(genres2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Convert from [-1, 1] to [0, 1]
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)

    def _create_genre_vector(self, genres: List[str]) -> np.ndarray:
        """
        Create a vector representation of a genre list

        Uses the similarity matrix to spread genre membership across
        related genres, creating a richer representation.
        """
        vector = np.zeros(self.vocab_size)

        for genre in genres:
            if genre in self.genre_vocab:
                idx = self.genre_vocab[genre]
                vector[idx] = 1.0

                # Spread weight to similar genres (weighted by similarity)
                if genre in self.similarity_matrix:
                    for similar_genre, weight in self.similarity_matrix[genre].items():
                        if similar_genre in self.genre_vocab:
                            similar_idx = self.genre_vocab[similar_genre]
                            # Use weighted contribution
                            vector[similar_idx] = max(vector[similar_idx], weight * 0.5)

        return vector

    def average_pairwise_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate average similarity across all genre pairs

        Computes similarity between every genre in list1 and every genre in list2,
        then averages. More comprehensive than maximum similarity.

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Average pairwise similarity score (0.0-1.0)
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        total_similarity = 0.0
        count = 0

        for g1 in genres1:
            for g2 in genres2:
                total_similarity += self._lookup_similarity(g1, g2)
                count += 1

        if count == 0:
            return 0.0

        return total_similarity / count

    def best_match_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate similarity using best matching pairs (Hungarian algorithm concept)

        For each genre in the smaller list, finds its best match in the larger list,
        then averages these best matches. Balanced approach.

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Best match similarity score (0.0-1.0)
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        # Use the smaller list as the reference
        if len(genres1) <= len(genres2):
            ref_genres = genres1
            target_genres = genres2
        else:
            ref_genres = genres2
            target_genres = genres1

        total_similarity = 0.0

        # For each genre in reference, find best match in target
        for ref_genre in ref_genres:
            best_sim = 0.0
            for target_genre in target_genres:
                sim = self._lookup_similarity(ref_genre, target_genre)
                best_sim = max(best_sim, sim)
            total_similarity += best_sim

        return total_similarity / len(ref_genres)

    def best_match_size_normalized(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Best-match similarity normalized by list sizes.

        - Uses best matches from the smaller list to the larger list (like best_match_similarity).
        - Scales by (min_len / max_len) so extra tags in a much larger list don't inflate the score.
        - Exact match on two tags vs two tags => 1.0; two tags vs ten tags with only those two matching => 0.2.
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        len1, len2 = len(genres1), len(genres2)
        min_len = min(len1, len2)
        max_len = max(len1, len2)

        if min_len == 0 or max_len == 0:
            return 0.0

        # Use the smaller list as reference
        if len1 <= len2:
            ref_genres = genres1
            target_genres = genres2
        else:
            ref_genres = genres2
            target_genres = genres1

        total_similarity = 0.0
        for ref_genre in ref_genres:
            best_sim = 0.0
            for target_genre in target_genres:
                sim = self._lookup_similarity(ref_genre, target_genre)
                best_sim = max(best_sim, sim)
            total_similarity += best_sim

        # Base best-match average
        base_score = total_similarity / min_len
        # Size normalization factor
        size_factor = min_len / max_len
        return base_score * size_factor

    def calculate_similarity(self, genres1: List[str], genres2: List[str],
                           method: str = "ensemble") -> float:
        """
        Calculate similarity between two genre lists using specified method

        Args:
            genres1: First list of genres
            genres2: Second list of genres
            method: Similarity method to use:
                - "jaccard": Pure set overlap
                - "weighted_jaccard": Relationship-aware set overlap
                - "cosine": Vector-based similarity
                - "average_pairwise": Mean of all genre pairs
                - "best_match": Best matching pairs
                - "ensemble": Weighted combination of all methods (recommended)
                - "legacy": Original max-similarity method (for comparison)

        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not genres1 or not genres2:
            return 0.0

        if method == "jaccard":
            return self.jaccard_similarity(genres1, genres2)

        elif method == "weighted_jaccard":
            return self.weighted_jaccard_similarity(genres1, genres2)

        elif method == "cosine":
            return self.cosine_similarity(genres1, genres2)

        elif method == "average_pairwise":
            return self.average_pairwise_similarity(genres1, genres2)

        elif method == "best_match":
            return self.best_match_similarity(genres1, genres2)

        elif method == "ensemble":
            # Weighted ensemble of multiple methods
            # Weights determined by empirical testing
            jaccard = self.jaccard_similarity(genres1, genres2)
            weighted_jaccard = self.weighted_jaccard_similarity(genres1, genres2)
            cosine = self.cosine_similarity(genres1, genres2)
            best_match = self.best_match_size_normalized(genres1, genres2)

            # Ensemble weights (can be tuned)
            ensemble_score = (
                jaccard * 0.15 +              # Pure overlap (low weight - too strict)
                weighted_jaccard * 0.35 +     # Relationship-aware overlap (high weight)
                cosine * 0.25 +               # Vector similarity (medium weight)
                best_match * 0.25             # Best matching (medium weight)
            )

            return ensemble_score

        elif method == "legacy":
            # Original maximum similarity method (for backward compatibility)
            return self._legacy_max_similarity(genres1, genres2)

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _legacy_max_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Original maximum similarity method (for comparison)

        Returns the single best match between any two genres.
        """
        if not genres1 or not genres2:
            return 0.0

        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        max_similarity = 0.0

        for g1 in genres1:
            for g2 in genres2:
                if g1 == g2:
                    return 1.0
                sim = self._lookup_similarity(g1, g2)
                max_similarity = max(max_similarity, sim)

        return max_similarity

    def compare_methods(self, genres1: List[str], genres2: List[str]) -> Dict[str, float]:
        """
        Compare all similarity methods on two genre lists

        Useful for analysis and tuning.

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Dictionary mapping method name to similarity score
        """
        methods = ["jaccard", "weighted_jaccard", "cosine", "average_pairwise",
                   "best_match", "ensemble", "legacy"]

        results = {}
        for method in methods:
            results[method] = self.calculate_similarity(genres1, genres2, method=method)

        return results

    def get_similar_genres(self, genre: str, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get all genres similar to the given genre

        Args:
            genre: Genre name
            min_similarity: Minimum similarity threshold

        Returns:
            List of (genre_name, similarity_score) tuples, sorted by similarity
        """
        genre = genre.lower().strip()

        if genre not in self.similarity_matrix:
            return []

        similar = [
            (g, score)
            for g, score in self.similarity_matrix[genre].items()
            if score >= min_similarity
        ]

        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)

        return similar

    def get_all_genres(self) -> Set[str]:
        """Get set of all genres in the similarity matrix"""
        return set(self.similarity_matrix.keys())


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    calc = GenreSimilarityV2()

    # Test cases
    test_cases = [
        (["indie rock", "shoegaze"], ["dream pop", "lo-fi"]),
        (["grunge", "alternative rock"], ["noise rock", "indie rock"]),
        (["jazz", "bebop"], ["slowcore", "lo-fi"]),
        (["rock"], ["rock"]),
        (["indie rock", "alternative rock", "post-punk"], ["indie rock", "shoegaze"])
    ]

    logger.info("Genre Similarity Method Comparison:")
    logger.info("=" * 100)

    for genres1, genres2 in test_cases:
        logger.info(f"Comparing: {genres1} vs {genres2}")
        results = calc.compare_methods(genres1, genres2)

        for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {method:20s}: {score:.3f}")
