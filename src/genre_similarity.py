"""
Genre Similarity Calculator
Calculates similarity between genres using a curated similarity matrix
"""
import yaml
from pathlib import Path
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


class GenreSimilarity:
    """Calculates genre similarity scores"""

    def __init__(self, similarity_file: str = "data/genre_similarity.yaml"):
        """
        Initialize genre similarity calculator

        Args:
            similarity_file: Path to genre similarity YAML file
        """
        self.similarity_matrix = {}
        self._load_similarity_matrix(similarity_file)
        logger.info(f"Loaded genre similarity matrix with {len(self.similarity_matrix)} genres")

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

    def calculate_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """
        Calculate similarity between two lists of genres

        Strategy:
        1. Direct match = 1.0
        2. Look up in similarity matrix
        3. Check reverse direction (genre2 -> genre1)
        4. Return maximum similarity found

        Args:
            genres1: First list of genres
            genres2: Second list of genres

        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not genres1 or not genres2:
            return 0.0

        # Normalize genre names (lowercase, strip)
        genres1 = [g.lower().strip() for g in genres1]
        genres2 = [g.lower().strip() for g in genres2]

        max_similarity = 0.0

        for g1 in genres1:
            for g2 in genres2:
                # Direct match
                if g1 == g2:
                    return 1.0

                # Look up in similarity matrix
                sim = self._lookup_similarity(g1, g2)
                max_similarity = max(max_similarity, sim)

        return max_similarity

    def _lookup_similarity(self, genre1: str, genre2: str) -> float:
        """
        Look up similarity between two genres

        Checks both directions: genre1->genre2 and genre2->genre1
        """
        # Check genre1 -> genre2
        if genre1 in self.similarity_matrix:
            if genre2 in self.similarity_matrix[genre1]:
                return self.similarity_matrix[genre1][genre2]

        # Check genre2 -> genre1 (reverse direction)
        if genre2 in self.similarity_matrix:
            if genre1 in self.similarity_matrix[genre2]:
                return self.similarity_matrix[genre2][genre1]

        return 0.0

    def get_similar_genres(self, genre: str, min_similarity: float = 0.5) -> List[tuple]:
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
