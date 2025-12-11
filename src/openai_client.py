"""
OpenAI API Client - Generates genre-based playlist titles
"""
from openai import OpenAI
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for generating playlist titles using OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_playlist_title(self, tracks: List[Dict[str, str]]) -> str:
        """
        Generate a genre-based playlist title

        Args:
            tracks: List of track dictionaries with 'title' and 'artist'

        Returns:
            Generated playlist title
        """
        # Format ALL tracks for analysis
        track_list = self._format_track_list(tracks, max_tracks=None)

        # Simple genre-focused prompt
        prompt = f"""Based on ALL these tracks, generate a simple playlist title that describes the basic genre/style.

Tracks:
{track_list}

Requirements:
- Be straightforward and descriptive
- Focus on genre, mood, or style (e.g., "Indie Rock Mix", "Ambient Electronic", "Jazz Fusion")
- Keep it 2-4 words
- No creative metaphors or poetry
- Just describe what this playlist actually is

Return ONLY the title, nothing else."""

        logger.info(f"Generating genre-based title for {len(tracks)} tracks")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3  # Lower temperature for more straightforward results
            )

            # Extract text from response
            title = response.choices[0].message.content.strip()

            # Clean up any quotes or extra formatting
            title = title.strip('"\'')

            logger.info(f"Generated title: {title}")
            return title

        except Exception as e:
            logger.error(f"Failed to generate title with OpenAI: {e}")
            # Fallback to a generic title
            return self._generate_fallback_title(tracks)

    def _format_track_list(self, tracks: List[Dict[str, str]],
                          max_tracks: int = None) -> str:
        """
        Format track list for inclusion in prompt

        Args:
            tracks: List of track dictionaries
            max_tracks: Maximum number of tracks to include (None = all)

        Returns:
            Formatted string of tracks
        """
        # Use all tracks unless max_tracks is specified
        if max_tracks:
            sample = tracks[:max_tracks]
        else:
            sample = tracks

        formatted = []
        for track in sample:
            artist = track.get('artist', 'Unknown Artist')
            title = track.get('title', 'Unknown Track')
            formatted.append(f"- {artist} - {title}")

        return '\n'.join(formatted)

    def _generate_fallback_title(self, tracks: List[Dict[str, str]]) -> str:
        """
        Generate a simple fallback title if API fails

        Args:
            tracks: List of track dictionaries

        Returns:
            Simple fallback title
        """
        # Try to extract primary artist
        if tracks and tracks[0].get('artist'):
            artist = tracks[0]['artist']
            return f"{artist} Mix"

        return "Music Mix"

    def generate_batch_titles(self, playlist_tracks: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generate titles for multiple playlists

        Args:
            playlist_tracks: List of track lists (one per playlist)

        Returns:
            List of generated titles
        """
        if len(playlist_tracks) == 1:
            logger.info(f"Generating title for playlist 1/1")
            return [self.generate_playlist_title(playlist_tracks[0])]

        # Generate all titles in one batch for efficiency
        logger.info(f"Generating {len(playlist_tracks)} genre-based titles in batch")

        try:
            # Build batch prompt with all playlists
            batch_prompt = self._build_batch_prompt(playlist_tracks)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": batch_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()
            titles = self._parse_batch_titles(response_text, len(playlist_tracks))

            logger.info(f"Generated {len(titles)} titles")
            for i, title in enumerate(titles, 1):
                logger.info(f"  Playlist {i}: {title}")

            return titles

        except Exception as e:
            logger.error(f"Batch title generation failed: {e}")
            # Fallback to individual generation
            logger.info("Falling back to individual title generation")
            titles = []
            for i, tracks in enumerate(playlist_tracks, 1):
                logger.info(f"Generating title for playlist {i}/{len(playlist_tracks)}")
                title = self.generate_playlist_title(tracks)
                titles.append(title)
            return titles

    def _build_batch_prompt(self, playlist_tracks: List[List[Dict[str, str]]]) -> str:
        """Build prompt for generating multiple titles at once"""
        playlist_summaries = []
        for i, tracks in enumerate(playlist_tracks, 1):
            track_list = self._format_track_list(tracks, max_tracks=None)
            playlist_summaries.append(f"PLAYLIST {i}:\n{track_list}")

        all_playlists = "\n\n".join(playlist_summaries)

        prompt = f"""Generate straightforward, genre-based titles for these {len(playlist_tracks)} playlists.

Requirements for each title:
- Describe the basic genre, style, or mood
- Keep it 2-4 words
- Be direct and descriptive (e.g., "Indie Rock Mix", "Ambient Electronic", "Jazz Fusion")
- No creative metaphors or poetry
- Each title should accurately describe that specific playlist

{all_playlists}

Return ONLY the titles, one per line, numbered 1-{len(playlist_tracks)}. Example format:
1. Indie Pop Mix
2. Electronic Ambient
3. Jazz Fusion"""

        return prompt

    def _parse_batch_titles(self, response_text: str, expected_count: int) -> List[str]:
        """Parse titles from batch response"""
        lines = response_text.strip().split('\n')
        titles = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering and quotes
            import re
            title = re.sub(r'^\d+[\.\)]\s*', '', line)
            title = title.strip('"\'')

            if title:
                titles.append(title)

        return titles[:expected_count]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("OpenAI Client module loaded")
