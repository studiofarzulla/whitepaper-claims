#!/usr/bin/env python3
"""
LLM-based Zero-Shot Classification for TENSOR-DEFI Expansion

Uses LLM API for fast, accurate classification of whitepaper chunks
into 10 functional categories. Batches chunks for efficiency.

Supports: OpenAI (gpt-4o-mini), Anthropic (claude-3-haiku), Local (LM Studio)
"""

import json
import logging
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm

from .taxonomy import LABEL_TO_CATEGORY, FUNCTIONAL_CATEGORIES, get_category_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 25  # Chunks per API call


class LLMClassifier:
    """
    LLM-based zero-shot classifier using OpenAI, Anthropic, or local LLM.

    Much faster than HuggingFace ensemble for this task.
    """

    def __init__(
        self,
        provider: str = "local",
        model: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        max_workers: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM classifier.

        Args:
            provider: "openai", "anthropic", or "local" (LM Studio)
            model: Model name (default varies by provider)
            batch_size: Chunks per API call
            max_workers: Parallel API calls
            api_key: Optional explicit API key
            base_url: Base URL for local LLM (default: http://localhost:1234/v1)
        """
        self.provider = provider
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.categories = get_category_names()

        if provider == "local":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install openai: pip install openai")

            self.model = model or "local-model"
            self.client = OpenAI(
                base_url=base_url or "http://localhost:1234/v1",
                api_key="lm-studio"  # LM Studio doesn't need real key
            )

        elif provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install openai: pip install openai")

            self.model = model or "gpt-4o-mini"
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=key)

        elif provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")

            self.model = model or "claude-3-5-haiku-20241022"
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=key)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Build category descriptions for the prompt
        self.category_desc = self._build_category_descriptions()

        logger.info(f"LLM Classifier: provider={provider}, model={self.model}, batch={batch_size}")

    def _build_category_descriptions(self) -> str:
        """Build formatted category descriptions for the prompt."""
        lines = []
        for cat in self.categories:
            info = FUNCTIONAL_CATEGORIES[cat]
            lines.append(f"- {cat}: {info['description']}")
        return "\n".join(lines)

    def _create_batch_prompt(self, chunks: list[str]) -> str:
        """Create prompt for batch classification."""
        chunks_formatted = "\n\n".join([
            f"[CHUNK {i+1}]\n{chunk[:2000]}"  # Truncate long chunks
            for i, chunk in enumerate(chunks)
        ])

        return f"""Classify each cryptocurrency whitepaper chunk into these 10 functional categories.
For each chunk, provide a confidence score (0.0-1.0) for EACH category.

Categories:
{self.category_desc}

Chunks to classify:
{chunks_formatted}

Respond with a JSON array where each element corresponds to a chunk and contains scores for all 10 categories.
Example format:
[
  {{"store_of_value": 0.1, "medium_of_exchange": 0.8, "smart_contracts": 0.2, "defi": 0.3, "governance": 0.1, "scalability": 0.4, "privacy": 0.0, "interoperability": 0.1, "data_storage": 0.0, "oracle": 0.0}},
  ...
]

Return ONLY the JSON array, no other text."""

    def _call_api(self, prompt: str) -> str:
        """Call the LLM API and return response text."""
        if self.provider == "local":
            # Local LM Studio - no response_format, just direct call
            # temperature=0 + seed for reproducible classification (jun27 clean re-run)
            # reasoning_effort=none disables reasoning-model thinking (e.g. Nemotron-3),
            # which otherwise balloons tokens past max_tokens and truncates the JSON.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0,
                seed=42,
                extra_body={"reasoning_effort": "none"}
            )
            return response.choices[0].message.content

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    def _clean_json(self, text: str) -> str:
        """
        Clean LLM output to extract valid JSON.

        Handles:
        - Code blocks (```json ... ```)
        - Comments (// ...)
        - Bold markers (**text**)
        - Python code wrapping
        """
        # Strip code blocks
        text = re.sub(r'```\w*\n?', '', text).strip()

        # Strip // comments (common in Ministral outputs)
        text = re.sub(r'//.*', '', text, flags=re.MULTILINE)

        # Strip bold markers
        text = re.sub(r'\*\*', '', text)

        # Find JSON array or object
        # Try to find array first (our expected format)
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group()

        # Try object
        obj_match = re.search(r'\{[\s\S]*\}', text)
        if obj_match:
            return obj_match.group()

        return text

    def classify_batch(self, chunks: list[str]) -> list[dict[str, float]]:
        """
        Classify a batch of chunks in a single API call.

        Returns list of score dicts, one per chunk.
        """
        if not chunks:
            return []

        prompt = self._create_batch_prompt(chunks)

        try:
            content = self._call_api(prompt).strip()

            # Clean the response (handles code blocks, comments, etc.)
            content = self._clean_json(content)

            # Parse JSON
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Find the array in the dict (OpenAI json_object mode)
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break

            scores_list = parsed

            # Validate and fill missing categories
            validated = []
            for scores in scores_list:
                validated_scores = {}
                for cat in self.categories:
                    validated_scores[cat] = float(scores.get(cat, 0.0))
                validated.append(validated_scores)

            return validated

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Raw content: {content[:500]}...")
            return [{cat: 0.0 for cat in self.categories} for _ in chunks]
        except Exception as e:
            logger.error(f"API error: {e}")
            return [{cat: 0.0 for cat in self.categories} for _ in chunks]

    def classify_document(
        self,
        chunks: list[str],
        symbol: str
    ) -> np.ndarray:
        """
        Classify all chunks for a document and aggregate.

        Args:
            chunks: List of text chunks
            symbol: Asset symbol (for logging)

        Returns:
            Document profile as normalized probability vector (10,)
        """
        if not chunks:
            logger.warning(f"{symbol}: No chunks")
            return np.zeros(len(self.categories))

        # Batch chunks
        all_scores = []
        batches = [chunks[i:i+self.batch_size] for i in range(0, len(chunks), self.batch_size)]

        for batch in batches:
            batch_scores = self.classify_batch(batch)
            all_scores.extend(batch_scores)

        # Convert to array
        score_matrix = np.array([
            [s[cat] for cat in self.categories]
            for s in all_scores
        ])

        # Aggregate: mean across chunks
        profile = score_matrix.mean(axis=0)

        # Normalize to probability distribution
        if profile.sum() > 0:
            profile = profile / profile.sum()

        return profile

    def build_claims_matrix(
        self,
        chunks_path: Path,
        output_dir: Path,
        parallel: bool = True
    ) -> dict:
        """
        Build claims matrix for all assets.

        Args:
            chunks_path: Path to extracted_chunks.json
            output_dir: Output directory
            parallel: Use parallel processing (faster)

        Returns:
            Dict with matrix, symbols, etc.
        """
        with open(chunks_path) as f:
            chunks_data = json.load(f)

        symbols = sorted(chunks_data.keys())
        n_entities = len(symbols)
        n_categories = len(self.categories)

        matrix = np.zeros((n_entities, n_categories))

        if parallel:
            # Process assets in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.classify_document,
                        chunks_data[sym]['chunks'],
                        sym
                    ): (i, sym)
                    for i, sym in enumerate(symbols)
                }

                for future in tqdm(as_completed(futures), total=n_entities, desc="LLM Classification"):
                    i, sym = futures[future]
                    try:
                        profile = future.result()
                        matrix[i] = profile
                    except Exception as e:
                        logger.error(f"{sym} failed: {e}")
        else:
            # Sequential processing
            for i, sym in enumerate(tqdm(symbols, desc="LLM Classification")):
                chunks = chunks_data[sym]['chunks']
                profile = self.classify_document(chunks, sym)
                matrix[i] = profile

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main output: claims_matrix_llm.npy
        np.save(output_dir / "claims_matrix_llm.npy", matrix)

        # Metadata
        metadata = {
            'symbols': symbols,
            'categories': self.categories,
            'shape': list(matrix.shape),
            'provider': self.provider,
            'model': self.model,
            'batch_size': self.batch_size
        }
        with open(output_dir / "claims_matrix_llm_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # CSV for inspection
        import pandas as pd
        df = pd.DataFrame(matrix, index=symbols, columns=self.categories)
        df.to_csv(output_dir / "claims_matrix_llm.csv")

        self._print_summary(symbols, matrix)

        return {
            'matrix': matrix,
            'symbols': symbols,
            'categories': self.categories
        }

    def _print_summary(self, symbols: list[str], matrix: np.ndarray):
        """Print classification summary."""
        print(f"\n{'='*60}")
        print("LLM CLAIMS MATRIX SUMMARY")
        print(f"{'='*60}")
        print(f"Assets:     {len(symbols)}")
        print(f"Categories: {len(self.categories)}")
        print(f"Provider:   {self.provider}")
        print(f"Model:      {self.model}")
        print(f"\nCategory distribution (mean across assets):")

        mean_scores = matrix.mean(axis=0)
        sorted_idx = np.argsort(mean_scores)[::-1]

        for idx in sorted_idx:
            cat = self.categories[idx]
            score = mean_scores[idx]
            bar = '#' * int(score * 40)
            print(f"  {cat:20s} {score:.3f} {bar}")

        print(f"{'='*60}")


def main():
    """Run LLM classification."""
    base_path = Path(__file__).parent.parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    if not chunks_path.exists():
        logger.error("Run pdf_extractor.py first")
        return

    # Try local first (LM Studio), then OpenAI, then Anthropic
    try:
        classifier = LLMClassifier(provider="local")
    except Exception:
        try:
            classifier = LLMClassifier(provider="openai")
        except (ImportError, ValueError):
            classifier = LLMClassifier(provider="anthropic")

    results = classifier.build_claims_matrix(chunks_path, output_dir)

    print(f"\nDone! Matrix shape: {results['matrix'].shape}")


if __name__ == "__main__":
    main()
