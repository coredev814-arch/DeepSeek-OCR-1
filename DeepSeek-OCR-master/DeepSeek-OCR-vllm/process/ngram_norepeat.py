import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int, window_size: int = 100,
                 whitelist_token_ids: set = None,
                 max_consecutive_empty_cells: int = 60):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        # Max consecutive table-structure tokens before banning whitelist tokens.
        # Table tokens = <td>(128821), </td>(128822), <tr>(128823), </tr>(128824)
        self.max_consecutive_table_tokens = max_consecutive_empty_cells
        self._table_token_ids = (whitelist_token_ids or set()) | {128823, 128824}

    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(input_ids) < self.ngram_size:
            return scores

        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])

        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        # Detect runaway empty table rows: if the last N tokens are all
        # table-structure tokens (<td>, </td>, <tr>, </tr>), the model is
        # stuck generating empty cells.  Ban the whitelist tokens to force
        # it to produce actual content or close the table.
        if self._table_token_ids and self.max_consecutive_table_tokens > 0:
            tail_len = min(len(input_ids), self.max_consecutive_table_tokens)
            tail = input_ids[-tail_len:]
            if tail_len >= self.max_consecutive_table_tokens and all(
                t in self._table_token_ids for t in tail
            ):
                banned_tokens |= self.whitelist_token_ids

        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")

        return scores
