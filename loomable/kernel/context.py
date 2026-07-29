"""Context Manager for the loomable agent framework.

Manages the agent's context window against a configured token budget.
Handles admission, eviction, and assembly of context items with pinning
support for system prompts and tool schemas.

Requirements covered: 9.1, 9.2, 9.3, 9.4, 9.5, 11.1
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loomable.kernel.models import ContextItem, ContextWindow


@dataclass
class AdmissionResult:
    """Result of attempting to admit an item into the context window.

    Attributes:
        admitted: Whether the item was successfully admitted.
        evicted: List of items that were evicted to make room.
    """

    admitted: bool
    evicted: list[ContextItem] = field(default_factory=list)


class ContextManager:
    """Manages the context window against a configured token budget.

    Key behaviors:
    - Token budget accounting: tracks live token count and updates on every admission.
    - Evict-then-admit: if admitting an item would exceed budget, lower-priority
      non-pinned items are evicted until count <= budget before admitting.
    - System prompt and tool schemas are always pinned and never evicted.
    - assemble() returns a ContextWindow with pinned items at the head.
    - current_tokens() returns the sum of tokens of currently retained items.
    """

    def __init__(self, token_budget: int) -> None:
        """Initialize the Context Manager.

        Args:
            token_budget: Maximum token count for the context window.
        """
        self.token_budget = token_budget
        self._items: list[ContextItem] = []

    def current_tokens(self) -> int:
        """Return the total token count of all retained items.

        This equals the sum of the tokens of each currently retained item.
        """
        return sum(item.tokens for item in self._items)

    def admit(self, item: ContextItem) -> AdmissionResult:
        """Attempt to admit a context item into the window.

        Uses evict-then-admit strategy:
        1. If the item fits within budget, admit immediately.
        2. If not, evict lowest-priority non-pinned items until it fits.
        3. If evicting all non-pinned items still can't make room (because
           pinned items + new item exceed budget), refuse admission.

        Pinned items (system prompt, tool schemas) are never evicted.

        Args:
            item: The context item to admit.

        Returns:
            AdmissionResult indicating whether the item was admitted
            and which items (if any) were evicted to make room.
        """
        current = self.current_tokens()
        new_total = current + item.tokens

        # Case 1: Item fits within budget - admit directly
        if new_total <= self.token_budget:
            self._items.append(item)
            return AdmissionResult(admitted=True, evicted=[])

        # Case 2: Need to evict non-pinned items to make room
        # Calculate how many tokens we need to free
        tokens_to_free = new_total - self.token_budget

        # Gather eviction candidates: non-pinned items sorted by priority (lowest first)
        candidates = [
            (i, it) for i, it in enumerate(self._items) if not it.pinned
        ]
        candidates.sort(key=lambda x: x[1].priority)

        evicted: list[ContextItem] = []
        freed = 0

        for _idx, candidate_item in candidates:
            if freed >= tokens_to_free:
                break
            evicted.append(candidate_item)
            freed += candidate_item.tokens

        # Case 3: Cannot free enough tokens without evicting pinned items
        if freed < tokens_to_free:
            return AdmissionResult(admitted=False, evicted=[])

        # Remove evicted items from the internal list
        evicted_set = set(id(item) for item in evicted)
        self._items = [it for it in self._items if id(it) not in evicted_set]

        # Admit the new item
        self._items.append(item)

        return AdmissionResult(admitted=True, evicted=evicted)

    def assemble(self) -> ContextWindow:
        """Assemble the context window with pinned items at the head.

        Returns a ContextWindow where:
        - Pinned items (system prompt, tool schemas) appear first, ordered
          by their original insertion order.
        - Non-pinned items follow, ordered by priority (highest first).

        This placement enables provider-side prefix caching.
        """
        pinned = [item for item in self._items if item.pinned]
        non_pinned = [item for item in self._items if not item.pinned]

        # Non-pinned items sorted by priority descending (highest priority first)
        non_pinned.sort(key=lambda x: x.priority, reverse=True)

        return ContextWindow(items=pinned + non_pinned)
