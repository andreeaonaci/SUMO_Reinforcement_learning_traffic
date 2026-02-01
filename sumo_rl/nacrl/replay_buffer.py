from collections import defaultdict, deque

class MultiAgentReplayBuffer:
    def __init__(self, max_size):
        """Initialize the multi-agent replay buffer.

        Args:
            max_size (int): Maximum size of the buffer (|B|).
        """
        self.max_size = max_size
        self.buffer = defaultdict(lambda: deque(maxlen=max_size))
        self.size = 0

    def add(self, agent_id, s, a, r, s_next):
        """Add a transition to the buffer for a specific agent.

        Args:
            agent_id (str): The ID of the agent.
            s: The current state.
            a: The action taken.
            r: The reward received.
            s_next: The next state.
        """
        self.buffer[agent_id].append((s, a, r, s_next))
        self.size = min(self.size + 1, self.max_size)

    def is_full(self):
        """Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self.size == self.max_size

    def sample(self):
        """Sample all stored transitions grouped by agent_id.

        Returns:
            dict: A dictionary where keys are agent IDs and values are lists of transitions.
        """
        return {agent_id: list(transitions) for agent_id, transitions in self.buffer.items()}

    def clear(self):
        """Clear the replay buffer."""
        self.buffer.clear()
        self.size = 0