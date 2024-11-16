import heapq
import random
import string

# Define the alphabet
alphabet = string.ascii_lowercase  # For lowercase letters


class BeamSearch:
    def __init__(self, beam_width):
        self.beam_width = beam_width

    def search(self, start_state, expand_fn, score_fn, end_fn):
        # Initialize the beam with the start state
        beam = [(0, start_state)]  # (score, state)

        while True:
            # Expand each state in the beam
            all_candidates = []
            for score, state in beam:
                if end_fn(state):
                    return state  # Return the sequence if end condition is met
                for next_state in expand_fn(state):
                    next_score = score + score_fn(next_state)
                    all_candidates.append((next_score, next_state))

            # Prune to keep only the top k states
            beam = heapq.nlargest(self.beam_width, all_candidates, key=lambda x: x[0])


# Example usage
def expand_fn(state):
    # Example expansion function: generate next states
    # Pick a random letter from the alphabet
    return [
        state + random.choice(alphabet),
        state + random.choice(alphabet),
        state + random.choice(alphabet),
    ]


def score_fn(state):
    # Example scoring function: score based on length
    return len(set(state))


def end_fn(state):
    # Example end condition: stop if state length is 5
    return len(state) == 36


beam_search = BeamSearch(beam_width=3)
start_state = ""
best_sequence = beam_search.search(start_state, expand_fn, score_fn, end_fn)
print("Best sequence:", best_sequence)
