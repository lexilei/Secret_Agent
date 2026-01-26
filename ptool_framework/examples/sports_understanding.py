"""Sports Understanding Example

This example is adapted from William Cohen's secretagent demo.
It demonstrates how ptools can be composed to solve the "sports understanding" task.

The task: Given a sentence about a sports event, determine if it makes sense
(i.e., the player, action, and event are all from the same sport).
"""

import pprint
from typing import Tuple

# Import the ptool framework
import sys
sys.path.insert(0, '..')
from ptool_framework import ptool, get_registry, enable_tracing


# Define ptools for the sports understanding task

@ptool(model="deepseek-v3")
def analyze_sentence(sentence: str) -> Tuple[str, str, str]:
    """Extract a names of a player, and action, and an optional event.

    The action should be as descriptive as possible.  The event will be
    an empty string if no event is mentioned in the sentence.

    Examples:
    >>> analyze_sentence("Bam Adebayo scored a reverse layup in the Western Conference Finals.")
    ('Bam Adebayo', 'scored a reverse layup', 'in the Western Conference Finals.')
    >>> analyze_sentence('Santi Cazorla scored a touchdown.')
    ('Santi Cazorla', 'scored a touchdown.', '')
    """
    ...


@ptool(model="deepseek-v3")
def sport_for(x: str) -> str:
    """Return the name of the sport associated with a player, action, or event.

    Examples:
    >>> sport_for('Bam Adebayo')
    'basketball'
    >>> sport_for('scored a reverse layup')
    'basketball'
    >>> sport_for('in the Western Conference Finals.')
    'basketball'
    >>> sport_for('Santi Cazorla')
    'soccer'
    >>> sport_for('scored a touchdown.')
    'American football and rugby'
    """
    ...


@ptool(model="deepseek-v3")
def consistent_sports(sport1: str, sport2: str) -> bool:
    """Compare two descriptions of sports, and determine if they are consistent.

    Descriptions are consistent if they are the same, or if one is more
    general than the other.
    """
    ...


# The workflow that composes the ptools

def sports_understanding_workflow(sentence: str) -> bool:
    """A workflow that uses the ptools defined above to check if a sports sentence makes sense."""

    # Step 1: Extract player, action, event from sentence
    player, action, event = analyze_sentence(sentence)
    print(f"  Player: {player}")
    print(f"  Action: {action}")
    print(f"  Event: {event}")

    # Step 2: Get sport for player and action
    player_sport = sport_for(player)
    action_sport = sport_for(action)
    print(f"  Player sport: {player_sport}")
    print(f"  Action sport: {action_sport}")

    # Step 3: Check consistency
    result = consistent_sports(player_sport, action_sport)

    # Step 4: If there's an event, check that too
    if event:
        event_sport = sport_for(event)
        print(f"  Event sport: {event_sport}")
        result = result and consistent_sports(player_sport, event_sport)

    print(f"  Final answer: {'yes' if result else 'no'}")
    return result


if __name__ == '__main__':
    # Enable tracing to collect execution data
    enable_tracing(True)

    # Test cases
    test_sentences = [
        "Tim Duncan scored from inside the paint.",  # True - basketball
        "Santi Cazorla scored a touchdown.",  # False - soccer player, football action
        "DeMar DeRozan was called for the goal tend.",  # True - basketball
        "Bam Adebayo scored a reverse layup in the Western Conference Finals.",  # True - basketball
    ]

    print("=" * 60)
    print("Sports Understanding Task")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        try:
            result = sports_understanding_workflow(sentence)
        except Exception as e:
            print(f"  Error: {e}")
