def generate_ludii_mnk_game(m, n, k):
    game_description = f"""
    (game "MNK Game"
        (players 2)
        (equipment
            {{ (board (square {m} {n})) }}
            (piece "Mark" Each)
        )
        (rules
            (play
                (move Add (to (empty)))
            )
            (end
                (if (is Line {k}) (result Mover Win))
                (if (no Moves Left) (result Draw))
            )
        )
    )
    """
    with open("input/input.txt", "w") as f:
        f.write(game_description.strip())