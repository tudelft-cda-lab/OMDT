import numpy as np

from itertools import product

from functools import reduce

from omdt.mdp import MarkovDecisionProcess

ACTIONLIST = {0: "skip", 1: "draw"}

CARDS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
BLACKJACK = 21
DEALER_SKIP = 17
STARTING_CARDS_PLAYER = 2
STARTING_CARDS_DEALER = 1

STATELIST = {0: (0, 0, 0)}  # Game start state
STATELIST = {
    **STATELIST,
    **{
        nr + 1: state
        for nr, state in enumerate(
            product(
                range(2),
                range(CARDS.min() * STARTING_CARDS_PLAYER, BLACKJACK + 2),
                range(CARDS.min() * STARTING_CARDS_DEALER, BLACKJACK + 2),
            )
        )
    },
}


def cartesian(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2).sum(axis=1)


def deal_card_probability(count_now, count_next, take=1):
    if take > 1:
        cards = reduce(cartesian, [CARDS] * take)
    else:
        cards = CARDS

    return (np.minimum(count_now + cards, BLACKJACK + 1) == count_next).sum() / len(
        cards
    )


def is_gameover(skipped, player, dealer):
    return any(
        [
            dealer >= DEALER_SKIP and skipped == 1,
            dealer > BLACKJACK and skipped == 1,
            player > BLACKJACK,
        ]
    )


def blackjack_probability(action, stateid_now, stateid_next):
    skipped_now, player_now, dealer_now = STATELIST[stateid_now]
    skipped_next, player_next, dealer_next = STATELIST[stateid_next]

    if stateid_now == stateid_next:
        # Game cannot stay in current state
        return 0.0

    if stateid_now == 0:
        if skipped_next == 1:
            # After start of the game the game cannot be in a skipped state
            return 0
        else:
            # State lower or equal than 1 is a start of a new game
            dealer_prob = deal_card_probability(
                0, dealer_next, take=STARTING_CARDS_DEALER
            )
            player_prob = deal_card_probability(
                0, player_next, take=STARTING_CARDS_PLAYER
            )

            return dealer_prob * player_prob

    if is_gameover(skipped_now, player_now, dealer_now):
        # We arrived at end state, now reset game
        return 1.0 if stateid_next == 0 else 0.0

    if skipped_now == 1:
        if skipped_next == 0 or player_next != player_now:
            # Once you skip you keep on skipping in blackjack
            # Also player cards cannot increase once in a skipped state
            return 0.0

    if ACTIONLIST[action] == "skip" or skipped_now == 1:
        # If willingly skipped or in forced skip (attempted draw in already skipped game):
        if skipped_next != 1 or player_now != player_next:
            # Next state must be a skipped state with same card count for player
            return 0.0

    if ACTIONLIST[action] == "draw" and skipped_now == 0 and skipped_next != 0:
        # Next state must be a drawable state
        return 0.0

    if dealer_now != dealer_next and player_now != player_next:
        # Only the player or the dealer can draw a card. Not both simultaneously!
        return 0.0

    # Now either the dealer or the player draws a card
    if ACTIONLIST[action] == "draw" and skipped_now == 0:
        # Player draws a card
        prob = deal_card_probability(player_now, player_next, take=1)
    else:
        # Dealer draws a card
        if dealer_now >= DEALER_SKIP:
            if dealer_now != dealer_next:
                # Dealer always stands once it has a card count higher than set amount
                return 0.0
            else:
                # Dealer stands
                return 1.0

        prob = deal_card_probability(dealer_now, dealer_next, take=1)

    return prob


def blackjack_rewards(action, stateid):
    skipped, player, dealer = STATELIST[stateid]

    if not is_gameover(skipped, player, dealer):
        return 0
    elif player > BLACKJACK or (player <= dealer and dealer <= BLACKJACK):
        return -1
    elif player == BLACKJACK and dealer < BLACKJACK:
        return 1.5
    elif player > dealer or dealer > BLACKJACK:
        return 1
    else:
        raise Exception(f"Undefined reward: {skipped}, {player}, {dealer}")


def generate_mdp():
    # Define transition matrix
    T = np.zeros((len(ACTIONLIST), len(STATELIST), len(STATELIST)))
    for a, i, j in product(ACTIONLIST.keys(), STATELIST.keys(), STATELIST.keys()):
        T[a, i, j] = blackjack_probability(a, i, j)

    # Define reward matrix
    R = np.zeros((len(STATELIST), len(ACTIONLIST)))
    for a, s in product(ACTIONLIST.keys(), STATELIST.keys()):
        R[s, a] = blackjack_rewards(a, s)

    # Check that we have a valid transition matrix with transition probabilities summing to 1
    assert (T.sum(axis=2).round(10) == 1).all()

    # Change reward axes from state, action to state, new_state, action
    R_new = np.empty((len(STATELIST), len(STATELIST), len(ACTIONLIST)))
    R_new[:, :, :] = R[:, np.newaxis, :]
    R = R_new

    # Change order of axes to state, new_state, action
    T = np.moveaxis(T, 0, -1)

    # Extract the observations from the STATELIST dict
    observations = np.array([STATELIST[i] for i in range(len(STATELIST))], dtype=float)
    feature_names = ["skipped", "player_total", "dealer_total"]

    # In this simplified blackjack game players only skip or draw
    action_names = ["Skip", "Draw"]

    # We always start in the state (0, 0, 0):
    # we haven't skipped, the player has no cards, the dealer has no cards
    initial_state_p = np.zeros(len(STATELIST))
    initial_state_p[0] = 1

    return MarkovDecisionProcess(
        trans_probs=T,
        rewards=R,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
