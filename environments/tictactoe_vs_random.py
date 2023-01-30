import numpy as np

from omdt.mdp import MarkovDecisionProcess

from itertools import product

from tqdm import tqdm


def insert_tuple(original, i, value):
    return (*original[:i], value, *original[i + 1 :])


def valid_moves(observation):
    next_states = []
    for i in range(9):
        # If this square is empty
        if observation[3 * i] == 0:
            next_states.append(insert_tuple(observation, 3 * i + 1, 1))
            next_states.append(insert_tuple(observation, 3 * i + 2, 1))
    return next_states


def insert_board(board, x, y, value):
    new_board = []
    for i, row in enumerate(board):
        if i == x:
            new_row = (*row[:y], value, *row[y + 1 :])
            new_board.append(new_row)
        else:
            new_board.append(row)
    return tuple(new_board)


def generate_next_boards(board, player=None):
    for x in range(3):
        for y in range(3):
            if board[x][y] == 0:
                if player is None:
                    yield insert_board(board, x, y, 1)
                    yield insert_board(board, x, y, 2)
                else:
                    yield insert_board(board, x, y, player)


def is_win(board, player):
    # From https://github.com/Cledersonbc/tic-tac-toe-minimax/blob/master/py_version/minimax.py
    win_states = [
        (board[0][0], board[0][1], board[0][2]),
        (board[1][0], board[1][1], board[1][2]),
        (board[2][0], board[2][1], board[2][2]),
        (board[0][0], board[1][0], board[2][0]),
        (board[0][1], board[1][1], board[2][1]),
        (board[0][2], board[1][2], board[2][2]),
        (board[0][0], board[1][1], board[2][2]),
        (board[2][0], board[1][1], board[0][2]),
    ]
    if (player, player, player) in win_states:
        return True
    else:
        return False


def is_tie(board):
    for row in board:
        for value in row:
            if value == 0:
                return False

    if is_win(board, 1) or is_win(board, 2):
        return False

    return True


def generate_mdp():
    start_board = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    all_boards = {start_board}
    stack = [(start_board, 1)]
    while stack:
        board, player_turn = stack.pop()
        for new_board in generate_next_boards(board, player_turn):
            if (
                new_board not in all_boards
                and not is_win(new_board, 1)
                and not is_win(new_board, 2)
            ):
                if player_turn == 1:
                    all_boards.add(new_board)
                    stack.append((new_board, 2))
                else:
                    all_boards.add(new_board)
                    stack.append((new_board, 1))
    all_boards = list(all_boards)

    observations = []
    for board in all_boards:
        observation = []
        for row in board:
            for value in row:
                if value == 0:
                    observation.extend((1, 0, 0))
                elif value == 1:
                    observation.extend((0, 1, 0))
                else:
                    observation.extend((0, 0, 1))
        observations.append(observation)

    # Observation for terminal state where actions don't matter
    observations.append([-1] * 3 * 9)

    observations = np.array(observations, dtype=np.int8)

    position_names = [
        "top_left",
        "top_center",
        "top_right",
        "center_left",
        "center",
        "center_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]
    action_names = position_names
    feature_names = []
    for name in position_names:
        feature_names.append(name + "_free")
        feature_names.append(name + "_cross")
        feature_names.append(name + "_circle")

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    board_to_state = {board: s for s, board in enumerate(all_boards)}

    for s, board in tqdm(enumerate(all_boards), total=len(all_boards)):
        for row_i in range(3):
            for col_i in range(3):
                action = 3 * row_i + col_i
                if board[row_i][col_i] != 0:
                    # Placing a symbol in an illegal square loses you the game
                    R[s, :, action] = -1
                    T[s, -1, action] = 1
                else:
                    board_after_action = insert_board(board, row_i, col_i, 1)
                    if is_win(board_after_action, 1):
                        # If this action wins the game then immediately get a reward
                        # and go to the terminal state
                        R[s, :, action] = 1
                        T[s, -1, action] = 1
                    elif is_tie(board_after_action):
                        # If this action ties the game then go to the terminal state
                        # without reward. Since our player starts the game a tie
                        # only occurs in this player's turn.
                        T[s, -1, action] = 1
                    else:
                        # Otherwise the opponent makes a move uniformly at random
                        next_states = []
                        n_losses = 0
                        for next_board in generate_next_boards(board_after_action, 2):
                            if is_win(next_board, 2):
                                n_losses += 1
                            else:
                                next_states.append(board_to_state[next_board])

                        if len(next_states) > 0:
                            # The states were not lost appear uniformly at random with 0 reward
                            T[s, next_states, action] = 1 / (
                                len(next_states) + n_losses
                            )

                        if n_losses > 0:
                            # Upon a loss we go to the terminal state and get -1 reward
                            T[s, -1, action] = n_losses / (len(next_states) + n_losses)
                            R[s, -1, action] = -1

    # The terminal state loops to itself with no reward
    T[-1, -1, :] = 1

    for i in np.unique(np.where(1 - np.isclose(T.sum(axis=1), 1))[0]):
        print(i, all_boards[i])

    # We always start with an empty board
    initial_state_p = np.zeros(n_states)
    initial_state_p[board_to_state[start_board]] = 1

    return MarkovDecisionProcess(
        trans_probs=T,
        rewards=R,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
