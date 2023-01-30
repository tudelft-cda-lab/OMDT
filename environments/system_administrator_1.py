from .system_administrator import generate_mdp as generic_system_administrator


def generate_mdp():
    return generic_system_administrator(topology="random1")
