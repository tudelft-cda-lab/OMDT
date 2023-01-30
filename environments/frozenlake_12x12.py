from .frozenlake import generate_mdp as generic_frozenlake_generator

desc = [
    "SFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "FFFHFFFFFFFH",
    "FFFFFHFFFFFF",
    "FFFHFFFFFFFF",
    "FHHFFFHFFHFF",
    "FHFFHFHFFFFF",
    "FFFHFFFFFFFF",
    "FFFFFFFFHFFF",
    "HFFFFHFFFFHH",
    "FFFFFFGFFFFF",
    "FFFFFFFFFFFF",
]


def generate_mdp():
    return generic_frozenlake_generator(desc=desc)
