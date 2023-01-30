from .frozenlake import generate_mdp as generic_frozenlake_generator


def generate_mdp():
    return generic_frozenlake_generator(map_name="8x8")
