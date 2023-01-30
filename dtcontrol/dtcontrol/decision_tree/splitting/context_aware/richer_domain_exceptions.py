class RicherDomainStrategyException(Exception):
    """
    Raised when an invalid state inside richer_domain_splitting_strategy.py is reached.
    """
    pass


class RicherDomainSplitException(Exception):
    """
    Raised when a Richer Domain Split Object reaches an invalid states.
    """
    pass


class RicherDomainGeneratorException(Exception):
    """
    Raised when an invalid state inside richer_domain_cli_strategy.py is reached.
    """
    pass


class RicherDomainPredicateParserException(Exception):
    """
    Raised when an invalid state inside predicate_parser.py is reached.
    """

    def __init__(self):
        super().__init__('Aborting: Invalid predicate. Check logger or comments for more information.')
