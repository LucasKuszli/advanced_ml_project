from src.chess.base import ChessPosition

__all__ = ["ChessPosition", "BoardRenderer"]


def __getattr__(name: str):
    if name == "BoardRenderer":
        from src.chess.visualize import BoardRenderer
        return BoardRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
