from omdt.drn import write_as_drn
from pathlib import Path
import importlib

environments = [
    file.stem
    for file in (Path(__file__).parent / "environments").iterdir()
    if file.is_file()
]

for env in environments:
    Path("drn_exports").mkdir(parents=True, exist_ok=True)
    environment = importlib.import_module(f"environments.{env}")
    mdp = environment.generate_mdp()
    write_as_drn(Path(f"drn_exports/{env}.drn"), mdp)
