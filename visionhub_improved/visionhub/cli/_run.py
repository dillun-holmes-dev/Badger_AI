import runpy


def run_module(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__")
