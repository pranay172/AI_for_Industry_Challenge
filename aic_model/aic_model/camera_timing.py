"""Exposure selection shared by live perception and offline replay (nanoseconds)."""


def synchronized_exposures(stamps, now_ns):
    fresh = {name: stamp for name, stamp in stamps.items()
             if stamp > 0 and 0 <= now_ns - stamp <= 500_000_000}
    newest = max(fresh.values(), default=0)
    return [name for name, stamp in fresh.items() if newest - stamp <= 50_000_000]
