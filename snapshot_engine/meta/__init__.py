"""
META level — the layer ABOVE applied_data.

The primitives below answer "what is happening in this price series". The meta
level asks the question that needs several of them at once: *what did one day
look like, taken as a whole, and which other days looked the same?*

Two rules define this layer:

  * It reads primitive results ONLY through :mod:`snapshot_engine.meta.cache`, the lazy
    get-or-compute-and-store front door. It never calls a primitive directly and
    never redoes a primitive's maths.
  * It respects the time-class split: what a day KNEW (Block A, strictly
    past-only) is kept apart from what the days AFTER it shaped (Block B). Only
    Block A is ever compared.

Nothing below this layer imports anything from it.
"""
