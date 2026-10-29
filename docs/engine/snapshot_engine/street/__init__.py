"""
THE STREET — where decisions are ranked, and nothing else happens.

A FIRING analysis (snapshot/robustness today) emits a :class:`snapshot_engine.street.model.Decision`
with an initial score and a GRIP CONDITION: the exact parameter picture the decision
is waiting for. An EVALUATING analysis (``ath_meta``, ``trend_meta``, anything added
later) never fires. Each is a STATION on the street: it inspects a passing decision
and returns a MULTIPLICATIVE factor — slightly above 1 to boost, below 1 to damp,
never 0, exactly 1.0 when it has nothing to say.

A decision walks every station in order, its score is multiplied by each factor and
capped, and it lands in a list sorted by score alone.
"""
