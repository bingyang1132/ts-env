# Known gaps

All 110 cards have an event handler, and `Game(strict_events=True)` raises if any card
lacks one. What follows is the list of clauses that are **not** faithfully implemented.
It is deliberately explicit: a documented gap is useful, a silent approximation is not.

Each item names the card, what is missing, and where the fix would go.

## Not implemented

| Card | Missing clause | Where a fix belongs |
|---|---|---|
| **#49 Missile Envy** | "Highest valued card" uses the **printed** operations value, not `effective_ops`, so Red Scare / Containment do not change which card is surrendered. | Arguably correct as-is; flagged because implementations differ. |
| **#40 Cuban Missile Crisis** | The cancel-by-removing-influence option is offered once, at the start of the threatened player's action round. The physical card allows it at any point during the turn. | `Game._offer_cuban_missile_crisis_cancel`; would need a general "player may invoke an ability now" decision point. |
| **#104 The Cambridge Five** | Playable as an event even when the US holds no scoring cards, in which case it reveals nothing. The card's own usage conditions only restrict the Late War, so this matches the data. | `_cambridge_five_playable`. |

## Deferred triggers

Cards worded "on your opponent's next action round ..." are handled by a general
mechanism rather than per-card hacks. `GameState.defer(card, kind, player=, when=)`
schedules a trigger; `Game._fire_deferred` resolves it at the start or end of that
player's next action round. `GameState.ar_sequence` is a monotonic count of action
rounds, and `not_before` is pinned to the current one, which is what makes "next" exclude
the round in progress. Handlers register with `@register_deferred(kind)`.

Two cards use it today:

- **#50 We Will Bury You** schedules its 3 VP for the end of the US player's next action
  round; UN Intervention played as an event in between calls
  `state.cancel_deferred(card="We Will Bury You")` and heads them off.
- **#49 Missile Envy** sets `state.must_play[side]`, which limits that player's next
  action round to spending the received card on operations. The compulsion is cleared
  after that one action round whether or not it could be satisfied, so a player who no
  longer holds the card is not stuck.

Anything else needing this shape should reuse it rather than adding another special case.

## Deliberate rulings

Places where the rules are ambiguous or sources disagree, and the choice made here.

- **Military operations from event-granted operations.** Rule 8.2.5 says a *free* coup
  does not count toward required military operations. Cards worded "conduct Operations
  as if they played an N Op card" are treated as spending real operations and **do**
  credit military ops (`special._conduct_operations`, `mid_war2._operations_with_card`),
  while cards granting an explicitly *free* coup (Junta, Che, Tear Down This Wall) do
  **not** (`helpers.free_coup`, `Game.free_operations`). Two different helpers exist for
  this reason; consolidating them into one `helpers.card_operations(...)` with an
  explicit flag would be tidier.
- **Free coups and DEFCON geography.** Rule 6.3.5 says free coups ignore the DEFCON
  region restrictions. The event-granted coups here are filtered through
  `rules.can_coup`, so they *do* respect DEFCON. This follows the printed prerequisites
  and matches the other open-source implementations, but contradicts a literal reading
  of 6.3.5.
- **#87 The Reformer / #90 Glasnost.** The game's own Lua data has Glasnost remove The
  Reformer from play. That would silently reopen Europe to USSR coups, which the printed
  rules make permanent, so Glasnost leaves it alone.
- **#20 Olympic Games boycott.** The DEFCON degradation is attributed to the sponsor
  (the phasing player), not the boycotting opponent, so a boycott at DEFCON 2 ends the
  game against the sponsor.
- **Cards that stay in play are not in `state.removed`.** A remove-on-event card that
  puts itself on the table (NATO, Formosan Resolution, The Iron Lady, Vietnam Revolts…)
  lives in `state.effects` until its effect ends, and only then reaches a pile. Deck
  integrity is preserved — `tests/test_engine.py::test_every_card_is_always_accounted_for`
  checks every card is always somewhere — but `state.removed` is not a complete list of
  removed cards while effects are live.

## Engine simplifications

- **`Game.clone()` replays the action history** rather than deep-copying, because live
  generator frames cannot be copied. Exact, but O(game length); tree search over long
  games would want its own snapshotting.
- **Forced choices are auto-resolved.** When a decision has exactly one legal option and
  passing is not allowed, the engine takes it without asking. This keeps trivial steps
  out of the trajectory, at the cost of an agent not "seeing" those moments.
- **No Turn Zero variant, no promo cards.** Both are extracted into
  `twilight/data/cards.json` (as `source: "turnzero"` / `"promo"`) but excluded from
  play. The Turn Zero module adds Statecraft cards, a pre-game crisis phase, and the
  Chinese Civil War space.
- **Optional cards are off by default.** Pass `Game(optional_cards=True)` for the seven
  cards numbered 104–110.

## Verifying coverage yourself

```bash
# every card has a handler, and strict mode never trips
python -m pytest tests/test_engine.py -q -k "event_handler or strict_mode"

# 400 games with full invariant checking, including card conservation
python tools/random_play.py --games 400

# regenerate the per-card spec, which marks each card IMPLEMENTED or TODO
python tools/dump_card_spec.py
```
