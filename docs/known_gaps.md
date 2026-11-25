# Known gaps

All 110 cards have an event handler, and `Game(strict_events=True)` raises if any card
lacks one. This page lists what is **not** faithfully implemented, the rulings the engine
follows where sources are silent or disagree, and what no test covers yet. A documented
gap is useful, a silent approximation is not.

The rules audit of 2026-11 (branch `rules-fix`) fixed every item it found; the fixes and
the rule behind each are listed in the README's *Changes* section. Regression tests:
`tests/test_engine_rules_fix.py` (named after the audit items, `test_b10_...`) and the
card-by-card suite `tests/cards/` (every expectation quoted from the card text, rulebook
or FAQ).

## Not implemented / simplified

| Item | What the engine does | Rule |
|---|---|---|
| Cuban Missile Crisis cancel timing | Offered at the start of the threatened player's action round and again before any coup they could be given (their own operations, an event's operations, a free coup), in either player's round. Not offered at other moments, where cancelling cannot change anything. | Card: "at any time"; FAQ #40 |
| "may" on beneficial effects | Some "may" / "up to" effects that can never hurt the player have no decline option: NORAD's and Puppet Governments' and Cambridge Five's influence, Chernobyl naming a region, an "as if N Ops" operations play. De-Stalinization, Pershing II, Wargames, The Reformer, Summit, Junta, Che, Ortega, Tear Down This Wall, SALT and Our Man in Tehran do offer the choice. | Card texts |
| Two VP awards of opposite sign in one event | The VP track is clipped at +/-20 at each award, so an event awarding both sides would be judged on the clipped running total. No card in the game awards both sides within one event (scoring cards are applied as a net difference), so this never happens. | R 10.2.3, 10.3.1 NOTE |
| Forced choices | A decision with exactly one country and no pass is auto-resolved; card choices (`choose_card`) are always asked, even with one option. | Engine simplification |
| `Game.clone()` | Replays the action history (live generator frames cannot be copied). Exact, O(game length). | Engine |
| Turn Zero, promo cards | Extracted into `twilight/data/cards.json` but excluded from play. Optional cards 104-110 are off unless `Game(optional_cards=True)`. | - |
| Cards in play are not in `state.removed` | A starred card that stays on the table (NATO, Formosan Resolution, The Iron Lady...) lives in `state.effects` until its effect ends; `test_every_card_is_always_accounted_for` checks every card is somewhere. | Engine convention |

## Rulings the engine follows

Sources: the 2nd-edition Deluxe rulebook (R), the designer-approved FAQ v5, the card
texts, and rulings given by the user on 2026-11-25 where the written sources could not
be found (the Deluxe rule 7.4.3 cited by the FAQ is not in any available copy).

- **"As if they played an N Ops card"** (CIA Created, Lone Gunman, ABM Treaty, Olympic
  Games boycott, Glasnost, KAL-007) is a normal operations play outside one's own round:
  the player's ops modifiers apply (Containment / Brezhnev to a maximum of 4, Red Scare
  to a minimum of 1), DEFCON geography applies, a coup counts toward military operations
  and degrades DEFCON in a battleground, and DEFCON 1 loses for the phasing player (the
  player of the card). Grain Sales' own operations, UN Intervention and a card taken by
  Missile Envy are normal plays too. *User ruling, 2026-11-25.*
- **Free coups** in a named region (Junta, Ortega Elected, Tear Down This Wall, Che)
  ignore DEFCON geography, still degrade DEFCON in a battleground, and do not count toward
  military operations. Che counts as free. *R 6.3.5, 8.2.5; FAQ #47, #96; Che: user
  ruling 2026-11-25.*
- **Nuclear Subs** spares US coups made with operations points (including "as if N Ops"
  operations), not free coups. *User ruling, 2026-11-25.*
- **DEFCON 1** always loses for the phasing player (the headline's player in the headline
  phase); the only exception is the Cuban Missile Crisis, which loses for whoever
  initiates the coup. *R 8.1.3, FAQ #40, #62, #67.*
- **Quagmire / Bear Trap**: a held card with *modified* ops >= 2 must be discarded (a
  compelled Missile Envy first); with no such card the trap stays and only a scoring card
  may be played; with as many scoring cards as rounds left a scoring card is played
  instead of the discard. *Card text, FAQ #42/#44, R 7.4.2; scoring-card exceptions: user
  ruling 2026-11-25.*
- **Missile Envy** takes the holder's highest *modified* ops card. *User ruling,
  2026-11-25.* A card carrying the taker's opponent's event is used for a normal
  operations play; the compulsion lasts until Missile Envy is spent and gives way to a
  scoring card that must be played (FAQ).
- **Extra action rounds** (Space Station, North Sea Oil: eight in all, not nine) may be
  forgone. *R 6.4.4; user ruling 2026-11-25.*
- **Vietnam Revolts'** +1 applies to any USSR operations play spent entirely in Southeast
  Asia, "as if N Ops" events included, not to event placements (Decolonization,
  De-Stalinization). Stacks with the China Card (R 7.4 example). *User ruling,
  2026-11-25.*
- **NORAD** fires after an action round in which DEFCON moved to 2 at any point; a drop
  in the headline phase does not count. *Card text; headline: user ruling 2026-11-25.*
- **Chernobyl** names one of the six scoring regions, not a sub-region. *User ruling,
  2026-11-25.*
- **Wargames** at DEFCON 2: the player chooses to give 6 VP and end the game, give 6 VP
  and go on, or neither; the card leaves the game in every case, and level VP after the
  6 is a draw. Above DEFCON 2 the event does not occur and the card is discarded, not
  removed. *FAQ #100, R 5.2; user ruling 2026-11-25.*
- **The Reformer** adds 4 influence, or 6 if the USSR leads on VP and chooses 6; either
  way the chosen amount is placed in full (up to 2 per country; if fewer legal points
  exist, placement stops when none is left). *User ruling, 2026-11-25.*
- **Star Wars** fetching a USSR event: the US chooses the card, the USSR resolves it,
  and the US (phasing) carries DEFCON blame. *User ruling, 2026-11-25.*
- **Grain Sales drawing Defectors** in the headline phase cancels the USSR headline if it
  has not resolved yet. *User ruling, 2026-11-25.*
- **The Reformer / Glasnost**: Glasnost does not remove The Reformer (card text).
- **We Will Bury You**: the 3 VP are scored the moment the US's next action round
  commits to anything but a UN Intervention event -- its card choice, or, when that card
  is UN Intervention / Missile Envy / Star Wars / Grain Sales (which can still fetch UN
  Intervention), the first event, card use or operations play inside the chain that is
  not one -- before any US VP. With no such moment the end of the round settles them.
  *Card text; FAQ #50.*
- **Olympic Games boycott** at DEFCON 2 loses for the sponsor (R 8.1.3 example).
- **Both players holding a scoring card** at turn end: the US wins (FAQ 8.5).
- **Final scoring**: 20 VP is not an automatic victory and the total is not clipped
  between regions; Europe control still wins outright. The China Card's turn-10 point is
  part of final scoring (R 10.3.2).

## Not verified / not covered

- **Space race box values.** The rulebook's text gives only Lunar Orbit (3/1) and the
  count of VP boxes (five); the track itself is a board graphic that is in neither
  available rulebook PDF. The engine's values (Deluxe: 2/1, ability, 2/0, ability, 3/1,
  ability, 4/2, 2/0; ops 2,2,2,2,3,3,3,4; success on 3,4,3,4,3,4,3,2) agree with those
  two facts and with the README's earlier cross-check, but box by box they are not
  verified against a primary source.
- **Untested boundaries**: no tests yet for Grain Sales drawing UN Intervention in the
  headline, or for an event awarding VP after a DEFCON drop to 1 in the same card.

## Verifying coverage yourself

```bash
# every card has a handler, and strict mode never trips
python -m pytest tests/test_engine.py -q -k "event_handler or strict_mode"

# the card-by-card suite and the audit regression tests
python -m pytest tests/cards tests/test_engine_rules_fix.py -q

# 400 games with full invariant checking, including card conservation
python tools/random_play.py --games 400
```
