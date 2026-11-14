# Decisions and deviations log (PROJECT_NEXT_PHASE.md, Section 0, item 3)

Every change to a pre-registered threshold, recipe, run list or data split after PREREG_NEXT_PHASE.md is committed is
logged here, newest last, with the date (UTC), the reason and its effect. The paper reports every entry.

| Date | Decision or deviation | Reason | Effect | Approved by |
|---|---|---|---|---|
| 2026-11-14 | PROJECT_NEXT_PHASE.md adopted as the plan for the remaining work; it supersedes PAPER_EXPERIMENTS.md and the old job lists for what runs next | Dawar's instruction | E3 (PREREG_E3.md) to be superseded by MUST-09, pending sign-off (PREREG_NEXT_PHASE.md Section 9, item 6) | Dawar |
| 2026-11-14 | PREREG_NEXT_PHASE.md signed off, with all 11 choices of its Section 9 as drafted | Dawar's reply "agreed" | Thresholds fixed from this commit (gate A met once MUST-00 is accepted) | Dawar |
| 2026-11-14 | GPU run moved to the morning of 15 November; H200 instance 574587 (Jarvis Labs) created, set up and paused after about 25 minutes (code, venv, structure files and Tier-2 caches kept under /home; ImageNet download to redo) | Dawar's request | About 1 hour of data build tomorrow before the first jobs | Dawar |
| 2026-11-14 | H200 instance 574587 deleted by Dawar after the pause (storage billing); setup to be redone on a new instance on 15 November | Dawar's choice | About 10 more minutes of setup | Dawar |
