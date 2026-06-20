# Simulated Customer Interview Notes — Cook Scheduling
**Source type:** Simulated interview vignettes  
**Purpose:** Provides a human-grounded source for `data/llm_eval_set_v0.1.json` examples (eval IDs V01–V15).  
**Methodology:** Vignettes are constructed to reflect realistic associate and operations manager decision-making patterns observed in convenience store hot food operations. Each vignette implies a specific cook-order scenario documented in the eval set.  

---

## Associate Perspective (Vignettes V01–V10)

### V01 — Urban store, Wednesday morning, 6:00 AM (4 items)
*"I always start with pizza on weekday mornings. By the time it's done, people are already coming in for their coffee. Wings I'll do second — the short-hold ones first because they go bad in two hours and there's nothing worse than tossing a full pan. Baked goods I leave for last. They don't go bad until like 10 PM, so there's no rush."*

**Implied scenario:** Urban store, 6 AM, 4 items present (pizza, wings_2h, wings_4h, baked_goods).  
**Expected ranking signal:** wings_2h/pizza urgency ahead of wings_4h; baked_goods last due to 24hr hold.  

---

### V02 — Highway store, Friday morning, 7:00 AM (3 items)
*"Truckers and commuters hit us hard between 6 and 9. Wings sell out fast here — faster than in the city stores I used to work at. So on a busy morning I'll start the wings first, especially the two-hour kind because if they sit more than two hours on the warmer they're done. Pizza comes next. Baked goods I barely touch until after rush."*

**Implied scenario:** Highway store, 7 AM, 3 items (wings_2h, pizza, baked_goods). High wings demand.  
**Expected ranking signal:** wings_2h first due to fast sell-through + short hold; pizza second; baked_goods last.  

---

### V03 — Suburban store, Saturday morning, 8:00 AM (3 items)
*"Weekends are different. Families come in, kids want baked goods — the mini muffins, the little pastries. But they don't expire fast so I still start with whatever has the shorter window. Usually that's wings or pizza. On Saturday mornings it's a toss-up but I lean pizza because families buy slices."*

**Implied scenario:** Suburban store, 8 AM weekend, 3 items (pizza, wings_2h, baked_goods).  
**Expected ranking signal:** pizza first (weekend demand + 2hr hold); wings_2h second; baked_goods last.  

---

### V04 — Urban store, Monday lunch, 12:00 PM (4 items)
*"Lunch rush downtown is pure pizza. I've never had leftover pizza at noon on a Monday. Wings are second — we get a lot of people grabbing wings with their drink. The four-hour wings can stay warm longer so I prioritize the two-hour ones. Baked goods at noon on a Monday? Nobody wants a muffin for lunch, but I still have to have them out."*

**Implied scenario:** Urban store, 12 PM weekday, 4 items. Pizza very high demand at lunch.  
**Expected ranking signal:** pizza first (demand spike + urgency); wings_2h before wings_4h; baked_goods last.  

---

### V05 — Highway store, Tuesday evening, 6:00 PM (3 items)
*"Evening shift is all about wings. People coming home from work grab wings, a drink, done. I always fire the two-hour wings first because they'll expire before the end of my shift if I'm not careful. Pizza is steady but slower in the evenings out here. I've almost never had to throw away baked goods — they just keep going."*

**Implied scenario:** Highway store, 18:00, 3 items (wings_2h, pizza, baked_goods). Evening shift.  
**Expected ranking signal:** wings_2h first; pizza second; baked_goods last.  

---

### V06 — Urban store, Thursday morning, 6:00 AM — baked goods demand spike (4 items)
*"We had this catering pre-order once for like 40 pastries for a morning meeting. I had to bump baked goods up that day even though normally they're last. The thing is baked goods cook in 10 minutes and I needed them out early. But I still did wings first because that window was closing. The system kept telling me to do baked goods and I kept overriding it — that felt wrong."*

**Implied scenario:** Urban store, 6 AM, 4 items. baked_goods forecast_demand ≥ 30 (spike). Known v1 failure mode where demand_density for baked_goods (30/1=30) dominates.  
**Expected ranking signal:** wings_2h/pizza should still precede baked_goods despite demand spike — expiry window beats demand volume.  
**Eval tag:** edge_case, divergence (v1 ranks baked_goods first; domain-expert label ranks it last).  

---

### V07 — Suburban store, Wednesday afternoon, 2:00 PM (2 items)
*"Mid-afternoon is quiet. I usually just have two things going — pizza and the four-hour wings. Pizza I'll do first because it only lasts two hours and if I wait too long I'm throwing it out before the dinner crowd. Wings are fine waiting, they've got four hours once they're done."*

**Implied scenario:** Suburban store, 14:00, 2 items (pizza, wings_4h). Quiet period.  
**Expected ranking signal:** pizza first (2hr hold vs 4hr hold; same remaining window means pizza more urgent).  

---

### V08 — Urban store, Friday noon, 12:00 PM — near-expiry wings (3 items)
*"It was a Friday and we were slammed. I noticed the wings_2h that were cooking had about 20 minutes left on the warmer. I threw fresh ones in immediately — even though pizza demand was through the roof. You never let wings expire. The waste is brutal and the smell is bad."*

**Implied scenario:** Urban store, 12 PM Friday, 3 items. wings_2h time_remaining ≤ 0.5hr.  
**Expected ranking signal:** wings_2h first regardless of pizza demand — near-expiry override.  
**Eval tag:** edge_case.  

---

### V09 — Highway store, Sunday morning, 9:00 AM — zero demand item (4 items)
*"Sunday morning one time I had baked goods on the list but we'd already sold a ton and my manager said we were cutting it from the morning rotation. Demand showed zero. I just cooked everything else — pizza, both wing types — and left baked goods for later. Zero demand means there's no point rushing it."*

**Implied scenario:** Highway store, 9 AM Sunday, 4 items, baked_goods forecast_demand = 0.  
**Expected ranking signal:** baked_goods ranked last when forecast_demand = 0.  
**Eval tag:** edge_case.  

---

### V10 — Suburban store, Saturday evening, 7:00 PM — wings_4h vs wings_2h same demand (3 items)
*"Sometimes I have both wing types up with the same forecast. I always do the two-hour ones first — they're done in the same time as the four-hour ones but go bad twice as fast. It's just common sense. If they both sell the same, the one that expires faster goes first."*

**Implied scenario:** Suburban store, 19:00, 3 items (wings_2h, wings_4h, pizza). wings_2h and wings_4h same forecast_demand.  
**Expected ranking signal:** wings_2h before wings_4h due to shorter hold; pizza by demand/time.  
**Eval tag:** edge_case (hold-time tie-break).  

---

## Operations Manager Perspective (Vignettes V11–V15)

### V11 — Urban district, weekday morning pattern (4 items)
*"Our urban stores downtown have a really predictable pattern. 6 to 9 AM, pizza and wings move fast. We're talking 15 to 20 units of pizza by 9 AM. My recommendation to associates is always: wings_2h first at 6 AM because they expire fast, then pizza, then wings_4h, baked goods whenever. If they do it in that order they almost never have waste."*

**Implied scenario:** Urban store, 6 AM, 4 items. High pizza + wings_2h demand pattern.  
**Expected ranking signal:** wings_2h first (expiry urgency); pizza second (high demand, 2hr hold); wings_4h third; baked_goods last.  

---

### V12 — Highway district, evening waste problem (3 items)
*"We had a waste issue at our highway locations in the 5–8 PM window. Associates were cooking baked goods late in the evening and they'd sit all night. The fix was simple: prioritize wings and pizza in the evening, leave baked goods for the morning rotation. Evening expiry windows for wings are tight and that's where the waste was coming from."*

**Implied scenario:** Highway store, 17:00–20:00 range, 3 items (wings_2h, pizza, baked_goods). Evening.  
**Expected ranking signal:** wings_2h and pizza ahead of baked_goods in evening shift.  

---

### V13 — Suburban district, weekend vs weekday (4 items, morning)
*"Suburban stores on weekends behave differently. Baked goods sell almost 40% faster on Saturday mornings than weekdays — families, kids. But even so, I tell my associates not to let that flip their cook order. Wings still expire faster than baked goods. Expiry window is the rule; demand is a tiebreaker, not the main factor."*

**Implied scenario:** Suburban store, Saturday 8 AM, 4 items. baked_goods demand elevated but baked_goods hold = 24hr.  
**Expected ranking signal:** Expiry-driven items (wings_2h) first; baked_goods still last despite high demand.  

---

### V14 — Urban district, high-demand low-hold vs low-demand high-hold (4 items, afternoon)
*"I use this scenario to train new managers: you've got wings_2h with 10 units forecast and a 2-hour hold, versus baked_goods with 25 units forecast and a 24-hour hold. Which do you cook first? Every new manager says baked goods because the number is bigger. Wrong. Wings expire 12 times faster. The smaller number with the shorter window always wins."*

**Implied scenario:** Urban store, afternoon, 4 items. High baked_goods demand (25 units, 24hr hold) vs wings_2h (10 units, 2hr hold).  
**Expected ranking signal:** wings_2h before baked_goods — perishability beats raw demand volume.  
**Eval tag:** edge_case (high-demand long-hold vs low-demand short-hold).  

---

### V15 — Highway district, early morning 2-item decision (2 items)
*"At 6 AM, if it's just pizza and wings_2h on the board — which it often is at highway locations — there's no debate. Both have a 2-hour hold. But wings_2h time-remaining is shorter because the window closes at 8 and pizza closes at 8:30. So wings go first. Half an hour difference but it matters when the trucker crowd hits at 7:30."*

**Implied scenario:** Highway store, 6 AM, 2 items (wings_2h, pizza). wings_2h time_remaining < pizza time_remaining, both 2hr hold.  
**Expected ranking signal:** wings_2h first; pizza second.  
