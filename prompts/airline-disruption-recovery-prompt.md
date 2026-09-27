# Prompt for a fresh Claude Code session in the repository "airline-disruption-recovery"

GAUNTLET LOOP. Fan out sub-agents and ultracode: use sub-agents in parallel for research and critique, and use the Workflow tool for each phase once I approve the plan.

## Who I am and how I work

I am the product owner. Follow these rules in everything you produce.

1. Use the most appropriate skill for each step (for example deep-research, docx, dataviz, workflow-authoring, run).
2. Write every document I will read as a well-edited Word file, in smooth narrative prose, with no bullet points and never the em dash character. Tables are fine where they help comparison. This rule applies to Word documents; the README and the ledgers may use normal Markdown structure.
3. Tell me when you do not know something. Never invent facts, numbers, sources, data or results.
4. Write code in Python wherever possible. The website is TypeScript by choice, for Next.js and Three.js type safety. Write complete files, never snippets, and double-check code step by step with tests.
5. Never create dummy data or fictitious companies. Use real, live or archived public data, and label any reconstruction or modeled estimate as such, both in documents and on screen.
6. Use Claude models when an LLM is appropriate, and open-weight models only when nothing else will work.
7. Before you start building, ask me every question you need answered, and tell me which secrets you need (see "Accounts and secrets").

## The mission

Build a working, publicly demonstrable disruption recovery system that rebuilds the plan in minutes when storms, outages or incidents break a schedule. It has two tracks sharing one engine.

The demonstration track is Boston's MBTA rapid transit network, running on live public data, sized as a hackathon-grade proof for the Hack-Nation MIT hub. The business track is US airline disruption recovery, replayed on real historical disruptions from public data, as the case to show an airline partner.

The core idea is a hybrid. A generative model trained on past disruptions that recovered well learns what good recoveries look like for this operator, including tacit knowledge that pure optimization misses. It proposes recovery plans in seconds. A classical solver (Google OR-Tools CP-SAT) then enforces every hard rule and polishes the plan, so that nothing infeasible ever reaches an operator. A Claude-powered copilot explains each proposed plan in plain language, and a human always approves.

## What we already learned (prior evidence, not fact about your data)

In an earlier experiment on 300 real Amazon Last Mile routes, learning from past high-quality executions clearly helped. The best approach predicted high-level structure (the order of delivery zones) and let OR-Tools fill in the details. However, a supervised network of the same architecture trained without diffusion matched or beat the diffusion model and ran faster. A published critique on the traveling salesman problem (Xia et al., ICML 2024, "Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale TSP") also showed that a non-learned heatmap can nearly match learned diffusion heatmaps once search is added. So never assume diffusion is the winning ingredient. Every experiment must include a supervised control with the same architecture and a non-learned historical baseline, and the final report must say honestly which component delivers the gain.

## Data (real only)

For the MBTA track, use the MBTA V3 API (api-v3.mbta.com; the free key raises the limit from 20 to 1,000 requests per minute) and the GTFS-Realtime feeds on cdn.mbta.com (vehicle positions, trip updates and alerts, no key needed). Use the LAMP daily subway performance Parquet files (performancedata.mbta.com) for historical events and headways, the LAMP GTFS schedule archive for the schedule in force on each date, and the MBTA Open Data Portal for historical headways and ridership. Use its origin-destination and gated-entry ridership for passenger weights; never assume weights. Use V3 alerts with the effect SHUTTLE as direct evidence of bus bridges. Stay under 1,000 requests per minute and comply with the MassDOT Developers License Agreement, including attribution and no implied endorsement. Verify what each source actually contains before relying on it.

Start with the Red Line, the busiest line, which serves the Kendall/MIT station, has two southern branches (Ashmont and Braintree) that make short-turn and branch-balancing decisions meaningful, and has a long record of slow zones and disruptions. Add the Orange Line second as a test of transfer to another line. Leave the Green Line (light rail with street running and several branches) and buses for later. Walk me through getting an MBTA V3 API key when it is first needed, since I have not used the MBTA developer portal before.

Define a disruption episode as a period when observed headways on a line deviate badly from schedule. Define recovery actions that can be observed or inferred from the data, such as short-turns, holds, expressing, added trips and shuttle bus bridges. Label an episode as a success by an explicit, documented outcome measure, for example how quickly passenger-weighted headway regularity returned to schedule. Say plainly which actions are directly observed and which are inferred, because the MBTA does not publish dispatcher decisions as such.

For the airline track, use the Bureau of Transportation Statistics Reporting Carrier On-Time Performance data, which includes tail numbers, so that real aircraft rotations, cancellations and delays can be reconstructed. Use the FAA ATCSCC Advisories Database (fly.faa.gov/adv) for historical ground stops and ground delay programs, and NOAA or National Weather Service historical weather for context. ASPM needs an FAA account, so do not rely on it. Use the DB1B ten percent ticket sample (quarterly, through Q2 2025; its successor DB1C/OD40 is a monthly forty percent sample from July 2025) only for origin-destination and connection-share patterns. DB1B has no travel dates, flight numbers or times, so passenger flows onto specific flights must be modeled; label every passenger and misconnection figure as a modeled estimate and document the method.

Real crew assignments are not public. Reconstruct plausible pilot pairings from FAA Part 117 duty and rest rules, which cover flight crew only; if you model flight attendants, use 14 CFR 121.467 and say so. Label reconstructed crews everywhere as reconstructions, not airline records. Never use reconstructed crew pairings as training labels; use them only in the feasibility layer and in evaluation. The airline proposer learns from real aircraft rotations, cancellations and delays. Replay real events such as the Southwest Airlines meltdown of December 21 to 30, 2022 and the Delta disruption after the CrowdStrike outage of July 19, 2024.

Split every dataset by date, never at random, so that test disruptions come from days the models never saw. Store the final test split encrypted, or outside the repository on Lightning, so that only a single scripted run, which I trigger after the improvement loop ends, can read it. Before building models, count the labeled success episodes per line and per carrier and report the counts. If there are fewer than about 200 per model, tell me before training rather than padding with synthetic episodes.

Record the license of every dataset and of parzival-gauntlet in DATA_LICENSES.md. BTS and FAA data are public domain; MBTA data falls under the MassDOT agreement.

## The engine

Build a shared Python package with five parts.

1. A problem encoder turns a disrupted state (network, vehicles or aircraft, crews, passengers and the disruption) into a graph.
2. A generative proposer is a conditional discrete diffusion model over plan decisions (for example which trips to cancel, short-turn or hold, or which aircraft cover which flights). It is trained only on episodes labeled as successes and samples several candidates per disruption.
3. Two comparison methods: a supervised control uses the same graph network trained directly on the same labels without diffusion, and a historical baseline reuses the most similar past successful recovery with no learning.
4. A feasibility layer uses OR-Tools CP-SAT to repair and polish any candidate so that all hard rules hold: vehicle and crew availability, Part 117 legality on the airline side, minimum turnaround and layover times, and capacity. It also solves the problem from scratch as the pure-optimization baseline.
5. A frozen evaluator computes the metrics.

For transit the metrics are passenger-weighted excess wait time, headway regularity, time to recover and number of operator actions. For airlines they are flights cancelled, modeled passenger delay minutes and misconnections, crew and aircraft legality violations (which must be zero after repair) and time to produce a plan. Report means, medians and paired bootstrap 95 percent confidence intervals. Always report time to plan, because recovery speed was the binding failure in the real 2022 and 2024 meltdowns.

## The Gauntlet harness (reuse and adapt)

Adapt the Gauntlet protocol from github.com/alessoh/parzival-gauntlet. Ask me for read access if you cannot reach it.

- **Frozen evaluator.** The evaluator and data preparation are read-only to the improvement loop; only the model and solver code may be edited.
- **Noise calibration.** Calibrate noise with at least five baseline runs and record it in NOISE.md. Sigma is the standard deviation of the paired metric difference across those runs.
- **Keep gate.** Use keep_gate.py to decide each change:
  - keep it if its paired improvement exceeds two sigma;
  - discard it if it is worse by more than one sigma;
  - otherwise replicate it, at most twice, then discard it.
- **Critic veto.** Before each experiment a critic sub-agent may veto it on validity grounds only.
- **Ledgers.** Record each experiment's predicted result in predictions.tsv before running it, and append every failure pattern to gauntlet.md.
- **Anti-gaming checks.** Independently check the feasibility of every plan and recompute every metric from the raw plan, so no agent can game the score. Enforce time limits from outside the agent's code.

## The website (Vercel, Next.js, Three.js)

Build a Next.js application deployed on Vercel with a Three.js scene as its centerpiece. For transit, render the MBTA rapid transit network in 3D from real GTFS shapes, with live train positions from the realtime feed, disruptions highlighted as they are detected, and the proposed recovery shown as an animated before-and-after that the user can scrub through. For airlines, render a 3D view of the United States with real flight arcs for a replayed day, the disruption spreading through the network hour by hour, and the recovered schedule side by side with what actually happened. Show the metrics comparison for every method (diffusion, supervised control, historical baseline, pure OR-Tools and what actually happened) in clear charts. Include the Claude copilot panel that explains any plan and answers "why" questions, citing the constraints that drove each decision. Make the site responsive, accessible, fast on a normal laptop, and honest: every reconstructed or modeled number must be labeled on screen.

Respect Vercel's limits. Use serverless functions only for light work, respect function duration limits and the 4.5 MB response body limit, serve prebuilt geometry and replay data as static files, and poll the realtime feeds rather than holding streaming connections in functions. Call the Lightning service for inference and solving, and keep Lightning cold starts out of the live demo path (for example by precomputing the replayed airline days and warming the service). I have a paid Vercel plan, so commercial use is allowed; still keep within its function and bandwidth limits.

## Compute

Training and heavy solving run on my Lightning AI account. The GPU budget is a hard limit of one H100 for eight hours in total across the whole project, covering every training run, sweep and Gauntlet experiment. Do all development, debugging, tests and small experiments on CPU first, and use the H100 only for runs that have already worked end to end on CPU at small scale. Keep a GPU ledger in GPU_LEDGER.md recording the start time, end time and purpose of every GPU session, and stop the Studio's GPU as soon as each run ends so idle time is not billed. Plan the eight hours before starting, tell me the plan, and stop and ask me before any run that would push the total past eight hours. Serve the proposer and the OR-Tools solver as an HTTPS inference API from Lightning (for example with LitServe) running on CPU, not on the H100, so that serving never consumes the GPU budget; if CPU inference is too slow, precompute results instead of serving from the GPU.

## Accounts, secrets and safety

You cannot see the accounts open on my desktop. Ask me to add these as environment secrets before you need them, and never print, log or commit them:

- an MBTA V3 API key;
- a Lightning AI API key, my LIGHTNING_USER_ID, and the target Studio or teamspace name;
- a Vercel token or a connected Vercel account, with the project name;
- an Anthropic API key for the copilot.

Proxy every key through server functions and never ship one to the browser. Rate-limit the public copilot endpoint per IP address, cache its answers, and cap Anthropic spending per day at an amount I will give you. If anything else is required, ask.

## How to work (phases, each a workflow)

1. **Understanding and planning.** Fan out research agents to confirm what each data source actually contains, the state of the art in airline and transit disruption recovery (column generation, CP and MIP recovery models, learned recovery, and any diffusion or generative work), and prior art for this exact idea. Then write a design document in Word and ask me to approve it.
2. **Data.** Build and test the ingestion pipelines, the date-based splits and the episode counts.
3. **Engine.** Build the engine and baselines, with tests.
4. **Improvement loop.** Run the Gauntlet loop on the models.
5. **Website.** Build the website and deploy it.
6. **Report.** Write the final report.

For every deliverable (each research section, each pipeline, the engine, each model, the website and the report), run a build-and-critique loop:

- **The critic.** A separate, extremely harsh critic sub-agent grades the work AAA, AA, A, B or C against the state of the art for the stated scope.
- **Blind research grading.** The critic first gathers its own reference points from primary sources, then reads the draft, so the comparison is blind.
- **Blind model judging.** A blind judge ranks anonymized method results before they are unmasked.
- **Stopping rule.** Keep revising until the critic grades the work AAA, up to five rounds per deliverable and no more than 40 critic rounds across the whole project. If a deliverable still falls short, stop, record exactly what the critic still objects to, and report it honestly rather than claiming AAA.

Commit and push after each phase to the branch I name in my answers (ask if I have not named one). Log what any cap or sampling dropped; never truncate silently.

## What done means

Done means all of the following:

- the Vercel URL loads, shows live MBTA train positions, and replays at least one real airline event with recovered plans;
- the Lightning service answers requests;
- all tests pass in CI;
- the Word report is delivered.

## Deliverables

- the working Vercel URL;
- the Lightning inference service;
- the complete Python package, with tests and a README that reproduces every result;
- the Gauntlet ledgers (NOISE.md, predictions.tsv, gauntlet.md and results.tsv);
- DATA_LICENSES.md;
- a Word design document;
- a final Word report.

The report must give the honest verdict on whether the generative proposer beats the supervised control, the historical baseline and pure OR-Tools on held-out real disruptions, with confidence intervals and the time to plan for each method.

## Before you begin

Read this whole prompt, then ask me your questions in one batch. Include anything ambiguous about:

- success labels;
- which airline and which historical events to replay first;
- the Anthropic spending cap and the GPU budget;
- the git branch.

Do not start building until I answer.
