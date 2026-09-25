# Physical AI and Autonomous Agents in Logistics: State and Missing Capabilities (2025-2026)

Method note: researched 2026-09-25 through web search. Direct fetches of ir.aurora.tech, spectrum.ieee.org and gartner.com were blocked by the network proxy, so several findings come from search-result summaries of those pages, not full-text reads. Figures from companies are marked **(self-reported)**.

## 1. Warehouse robotics: what limits general-purpose picking and manipulation?

### Takeaway
Robots now handle most SKUs at roughly human speed in narrow, structured tasks: Amazon Vulcan covers about 75% of items, Stretch unloads up to 700 cases/hr, and Digit moves totes. The remaining gap is the last 20-35% of items (deformable, loose or complex packaging) plus very high reliability at fleet scale. Commercial humanoids still do only simple tote moves. The research bottlenecks are data scarcity, the sim-to-real gap, embodiment heterogeneity and tactile/force sensing.

### Cited Findings
**Amazon Sparrow / Vulcan / Covariant**
- Sparrow (announced Nov 2022) could pick about 65% of Amazon's items. It struggled with items in loose or complex packaging. **(self-reported)** — [The Robot Report](https://www.therobotreport.com/meet-sparrow-amazons-new-item-picking-robot/)
- Vulcan (announced May 2025) is Amazon's "first robot with a sense of touch". It uses force sensing to stow into and pick from fabric pods. **(self-reported)** — [About Amazon](https://www.aboutamazon.com/news/operations/amazon-vulcan-robot-pick-stow-touch); [The Robot Report](https://www.therobotreport.com/amazon-vulcan-robot-uses-force-sensing-to-stow-items/); technical paper: [Amazon Science, "Vulcan Pick"](https://www.amazon.science/publications/vulcan-pick-a-robotic-system-for-picking-targeted-objects-from-fabric-pods)
- Vulcan can pick and stow about 75% of item types at speeds comparable to human workers. Amazon's target is to stow 80% of items at 300 items/hour over 20 hours/day. **(self-reported)** — [IEEE Spectrum](https://spectrum.ieee.org/amazon-robotics-vulcan-warehouse-picking) (seen via search summary; full text blocked)
- IEEE Spectrum reports that Amazon's next-year focus for Vulcan is reliability ("getting better at not screwing up"). Small error rates multiplied across huge deployments are unacceptable. — [IEEE Spectrum](https://spectrum.ieee.org/amazon-robotics-vulcan-warehouse-picking)
- In Aug 2024 Amazon took a non-exclusive license to Covariant's robot foundation models (RFM-1, announced March 2024). It hired Covariant's founders (Abbeel, Chen, Duan) and about 25% of staff, a "reverse acquihire". — [About Amazon](https://www.aboutamazon.com/news/company-news/amazon-covariant-ai-robots); [GeekWire](https://www.geekwire.com/2024/amazon-hires-covariant-founders-inks-licensing-deal-with-robotics-ai-startup-in-latest-reverse-acquihire-deal/)
- A 2025 whistleblower complaint to the FTC, SEC and DOJ reportedly put the price at $380M plus a $20M final licensing payment. This is an allegation, not confirmed by Amazon. — [Wikipedia summary](https://en.wikipedia.org/wiki/Covariant_(company)); [Hard Reset Media](https://www.hardresetmedia.com/p/whistleblower-ftc-complaint-about-amazon-covariant)

**Symbotic / Walmart**
- Jan 16, 2025: Symbotic agreed to acquire Walmart's Advanced Systems and Robotics business (closed Jan 28, 2025). Terms were $200M cash plus up to $350M contingent. Walmart will pay $520M for development. Walmart committed to up to 400 store-attached Accelerated Pickup and Delivery (APD) systems if performance criteria are met. The deal could add more than $5B to Symbotic's backlog. — [Symbotic 8-K (SEC)](https://www.sec.gov/Archives/edgar/data/1837240/000119312525007370/d916752dex991.htm); [Digital Commerce 360](https://www.digitalcommerce360.com/2025/01/21/walmart-deal-symbotic-to-acquire-robotics-business/)
- The rollout depends on "performance criteria". The contract structure makes proven throughput and reliability a condition for scale. — [Symbotic 8-K](https://www.sec.gov/Archives/edgar/data/1837240/000119312525007370/d916752dex991.htm)

**Boston Dynamics Stretch**
- May 13, 2025: DHL signed an MOU for more than 1,000 additional Stretch units. Stretch deployments reach up to 700 cases/hour for trailer unloading **(self-reported)**. DHL plans to extend Stretch to case picking. — [DHL Group press release](https://group.dhl.com/en/media-relations/press-releases/2025/dhl-group-signs-mou-with-boston-dynamics-and-accelerates-cross-business-automation-strategy.html); [The Robot Report](https://www.therobotreport.com/dhl-buying-1000-stretch-robots-from-boston-dynamics/)
- An MOU is not a firm order. The gap between a 1,000-unit MOU and the installed base is large; one secondary source cites only about 10 units across 3 countries as of mid-2025 (this figure needs verification). — [Automated Warehouse](https://www.automatedwarehouseonline.com/from-case-unloading-picking-beyond-dhl-boston-dynamics/)

**Humanoids (Agility, Figure)**
- Agility Digit at GXO (Flowery Branch, GA) was the industry's first formal commercial humanoid deployment and its first humanoid RaaS contract (2024). Digit has since moved more than 100,000 totes. **(self-reported)** — [Agility: GXO RaaS](https://www.agilityrobotics.com/content/digit-deployed-at-gxo-in-historic-humanoid-raas-agreement); [Agility: 100k totes](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes)
- Other Digit deployments: Mercado Libre (San Antonio) and Toyota Motor Manufacturing Canada (7 Digits, Feb 2026) — [Agility Robotics Wikipedia](https://en.wikipedia.org/wiki/Agility_Robotics). A "nine customer facilities" figure attributed to Agility's 2026 SEC filings comes from a secondary summary and is unverified.
- Digit's task is tote transport with about 35 lb payload. This is repetitive material movement, not dexterous piece-picking. — [RoboZaps review](https://blog.robozaps.com/b/agility-robotics-digit-review) (secondary)
- I found no independent throughput, uptime or cost-per-task figures for humanoids in logistics.

**Robot foundation models / research state**
- General-purpose robot models such as π0 (Physical Intelligence), NVIDIA GR00T, Gemini Robotics and Figure Helix arrived in quick succession in 2025. The main VLA bottlenecks remain data scarcity, embodiment heterogeneity and the sim-to-real gap. — [Pebblous VLA explainer (2026)](https://blog.pebblous.ai/project/PhysicalAI/physical-ai/en/); survey: [arXiv 2505.04769](https://arxiv.org/html/2505.04769v1)
- NVIDIA is pushing Cosmos world models for synthetic data and Isaac GR00T open models (2026). — [NVIDIA blog, National Robotics Week 2026](https://blogs.nvidia.com/blog/national-robotics-week-2026/)
- Benchmarks are still immature. REALM (Dec 2025) builds a real-to-sim-validated benchmark for manipulation generalization because sim results often fail to predict real performance. — [arXiv 2512.19562](https://arxiv.org/pdf/2512.19562)
- A Samsung/Rainbow Robotics dual-arm robot ran a proof-of-concept at a Coupang logistics center. Soft-body parcel picking from filled containers is a target scenario. — [Pebblous](https://blog.pebblous.ai/project/PhysicalAI/physical-ai/en/)

### Inferences
- The binding constraint has shifted from average speed to tail reliability and SKU coverage. Robots match humans at the median but fail on the long tail (deformables, polybags, tangled or loose packaging, tight fabric pods).
- Needed capabilities:
  - tactile/force-aware policies (Vulcan points this way)
  - failure detection and self-recovery
  - fleet-scale learning from deployment data
  - validated sim-to-real evaluation so labs can certify reliability before deployment
- Big players are buying talent and models rather than startups building standalone businesses (the Covariant pattern). General-purpose picking is capital-intensive and consolidating around integrators with large data flywheels (Amazon, Walmart/Symbotic).
- Humanoids have commercial traction only for simple tote logistics. Their economic case against wheeled mobile manipulators and fixed cells is unproven publicly.

### Gaps
- No independent (non-vendor) data on mean picks between failure, intervention rates or cost per pick for Vulcan, Sparrow, Covariant or Ocado.
- Ocado 2025-2026 status was not found in my searches: partner closures, the Kroger CFC changes and robotic-pick (OGRP) performance. This needs a dedicated search.
- Figure AI logistics deployments (BMW, parcel-sorting demos) and Figure's performance claims were not verified.
- Humanoid safety standards (e.g., ISO work on dynamically stable mobile robots) were not researched in depth. A DC Velocity piece on this exists but was not read: [DC Velocity](https://www.dcvelocity.com/material-handling/robots-still-making-strides-in-logistics).

## 2. Autonomous trucking and last-mile delivery: status and remaining barriers

### Takeaway
Driverless trucking went commercial in Texas in 2025, led by Aurora with about 20 trucks and 10 routes. It is still tiny relative to trucking, heavily loss-making, and depends on federal waivers. Drone delivery is scaling fastest through Walmart+Wing (1M+ deliveries), but Amazon Prime Air shows repeated safety incidents, and the FAA's Part 108 BVLOS rule is still unpublished as of Sept 2026. Sidewalk robots are growing but still earn small revenue.

### Cited Findings
**Aurora**
- Commercial driverless operations launched Houston-Dallas on I-45 (April/May 2025). Later expansions: Fort Worth-El Paso on I-20, and a validated ~1,000-mile Fort Worth-Phoenix lane. A later release says Aurora "tripled" its driverless network to 10 routes and is preparing Sun Belt expansion. **(self-reported)** — [Aurora IR: Texas launch](https://ir.aurora.tech/news-events/press-releases/detail/119/aurora-begins-commercial-driverless-trucking-in-texas-ushering-in-a-new-era-of-freight); [Aurora IR: El Paso](https://ir.aurora.tech/news-events/press-releases/detail/128/aurora-expands-driverless-trucking-service-from-fort-worth-to-el-paso); [Aurora IR: 10 routes](https://ir.aurora.tech/news-events/press-releases/detail/132/aurora-triples-driverless-network-to-10-routes-and-prepares-to-expand-across-u-s-sun-belt); [HDT: Fort Worth-Phoenix](https://www.truckinginfo.com/news/aurora-adds-1000-mile-driverless-run-from-fort-worth-to-phoenix)
- More than 250,000 driverless miles by Jan 2026 with zero Aurora Driver-attributed collisions **(self-reported)**. Targets are 200+ driverless trucks by end-2026 and a next-gen hardware kit on International LT trucks without an observer in Q2 2026. — [National Today/Dallas (Mar 5, 2026)](https://nationaltoday.com/us/tx/dallas/news/2026/03/05/aurora-innovation-touts-250k-incident-free-driverless-miles-targets-200-trucks-in-2026/); [Startup Fortune](https://startupfortune.com/aurora-innovation-races-toward-200-driverless-trucks-by-year-end/)
- About 20 driverless trucks were operating as of Sept 2026 against a goal of 200 by year end. Aurora lost more than $800M in 2025 and targets positive free cash flow by 2028. — [Breitbart (Sept 23, 2026)](https://www.breitbart.com/tech/2026/09/23/aurora-tests-autonomous-semi-truck-in-texas-aims-to-replace-human-drivers/). This is a lower-quality outlet; the numbers need cross-checking against Aurora's 10-Q.
- Regulatory: FMCSA rules require drivers to place reflective warning triangles when a truck stops. Aurora sued FMCSA; FMCSA then granted a waiver (Oct 10, 2025 - Jan 9, 2026) allowing cab-mounted warning beacons. Aurora reports 34 trucks drove more than 500,000 miles under the waiver, with beacons active about 10 hours in total and no roadside collisions **(self-reported)**. In April 2026 Aurora applied for a 5-year exemption (comments due May 15, 2026); the extended waiver ran to July 9, 2026. — [FMCSA waiver terms (PDF)](https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/2025-10/Letter%20to%20Aurora%20Operations,%20Inc.%20-Waiver%20of%20Warning%20Device%20Requirements%20Terms%20and%20Conditions.pdf); [Federal Register, Apr 15, 2026](https://www.federalregister.gov/documents/2026/04/15/2026-07288/parts-and-accessories-necessary-for-safe-operation-application-for-exemption-from-aurora-operations); [CCJ](https://www.ccjdigital.com/equipment-controls/article/15769994/aurora-receives-warningdevice-waiver-from-fmcsa)

**Kodiak, Waabi, Plus, Gatik**
- Kodiak says its driverless long-haul safety case is 93% complete and it is on track for commercial driverless highway operations by end-2026 **(self-reported)**. Its off-road Permian Basin driverless fleet (Atlas) is expanding from 28 to 100 trucks, with public-highway driving expected in early 2027. — [HDT](https://www.truckinginfo.com/news/kodiak-says-driverless-long-haul-launch-on-track-for-2026); [Fox News](https://www.foxnews.com/tech/100-kodiak-driverless-trucks-headed-public-roads); [Kodiak 2025 review](https://kodiak.ai/news/best-of-2025)
- Waabi partnered with Volvo (Feb 2025) and is reported to have raised $750M. It develops a generative-AI/simulation-first stack. — [TechCrunch](https://techcrunch.com/2025/02/04/waabi-and-volvo-team-up-to-build-self-driving-trucks-at-scale); [The Curbivore](https://www.thecurbivore.com/p/autonomous-trucking-shifts-into-high) (secondary)
- PlusAI is going public via SPAC and reportedly raised about $300M. — [The Curbivore](https://www.thecurbivore.com/p/autonomous-trucking-shifts-into-high) (secondary)
- Gatik raised a $200M Series D and targets more than 100 trucks by end-2026. It runs driverless middle-mile box trucks on fixed routes in DFW, Phoenix, NW Arkansas and Toronto for Walmart, Kroger, Tyson, Loblaw and Frito-Lay. — [Sourcing Journal/WWD](https://wwd.com/sourcing-journal/logistics/gatik-200-million-series-d-funding-ai-driverless-autonomous-trucks-trucking-1239164645/)

**Drones**
- FAA Part 108 (BVLOS): the NPRM was published Aug 7, 2025, with about 4,000 comments. It missed the statutory deadline of Jan 16, 2026 (FAA Reauthorization Act 2024) and the EO 14307 deadline (about Feb 1, 2026). The final rule has been at White House review since July 10, 2026, and was unpublished as of Sept 18, 2026. — [The Flight Brief (Sept 2026)](https://www.theflightbrief.com/articles/faa-part-108-bvlos-update-september-2026); [Airdata](https://airdata.com/blog/2026/part-108)
- Walmart reached 1 million drone deliveries (May 29, 2026). Walmart+Wing are expanding to 150 more stores (announced Jan 2026; 40M+ potential customers) and more than 270 locations in 2027, including new metros (Memphis, New Orleans, Philadelphia, Phoenix, San Diego, SF Bay Area, Salt Lake City). **(self-reported)** — [Walmart corporate](https://corporate.walmart.com/news/2026/05/29/walmart-celebrates-1-million-drone-deliveries-marking-a-major-milestone-in-customer-convenience); [Wing](https://wing.com/news/wing-walmart-expand-drone-delivery-coast-to-coast); [Supply Chain Dive](https://www.supplychaindive.com/news/walmart-wing-drone-delivery-coverage-expansion/809326/)
- Amazon Prime Air had a series of problems:
  - US pause in Jan 2025 after Dec 2024 test incidents (software/light rain)
  - Oct 2025: two MK30s hit the same crane in Tolleson, AZ minutes apart, prompting FAA/NTSB probes and a suspension
  - Nov 2025: a drone clipped a cable in Waco, TX
  - Feb 2026: a drone hit an apartment building in Richardson, TX
  - a UK crash (Darlington) in 2026

  Sources: [CNN](https://www.cnn.com/2025/10/02/us/arizona-amazon-drones-crash); [CNBC](https://www.cnbc.com/2025/10/02/amazon-drone-crash-faa-ntsb.html); [DroneXL](https://dronexl.co/2026/04/02/amazon-prime-air-mk30-drone-2/); [TechTimes, Jul 30, 2026](https://www.techtimes.com/articles/322273/20260730/amazon-kept-flying-after-arizona-waco-richardson-darlington-uk-regulators-call.htm)

**Sidewalk robots**
- Serve Robotics reports about 2,000 sidewalk robots, the largest US commercial fleet. FY2025 revenue was $2.7M, Q2 2026 revenue $3.2M, and its 2026 outlook is about $26M. It holds more than $240M in cash. **(self-reported)** — [Serve Q2 2026 results](https://investors.serverobotics.com/news-releases/news-release-details/serve-robotics-announces-second-quarter-2026-results); [SEC 8-K](https://www.sec.gov/Archives/edgar/data/0001832483/000183248326000034/serv-20260630_ex991earning.htm)
- Starship raised $50M to scale in the US. — [Restaurant Dive](https://www.restaurantdive.com/news/starship-technologies-raises-50-million-us-robotics-delivery/802863/)

### Inferences
- **Technical barriers:**
  - Long-tail perception around obstacles like cranes and cables. Amazon's crane and cable strikes point to obstacle detection in dynamic urban environments.
  - Weather and night operation for trucks (not verified here).
  - Hardware redundancy at automotive-grade scale, which is why Aurora is waiting on next-gen hardware.
  - Autonomous handling of roadside stops, inspections and incidents (the warning-triangle issue is an example).
- **Regulatory barriers:**
  - No federal AV framework for trucks; operation relies on case-by-case waivers and exemptions.
  - Part 108 is delayed.
  - A state patchwork, with Texas favorable.
- **Economic barriers:**
  - Capex and losses are large relative to revenue (Aurora more than $800M lost in 2025; Serve about $26M projected revenue).
  - Scale is limited: about 20 trucks vs. roughly 200 targeted.
  - Hub-to-hub models still need human drayage and terminals.
- Safety cases are the gating artifact. Kodiak's "93% complete safety case" and Aurora's safety-case framing show that there is no regulator-accepted standard for "safe enough".

### Gaps
- Zipline 2025-2026 delivery counts and state expansion were not verified with primary sources.
- Plus/IVECO/International and Waabi driverless launch dates were not confirmed.
- NHTSA/FMCSA broader AV rulemaking status in 2026 (e.g., an AV framework or FMVSS exemptions) was not researched.
- Independent crash or disengagement data for trucks were not found. All safety numbers are company-reported.

## 3. Agentic AI in supply chain: launches, evidence, failure modes

### Takeaway
Every major SCM vendor shipped "agents" in 2025-2026: SAP Joule, Blue Yonder, Kinaxis Maestro and o9. Evidence of results is mostly vendor claims. Gartner is sharply skeptical. It predicts that more than 40% of agentic AI projects will be canceled by end-2027 and that only 5% of organizations will make at least 10% of planning decisions autonomously by 2030. It also warns of "agent washing" and restricts autonomy to low-risk decisions.

### Cited Findings
- **SAP:** Joule now has 40+ agents and about 2,400 skills, and Joule Studio is GA (Q1 2026). Agents can release production orders and rebalance inventory. A "Supply Chain Orchestration" product uses internal and external signals. **(vendor claims via secondary summaries)** — [SAVIC Technologies](https://www.savictech.com/insights/sap-joule-agentic-platform-40-agents-2026/); [WWD](https://wwd.com/business-news/technology/agentic-artificial-intelligence-retail-chatgpt-amazon-sap-blue-yonder-google-1238432952/)
- **Blue Yonder:** expanded agentic AI across supply chain execution (Mar 12, 2026). Its agents include "self-healing" master data that auto-corrects lead times and safety stock, claiming "up to 80%" less manual planner workload. **(self-reported)** — [Logistics Viewpoints](https://logisticsviewpoints.com/2026/03/12/blue-yonder-expands-agentic-ai-and-mobile-experiences-for-industry-specific-supply-chain-execution/); [Tellius](https://www.tellius.com/resources/blog/agentic-ai-in-supply-chain-use-cases-platforms-and-whats-shipping-2026)
- **Kinaxis:** Maestro Agents launched Oct 2025 inside concurrent planning, with human-in-the-loop guardrails. A no-code Agent Studio and marketplace are due in 2026. — [Tellius](https://www.tellius.com/resources/blog/agentic-ai-in-supply-chain-use-cases-platforms-and-whats-shipping-2026)
- **o9:** composite cross-functional agents on its Enterprise Knowledge Graph. — [Tellius](https://www.tellius.com/resources/blog/agentic-ai-in-supply-chain-use-cases-platforms-and-whats-shipping-2026)
- **Oracle** positions "Oracle AI for SCM" against SAP, Kinaxis, Blue Yonder and others. This is a marketing comparison page. — [Oracle](https://www.oracle.com/scm/oracle-vs-competition/)
- **Gartner (Jun 25, 2025):** more than 40% of agentic AI projects will be canceled by end-2027, due to escalating costs, unclear business value or inadequate risk controls. Current models lack maturity to autonomously achieve complex goals or follow nuanced instructions over time. Vendors are "agent washing", and only about 130 of thousands of "agentic" vendors are real. — [Gartner press release](https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027)
- **Gartner (May 21, 2025):** by 2030, 50% of cross-functional SCM solutions will include agents that autonomously execute decisions. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-05-21-gartner-predicts-half-of-supply-chain-management-solutions-will-include-agentic-ai-capabilities-by-2030)
- **Gartner (Mar 18, 2026):** 60% of supply chain disruptions will be resolved without human intervention by 2031. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-03-18-gartner-predicts-60-percent-of-supply-chain-disruptions-will-be-resolved-without-human-intervention-by-2031)
- **Gartner (Apr 7, 2026):** spend on SCM software with agentic AI will grow to $53B by 2030. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-04-07-gartner-forecasts-supply-chain-management-software-with-agentic-ai-will-grow-to-53-billion-in-spend-by-2030)
- **Gartner (Jun 30, 2026)** named agentic AI and physical AI top supply chain technology trends for 2026. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-06-30-gartner-identifies-top-supply-chain-technology-trends-for-2026); [Supply Chain Digital](https://supplychaindigital.com/news/gartner-agentic-physical-ai-top-2026-supply-chain-trends)
- **Gartner (Sept 24, 2026):** only 5% of organizations implementing planning automation will make at least 10% of planning decisions autonomously by 2030. Gartner cites technological immaturity and data availability, which restrict autonomy to low-risk decisions. It calls for a "decision stack" of data, workflows, governance and human context, with mapped critical decisions and guardrails. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-09-24-gartner-predicts-only-5-percent-of-organizations-will-make-at-least-10-percent-of-supply-chain-planning-decisions-autonomously-by-2030); [Gartner, May 4, 2026: three building blocks](https://www.gartner.com/en/newsroom/press-releases/2026-05-04-gartner-highlights-three-building-blocks-for-autonomous-supply-chain-future) (read via search summary)
- 72% of supply chain organizations deploy GenAI with "middling results", and only 23% of 120 surveyed executives have a formal AI strategy. — [Smart Supply Hub (Jul 15, 2026)](https://smartsupplyhub.com/2026/07/15/agentic-ai-supply-chain-projects-cancelled-2027/). This is a secondary source attributing the figures to Gartner; the primary source was not verified.

### Inferences
- There is a gap between vendor launch velocity (dozens of agents) and measured outcomes. I found no independent, audited case study with before/after KPIs for agentic SCM.
- The reported and implied failure modes are:
  - poor master data and siloed data
  - agents unable to follow nuanced policy over long horizons
  - unclear ROI
  - weak risk controls and auditability
  - agent washing
- Autonomy will be confined to low-risk, reversible decisions for years. Gartner's own forecasts conflict in tone: 60% of disruptions auto-resolved by 2031 versus 5% of organizations at 10% autonomous planning by 2030.

### Gaps
- Microsoft and Google procurement/freight agents were not researched in detail. Examples would be Dynamics 365 supplier-communications agents and Google agents for logistics.
- There is no quantitative incident data on agent errors in supply chain, such as wrong POs or bad replenishment.
- The primary Gartner release text was not fetched (proxy block); details come from search summaries.

## 4. What needs to be created: agent-commerce standards, safety cases, simulation environments

### Takeaway
Early building blocks exist, but none are logistics-specific or mature:
- consumer-oriented agent payment and interoperability protocols (MCP, A2A, AP2)
- company-specific safety cases for AV trucks
- waiver-based regulation
- sim/world-model tools (NVIDIA Cosmos/Isaac)

Missing pieces include:
- B2B agent-to-agent negotiation and contracting standards covering liability
- regulator-accepted safety-case standards and performance metrics for driverless trucks and drones
- validated real-to-sim benchmarks for manipulation
- supply-chain "digital twin" sandboxes for testing agents before they act

### Cited Findings
- Google announced AP2 (Agent Payments Protocol) on Sept 16, 2025 with 60+ partners (Mastercard, PayPal, Amex, Coinbase, Salesforce). It uses signed Intent, Cart and Payment "mandates" to prove user authorization. It extends A2A (Apr 2025) and MCP. — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol); [AP2 docs](https://ap2-protocol.org/)
- AP2 envisions B2B procurement uses, such as autonomous procurement via Google Cloud Marketplace and procurement agents bounded by mandates on vendors, budgets and validity periods. — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol)
- Competing protocols exist (OpenAI/Stripe ACP vs. AP2 vs. A2A vs. MCP), so the standards landscape is fragmented. — [Grid Dynamics](https://www.griddynamics.com/blog/agentic-payments); [Cipher Projects](https://www.cipherprojects.com/blog/posts/agentic-commerce-protocols-mcp-a2a-acp-ap2-compared/)
- Academic work is finding security gaps in AP2 ("Beyond the Mandate", arXiv 2608.23858, Aug 2026). Delayed-fulfilment and refund flows, which are central to freight and procurement, need new profiles: "Agentic Settlement Protocol", arXiv 2609.02208 (Sept 2026). — [arXiv 2608.23858](https://arxiv.org/pdf/2608.23858); [arXiv 2609.02208](https://arxiv.org/pdf/2609.02208) (abstract titles only; contents not read)
- Safety cases: Kodiak reports its safety case is 93% complete **(self-reported)**. Aurora operates under FMCSA waivers and exemptions rather than a performance standard. — [HDT](https://www.truckinginfo.com/news/kodiak-says-driverless-long-haul-launch-on-track-for-2026); [Federal Register](https://www.federalregister.gov/documents/2026/04/15/2026-07288/parts-and-accessories-necessary-for-safe-operation-application-for-exemption-from-aurora-operations)
- Simulation: NVIDIA Cosmos world models and Isaac GR00T target synthetic data. REALM shows that sim benchmarks need real-to-sim validation. — [NVIDIA](https://blogs.nvidia.com/blog/national-robotics-week-2026/); [arXiv 2512.19562](https://arxiv.org/pdf/2512.19562)

### Inferences
Candidate things to invent or standardize:
1. B2B agent-commerce standards for freight and procurement. These need to cover negotiation, tendering and acceptance, delayed fulfilment, partial delivery, claims and chargebacks, liability allocation, and audit trails. AP2/ACP are payment- and consumer-centric.
2. Shared safety-case frameworks and public performance metrics for L4 trucks and BVLOS drones. These would replace one-off waivers and allow independent verification of "zero-attributed-collision" claims.
3. Real-to-sim validated benchmarks for warehouse manipulation. They should report tail reliability (e.g., interventions per 10k picks) across standardized SKU sets.
4. Supply-chain agent sandboxes and digital twins where agents are stress-tested on disruptions before they gain write access to ERP.
5. Governance tooling that encodes Gartner's "decision stack": decision inventories, reversibility and risk tiers, and human-override logging.

### Gaps
- There is no known logistics-specific agent interoperability standard (e.g., GS1, DCSA or EDI bodies working on agent protocols). Not researched; worth a targeted search.
- ISO/UL standards status for humanoids and for AV-truck safety cases (UL 4600) in logistics was not verified.
