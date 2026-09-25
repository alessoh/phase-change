# Data, Interoperability and Infrastructure Gaps Limiting AI in Logistics (2025-2026)

Research note: compiled 2026-09-25. Many primary sites (gartner.com, mckinsey.com, iata.org, arxiv.org, gtreview.com) were blocked from direct fetch in this session, so several figures come from search-result snippets of those primary pages or from secondary reporting. They are marked [snippet] or [secondary] and should be spot-checked before they are quoted as final.

## 1. How bad are logistics data quality and visibility today?

### Takeaway
Companies see their Tier-1 suppliers but little beyond them. Between roughly 40% and 58% have any visibility at Tier-2 or deeper. Data readiness is repeatedly named as the main thing stopping AI in supply chains from scaling: Gartner predicts that through 2026, 60% of AI projects without AI-ready data will be abandoned, and only 23% of supply chain organizations have a formal AI strategy. Investment in digital supply chain systems has also slowed sharply.

### Cited Findings
- **McKinsey supply chain risk survey (2024 edition, ~100 leaders; published as "Supply chains: Still vulnerable")**: 95% of respondents have visibility into at least Tier-1 supplier risks, but only 42% of them have visibility into Tier-2 or beyond. The share planning major investments in digital supply chain systems fell from 47% to 25% in one year. Two-thirds have started advanced planning and scheduling (APS) systems, but only ~10% have fully deployed them. About one-third lack clear business cases for digital systems, and some lack the technology to monitor deep-tier risk at scale. [snippet] — [McKinsey 2024](https://www.mckinsey.com/capabilities/operations/our-insights/supply-chain-risk-survey-2024); [WEF, Jan 2025](https://www.weforum.org/stories/2025/01/supply-chain-disruption-digital-winners-losers/)
- **McKinsey Supply chain risk pulse 2025**: reports a 22-percentage-point rise in the share of organizations with Tier-2 supplier visibility, reversing several years of decline. [snippet; I could not confirm whether the 42% figure belongs to 2024 or 2025] — [McKinsey 2025 pulse](https://www.mckinsey.com/capabilities/operations/our-insights/supply-chain-risk-survey)
- **Conflicting secondary citations of the McKinsey 2025 data**: one source gives 90% Tier-1 and 58% Tier-2+ visibility ([D&B blog](https://www.dnb.co.uk/blog/supplier-risk/multi-tier-supply-chain-visibility-gap.html)). Another says 45% still see only Tier-1 and 56% can trace material origin to Tier 3-4 ([Log-hub Pulse](https://log-hub.com/pulse-most-supply-chains-still-go-dark-below-tier-1/)). These are inconsistent restatements, so the primary source is needed.
- 93% of executives report moderate or very high visibility, yet they rank Tier-2+ suppliers as the least visible part of their chain. This is a gap between how confident executives are and what they can actually see. [secondary vendor summary] — [Tradeverifyd](https://tradeverifyd.com/resources/supply-chain-executive-industry-trends)
- A vendor blog claims only 15% of companies have "high visibility" across their supply chains. The original survey is not identified, so treat this with low confidence. — [DataDocks / search summary](https://datadocks.com/posts/edi-vs-api)
- **Gartner (Feb 26, 2025)**: through 2026, organizations will abandon 60% of AI projects not supported by AI-ready data. In a Q3 2024 survey of 248 data-management leaders, 63% either lack the right data management practices for AI or are unsure whether they have them. — [Gartner press release](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)
- **Gartner (Jun 11, 2025)**: only 23% of supply chain organizations have a formal supply chain AI strategy. — [Gartner](https://www.gartner.com/en/newsroom/2025-06-11-gartner-survey-shows-just-23-percent-of-supply-chain-organizations-have-a-formal-ai-strategy)
- **Gartner (Feb 18, 2025)**: only 29% of supply chain organizations have built at least 3 of 5 capabilities needed for future readiness. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-02-18-gartner-survey-shows-only-29-percent-of-supply-chain-organizations-have-built-necessary-capabilities-to-deliver-on-future-performance)
- **Gartner (Apr 29, 2026)**: technology integration and talent are seen as the main barriers to scaling AI in supply chains. Headline only; full text blocked. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-04-29-gartner-survey-finds-technology-integration-and-talent-perceived-as-key-roadblocks-to-scaling-ai-in-supply-chain)
- **Gartner (May 6, 2026; survey of 140 senior supply chain leaders at companies with ≥$250M revenue, Oct-Nov 2025)**: only 17% are pursuing immediate transformational redesign with AI, while 83% apply it incrementally. Gaps in data readiness, the need for upskilling, and fragmented vendor landscapes are constraining near-term deployment. [snippet] — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-05-06-gartner-survey-shows-ai-is-not-driving-supply-chain-operating-model-transformation); [MMH coverage](https://www.mmh.com/article/gartner_survey_ai_is_not_driving_supply_chain_operating_model_transformation)
- **Gartner (Sep 24, 2026)**: predicts only 5% of organizations will make at least 10% of supply chain planning decisions autonomously by 2030. Headline only. — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2026-09-24-gartner-predicts-only-5-percent-of-organizations-will-make-at-least-10-percent-of-supply-chain-planning-decisions-autonomously-by-2030)
- A secondary source reports that Gartner found 72% of supply chain organizations already deploying generative AI, with mostly middling results. — [Zennify summary](https://www.zennify.com/articles/60-of-ai-projects-will-be-abandoned-heres-what-decides-the-rest)

### Inferences
- The binding constraint on AI in logistics is usually the data underneath it: fragmented, ungoverned data, poor multi-tier coverage, and APS/ERP systems that are only partly deployed. Model quality is rarely the limit.
- Visibility is improving at Tier-2, but the dip in digital investment in 2024-25 (47% down to 25% planning major investment) suggests infrastructure spending is lagging behind AI ambitions.

### Gaps
- I could not fetch or confirm specific 2025-26 figures from MIT CTL, Deloitte, BCG or WEF on real-time visibility shares or data quality. Their primary pages were not reached.
- I found no single authoritative figure for the "% of companies with real-time end-to-end visibility." The figures that circulate (for example 15%) come from vendor blogs with unclear sources.
- The McKinsey 2024 and 2025 visibility numbers are inconsistent across secondary sources.

## 2. Interoperability standards and data-sharing efforts: what exists and how far adoption has got

### Takeaway
The standards are mostly in place: DCSA eBL and APIs, IATA ONE Record, GS1 Digital Link, eFTI, the DPP, Catena-X and FIATA eFBL. Real adoption is still early. Around 5% of ocean bills of lading were electronic in early 2025. eBL platforms only became cross-platform interoperable in mid-2026. ONE Record reached its 1 Jan 2026 "preferred standard" date with only about half the industry ready. EU eFTI and Battery Passport mandates arrive in 2027. EDI remains the backbone.

### Cited Findings
**Ocean: DCSA eBL**
- DCSA member carriers, which handle about 75% of global container trade, committed in Feb 2023 to 100% eBL adoption by 2030. — [DCSA 100% eBL](https://dcsa.org/get-involved/100-percent-ebl); [DCSA commitment](https://dcsa.org/newsroom/dcsas-member-carriers-commit-to-a-fully-standardised-electronic-bill-of-lading-by-2030)
- eBL adoption roughly doubled to about 5% (5.7% in January 2025, per snippet), but barriers remain. — [GTR](https://www.gtreview.com/news/digital-trade/ebl-adoption-doubles-to-5-but-barriers-to-digitisation-remain-dcsa-finds/). A vendor blog claims ~11% in 2025, up from ~1.2% in 2021. This conflicts with DCSA's figure and should be treated with caution. — [ClearEye](https://cleareye.ai/electronic-bill-of-lading-news-2026-adoption-standards/)
- Bulk miners BHP, Rio Tinto, Vale and Anglo American reached 25.1% eBL usage by mid-2024. [secondary] — [Lester Aldridge](https://www.lesteraldridge.com/blog/marine/where-the-shipping-industry-stands-on-the-adoption-of-electronic-bills-of-lading-in-2025/)
- May 2026: CargoX and edoxOnline completed the first live, fully interoperable, standards-based eBL transfer between two platforms. — [DCSA](https://dcsa.org/newsroom/ebl-interoperability-milestone)
- June 2026: five eBL platforms (CargoX, edoxOnline, TradeGo, WaveBL, eTEU) implemented the DCSA Interoperability Annex v2 and gained IG P&I Club approval. For the first time, an eBL can be exchanged across platforms without every party using the same system. The framework has three parts: a standard API, a shared legal agreement, and a control registry to prevent duplicate titles. — [DCSA](https://dcsa.org/newsroom/five-ebl-platforms-adopt-dcsa-interoperability-annex); [Cyprus Shipping News, 16 Jun 2026](https://cyprusshippingnews.com/2026/06/16/five-ebl-platforms-adopt-dcsa-standard-annex-v-2-with-igpi-approval-enabling-global-cross-platform-ebl-exchange/)

**Freight forwarding: FIATA eFBL**
- About 1,000 eFBLs had been issued by early 2024, with more than 30 software providers integrated. The eFBL is built on UNCITRAL MLETR. FIATA published a practical eFBL guide on 1 Apr 2026. — [FIATA 2024 stats](https://fiata.org/n/digital-extravaganza-cybersecurity-guide-launched-/); [FIATA eFBL guide](https://fiata.org/n/fiata-launches-efbl-guide-a-roadmap-to-digital-freight-forwarding-and-trusted-trade/)

**Air cargo: IATA ONE Record**
- ONE Record became the preferred air cargo data-sharing standard on 1 Jan 2026. In IATA's December 2025 survey, more than 70% of respondents were aware of it and nearly 50% said they were ready. 78% wanted more pilots and 75% wanted more peer examples. — [IATA press release, 10 Dec 2025](https://www.iata.org/en/pressroom/2025-releases/2025-12-10-02/); [Air Cargo News](https://www.aircargonews.net/technology/2025/12/iata-survey-suggests-one-record-readiness/)
- Airlines representing 72% of AWB volumes were reported to be "on track." IATA says it needs three things: carriers and forwarders to implement, governments to accept ONE Record for regulatory filings, and developers to build open, compatible platforms. [secondary] — [Aviation Business News](https://www.aviationbusinessnews.com/cargo/accelerated-adoption-required-to-realise-full-value-of-one-record-by-2026-says-iata/); [IATA fact sheet](https://www.iata.org/en/iata-repository/pressroom/fact-sheets/fact-sheet-one-record/)
- Cathay Cargo was the first airline to run ONE Record in daily operations, ahead of the 2026 target. — [Aviation Business News](https://www.aviationbusinessnews.com/cargo/cargo-news/cathay-cargo-achieves-one-record-milestone-a-year-early)

**EU: eFTI regulation (EU 2020/1056)**
- From 9 July 2027, Member State authorities must accept freight information supplied electronically through certified eFTI platforms. This covers road, rail, inland waterway and air. Implementing Regulation (EU) 2025/2243, adopted 6 Nov 2025, sets the certification rulebook for platforms. — [ERTICO](https://www.ertico.com/efti-regulation-digital-transformation-of-eu-freight-transport); [European Commission, Jan 2025](https://transport.ec.europa.eu/news-events/news/towards-paperless-freight-transport-eu-takes-step-forward-efti-regulation-implementation-2025-01-09_en); [CLECAT](https://www.clecat.org/en/news/newsletters/new-milestone-for-the-efti-regulation)
- The Open Logistics Foundation (about 50 members and network partners as of May 2025; including Rhenus, Dachser, Blue Yonder and Markant) released open-source, interoperable eCMR software at transport logistic 2025. — [AJOT](https://www.ajot.com/news/open-logistics-foundation-rhenus-and-dachser-presents-new-digital-consignment-note-standard-at-transport-logistic-2025); [Open Logistics Foundation](https://openlogisticsfoundation.org/)

**Product data: GS1 and the Digital Product Passport**
- GS1 Sunrise 2027: retail point-of-sale scanners must read 2D codes (GS1 Digital Link, ISO/IEC 18975) by the end of 2027. Some 48 countries representing about 88% of global GDP are preparing. — [GS1 US](https://www.gs1us.org/industries-and-insights/by-topic/sunrise-2027); [TrackVision, Mar 2026](https://trackvision.ai/blog/2026-03-27-gs1-sunrise-2027-compliance-deadlines)
- The EU Battery Passport, the first mandatory Digital Product Passport (DPP), is due from February 2027. Catena-X, working with the OPC Foundation (Aug 2025), provides DPP data-exchange standards. BASF, BMW, CATL, Henkel, Siemens and VW have built the Path.Era battery passport on Catena-X. — [Catena-X DPP](https://catena-x.net/use-case-cluster/digital-product-passport/); [Catena-X / OPC](https://catena-x.net/news/catena-x-and-opc-foundation-join-forces-to-enable-seamless-industrial-data-exchange-for-the-digital-product-passport/); [Sustainability Magazine](https://sustainabilitymag.com/news/basf-bmw-siemens-digital-passports-batteries)
- Catena-X has demonstrated data-space interoperability with Japan's Ouranos Ecosystem and serves as the template for Manufacturing-X data spaces in other sectors. — [Catena-X / Ouranos](https://catena-x.net/news/catena-x-and-ouranos-ecosystem-successfully-demonstrate-data-space-interoperability/)

**EDI versus APIs**
- EDI remains the backbone of B2B logistics messaging. Companies are adding APIs and cloud integration alongside it (hybrid) rather than replacing it. The US trucking body NMFTA is pushing API standards (its Digital Standards Development Council). — [NMFTA](https://nmfta.org/news/edi-vs-api-in-trucking-why-its-time-to-embrace-api-standards-api-101/); [Coneksion 2026](https://www.coneksion.com/blog/25-great-edi-service-providers)
- API logistics solutions are projected to grow at a 20.2% CAGR to $5.23B by 2030. [vendor-cited market estimate, low confidence] — [Locus](https://locus.sh/blogs/edi-vs-api-integration-logistics/)

### Inferences
- The 2026-2027 window is decisive: eBL cross-platform interoperability (June 2026), ONE Record's preferred-standard date (Jan 2026), the Battery Passport (Feb 2027), eFTI acceptance (Jul 2027) and GS1 Sunrise (end-2027) all fall within it. These deadlines should produce structured, machine-readable event and document data that AI agents can use, but only if companies adopt them beyond compliance minimums.
- Many standards are specific to one mode (ocean, air, road, product). Multimodal AI will need semantic mapping between them, for example UN/CEFACT reference models, which I did not verify.

### Gaps
- I found no current (2026) measured eBL penetration figure from DCSA. The best available is about 5% in early 2025.
- I found no quantified eFTI platform certification count, and no quantified ONE Record live transaction volume.
- UN/CEFACT MMT reference data model adoption and the WCO/UN standards were not researched in depth here.
- I found no reliable survey giving the EDI versus API share by shipper or carrier.

## 3. Digital twins, real-time IoT/telemetry, and federated or privacy-preserving data sharing

### Takeaway
Digital twins are still a minority practice: about 21% of operations use them, per PwC 2025. Users rate them highly, but they depend on the same data plumbing that is missing. Federated learning and data spaces are the proposed way for competitors to share without exposing sensitive data. Outside Catena-X-style data spaces, the evidence is still mostly academic, and free-riding and trust are unresolved problems.

### Cited Findings
- PwC 2025 Digital Trends in Operations Survey: 21% of companies report using digital twins in operations, and 97% of those users say the twins are somewhat or very effective at creating value. [snippet] — [PwC](https://www.pwc.com/us/en/services/consulting/supply-chain-operations/digital-supply-chain-survey.html)
- Forecasts cited secondhand: Gartner expects more than 40% of large enterprises to use digital twins by 2027, and IDC expects 60% of G2000 manufacturers to use them for supply chain simulation by 2027. [secondary, unverified] — [Locus blog](https://locus.sh/blogs/real-time-supply-chain-digital-twins-go-mainstream/)
- Ford Motor Company case study on building a supply chain digital twin at scale (Omega, 2025). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0305048325001732)
- A 2025 review of digital twins for supply chain resilience finds integration and data-quality barriers. — [MDPI Logistics 9(1):22](https://www.mdpi.com/2305-6290/9/1/22)
- Federated learning for logistics: the 2025 SafeLogFL system has cross-border logistics participants train a risk-warning model locally without sharing their raw data. — [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-25507-1)
- Adaptive federated learning for supply chain information sharing. — [IJPR 2024](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2432469)
- Free-riding is a key obstacle when competitors use federated learning. — [arXiv 2410.12723](https://arxiv.org/pdf/2410.12723); [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2949948825000046)
- Data spaces (Catena-X, compliant with the International Data Spaces approach) use connectors that keep data sovereign, so partners can share selectively. This is the main production-grade model for cross-company sharing in Europe. — [Catena-X](https://catena-x.net/use-case-cluster/digital-product-passport/)

### Inferences
- Digital twins are only as good as their inputs. With less than 60% Tier-2 visibility and EDI batch latency, most "twins" are partial models, not real-time mirrors of the network.
- What is missing: shared event-data semantics across modes, trusted incentive and governance schemes for sharing between competitors, and production evidence (not just papers) that federated learning works in freight.

### Gaps
- I found no reliable 2025-26 statistic on the share of shipments or containers with real-time IoT telemetry. For example, DCSA/carrier smart-container penetration was not found.
- I found no quantified adoption data for federated learning in commercial logistics.

## 4. Public logistics datasets and benchmarks for AI

### Takeaway
Recent research consistently says the lack of large, realistic, open supply chain datasets is a key limit on AI in this field. Existing datasets are proprietary, small or poorly maintained. Researchers are responding with synthetic data generators and new LLM benchmarks (2024-2026), but nothing yet matches ImageNet or GLUE for logistics.

### Cited Findings
- The SynDelay paper (Sept 2025) says AI in supply chain management is "severely constrained by the lack of high-quality, openly available datasets." It describes existing datasets as proprietary, small-scale or poorly maintained, which limits reproducibility and benchmarking for delay prediction, demand forecasting and inventory. It contrasts this with computer vision and NLP. [snippet] — [arXiv 2509.05325](https://arxiv.org/pdf/2509.05325)
- Existing real-world datasets include the Amazon Last-Mile Routing dataset and LaDe (a last-mile delivery dataset). [snippet] — [arXiv 2509.05325](https://arxiv.org/pdf/2509.05325)
- SupplyGraph (2024) is a graph neural network (GNN) benchmark built from an FMCG company's data, created because public data is scarce. — [arXiv 2401.15299](https://arxiv.org/html/2401.15299v3)
- New LLM benchmarks: SupChain-Bench (Feb 2026) for real-world supply chain management with LLMs, and CSCBench (Jan 2026) for commodity supply chain reasoning. — [arXiv 2602.07342](https://arxiv.org/pdf/2602.07342); [arXiv 2601.01825](https://arxiv.org/pdf/2601.01825)
- Synthetic data and simulation to fill the gap: ISOMORPH supply chain digital twin for dataset generation (May 2026), LLM-driven synthetic supply chain data (May 2026), and a review of synthetic data for supply chain ML. — [arXiv 2605.12768](https://arxiv.org/pdf/2605.12768); [arXiv 2605.26823](https://arxiv.org/pdf/2605.26823); [IJPR 2025](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2447927)

### Inferences
- Benchmarks are beginning to emerge in 2026, especially for LLMs and agents, but they are fragmented and mostly synthetic or taken from a single company. What is needed is shared, anonymized, multi-party datasets (for example from data spaces or regulatory data like eFTI) and standard evaluation tasks.

### Gaps
- I could not fetch the arXiv abstracts to confirm details or results for SupChain-Bench, CSCBench or ISOMORPH.
- I found no survey measuring how many logistics AI papers use proprietary versus public data.
