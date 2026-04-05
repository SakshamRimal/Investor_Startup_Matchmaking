

import numpy as np
import pandas as pd

np.random.seed(42)

SECTORS = [
    "FinTech", "HealthTech", "AI/ML", "CleanTech", "EdTech",
    "SaaS", "Cybersecurity", "Web3", "BioTech", "SpaceTech",
]
STAGES = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]

INVESTOR_THESES = [
    "deep tech and frontier science innovation",
    "consumer mobile and marketplace businesses",
    "enterprise SaaS and developer tools",
    "climate tech and sustainability solutions",
    "healthcare innovation and digital health platforms",
    "fintech and financial inclusion for underserved markets",
    "AI-first companies with strong data moats",
    "web3 infrastructure and decentralized protocols",
]

STARTUP_PITCHES = [
    "machine learning platform automates financial reconciliation for SMBs",
    "AI connects patients with mental health providers via smart matching",
    "LLM-powered developer tooling for enterprise code review and security",
    "carbon credit marketplace using satellite data and blockchain verification",
    "personalized adaptive AI learning platform for K-12 students",
    "API-first banking infrastructure for neobanks in emerging markets",
    "zero-trust cybersecurity platform designed for remote-first enterprises",
    "decentralized identity protocol for cross-chain credential verification",
    "CRISPR-based gene therapy delivery platform targeting rare diseases",
    "satellite IoT connectivity for precision agriculture and food security",
]

INVESTOR_NAMES = [
    "Sequoia Capital", "Andreessen Horowitz", "Accel Partners", "Benchmark Capital",
    "Bessemer Venture", "Founders Fund", "Kleiner Perkins", "General Catalyst",
    "Lightspeed Venture", "NEA Partners", "Index Ventures", "Insight Partners",
    "Tiger Global", "Coatue Management", "GGV Capital", "IVP Ventures",
    "Greylock Partners", "Redpoint Ventures", "Spark Capital", "Union Square",
]

STARTUP_NAMES = [
    "NeuralPay", "MediMatch AI", "CodeGuard", "CarbonLedger", "EduFlow",
    "BankStack", "ZeroTrust Labs", "ChainID", "GenomeCure", "AgroSat",
    "FinBot Pro", "HealthLink", "DevSecAI", "GreenCredit", "LearnPath",
    "NeoBank API", "ShieldAI", "DecentraID", "CRISPRx", "FarmSat",
    "PayFlow AI", "CareConnect", "SecureCode", "EcoChain", "SmartLearn",
    "CoreBanking", "CyberWatch", "BlockPass", "BioTherapy", "PrecisionAg",
]


def generate_investors(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        sectors = np.random.choice(SECTORS, size=np.random.randint(2, 5), replace=False).tolist()
        stages  = np.random.choice(STAGES,  size=np.random.randint(1, 3), replace=False).tolist()
        rows.append({
            "investor_id":       f"INV_{i:03d}",
            "name":              INVESTOR_NAMES[i % len(INVESTOR_NAMES)] + f" {i//len(INVESTOR_NAMES) or ''}".strip(),
            "thesis":            np.random.choice(INVESTOR_THESES),
            "preferred_sectors": sectors,
            "preferred_stages":  stages,
            "min_check":         int(np.random.choice([50_000, 100_000, 250_000, 500_000, 1_000_000])),
            "max_check":         int(np.random.choice([1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000])),
            "portfolio_size":    int(np.random.randint(5, 60)),
            "follow_on_rate":    round(float(np.random.uniform(0.2, 0.8)), 2),
            "years_active":      int(np.random.randint(1, 20)),
            "risk_appetite":     round(float(np.random.uniform(0.3, 1.0)), 2),
            "num_exits":         int(np.random.randint(0, 15)),
            "geography":         np.random.choice(["Global", "US", "Europe", "Asia", "LatAm"]),
        })
    return pd.DataFrame(rows)


def generate_startups(n: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mrr       = float(np.random.lognormal(10, 2))
        burn_rate = mrr * float(np.random.uniform(0.5, 3.0))
        rows.append({
            "startup_id":      f"STP_{i:03d}",
            "name":            STARTUP_NAMES[i % len(STARTUP_NAMES)] + f" {i//len(STARTUP_NAMES) or ''}".strip(),
            "pitch":           np.random.choice(STARTUP_PITCHES),
            "sector":          np.random.choice(SECTORS),
            "stage":           np.random.choice(STAGES),
            "mrr":             round(mrr, 2),
            "growth_rate":     round(float(np.random.uniform(-0.05, 0.35)), 4),
            "burn_rate":       round(burn_rate, 2),
            "runway_months":   round(float(np.random.uniform(3, 24)), 1),
            "team_size":       int(np.random.randint(2, 80)),
            "prior_exits":     int(np.random.randint(0, 3)),
            "total_raised":    round(float(np.random.lognormal(12, 2)), 2),
            "active_users":    int(np.random.randint(100, 500_000)),
            "nps_score":       int(np.random.randint(20, 80)),
            "churn_rate":      round(float(np.random.uniform(0.01, 0.15)), 3),
            "patent_count":    int(np.random.randint(0, 10)),
            "years_operating": round(float(np.random.uniform(0.5, 7)), 1),
        })
    return pd.DataFrame(rows)


def generate_match_history(investors: pd.DataFrame, startups: pd.DataFrame,
                           n: int = 500) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        inv = investors.sample(1).iloc[0]
        stp = startups.sample(1).iloc[0]

        sector_ok = int(stp["sector"] in inv["preferred_sectors"])
        stage_ok  = int(stp["stage"]  in inv["preferred_stages"])
        check_ok  = int(inv["min_check"] <= stp["total_raised"] * 0.3 <= inv["max_check"])
        growth_ok = int(stp["growth_rate"] > 0.10)
        runway_ok = int(stp["runway_months"] > 6)

        prob  = (0.10 + 0.25*sector_ok + 0.20*stage_ok + 0.15*check_ok
                 + 0.15*growth_ok + 0.10*runway_ok + np.random.normal(0, 0.08))
        label = int(np.clip(prob, 0, 1) > 0.5)

        rows.append({
            "investor_id":    inv["investor_id"],
            "startup_id":     stp["startup_id"],
            "sector_match":   sector_ok,
            "stage_match":    stage_ok,
            "check_fit":      check_ok,
            "growth_good":    growth_ok,
            "runway_ok":      runway_ok,
            "risk_appetite":  inv["risk_appetite"],
            "portfolio_size": inv["portfolio_size"],
            "follow_on_rate": inv["follow_on_rate"],
            "mrr":            stp["mrr"],
            "growth_rate":    stp["growth_rate"],
            "burn_rate":      stp["burn_rate"],
            "runway_months":  stp["runway_months"],
            "nps_score":      stp["nps_score"],
            "churn_rate":     stp["churn_rate"],
            "team_size":      stp["team_size"],
            "prior_exits":    stp["prior_exits"],
            "outcome":        label,
        })
    return pd.DataFrame(rows)


# Singletons used across the app
INVESTORS   = generate_investors(30)
STARTUPS    = generate_startups(50)
MATCH_HIST  = generate_match_history(INVESTORS, STARTUPS, 500)
SECTORS_LIST = SECTORS
STAGES_LIST  = STAGES
