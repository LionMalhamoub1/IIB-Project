"""
Generate a synthetic protest-article dataset for testing the clustering pipeline.

Designed clusters
-----------------
A  Chilean teachers' strike           3 articles  explicit_duration "three-week strike"
B  French transport walkout           4 articles  reported_day_number "entered its 8th day"
C  Indian farmers' march              5 articles  long date span (lower-bound coverage)
D  UK nurses' strike                  2 articles  explicit_duration "two-day strike"
E  Brazilian general protest          3 articles  event_start_reference (absolute date)
F  Egyptian political protests        4 articles  continuation_indicator, ordinal day refs
G  Chilean miners' strike (decoy)     2 articles  same country as A, different actors/issue
H  Isolated singletons                4 articles  one each from 4 different countries

Total: 27 articles → expected 8 clusters (A–H, where H = 4 singletons).

Output: data/raw/synthetic_articles.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_OUT = Path(__file__).resolve().parent / "synthetic_articles.csv"

# ── Article builder ────────────────────────────────────────────────────────────

def art(
    article_id: str,
    published_date: str,
    url: str,
    title: str,
    country: str,
    city: str,
    region: str,
    protest_type: str,
    groups: list,
    orgs: list,
    target: str,
    issue: str,
    sector: str,
    participants: str,
    start_ref: str,
    day_number: str,
    explicit_dur: str,
    continuation,
    resolution,
) -> dict:
    ev = {
        "country":                  country,
        "region_or_state":          region,
        "city":                     city,
        "specific_location":        "",
        "protest_type":             protest_type,
        "protesting_groups":        groups,
        "organizations_or_companies": orgs,
        "target_of_protest":        target,
        "issue":                    issue,
        "sector":                   sector,
        "estimated_participants":   participants,
        "event_start_reference":    start_ref,
        "reported_day_number":      day_number,
        "explicit_duration":        explicit_dur,
        "continuation_indicator":   continuation,
        "resolution_indicator":     resolution,
    }
    return {
        "article_id":     article_id,
        "published_date": published_date,
        "url":            url,
        "title":          title,
        "event_json":     json.dumps(ev),
    }


# ── Cluster A — Chilean teachers' strike (3 articles) ─────────────────────────
# All mention "three-week strike" → explicit_duration = 21 days

cluster_a = [
    art("A1", "2023-04-03", "https://news.cl/a1", "Chilean teachers begin nationwide strike",
        "Chile", "Santiago", "Metropolitan Region",
        "strike",
        ["Chilean Teachers Union", "National Educators Federation"],
        ["Colegio de Profesores"],
        "Ministry of Education",
        "wage increase, education funding, class size reduction",
        "education",
        "50000",
        "April 3, 2023",
        "",
        "",
        True, None),

    art("A2", "2023-04-07", "https://latam.cl/a2", "Strike enters second week with no resolution",
        "Chile", "Santiago", "Metropolitan Region",
        "strike",
        ["Chilean Teachers Union"],
        ["Colegio de Profesores"],
        "Ministry of Education",
        "wage increase education funding",
        "education",
        "45000",
        "3 April 2023",
        "second week of the strike",
        "three-week strike looms as talks stall",
        True, None),

    art("A3", "2023-04-21", "https://bbc.com/a3", "Chile three-week teachers strike ends with deal",
        "Chile", "Santiago", "Metropolitan Region",
        "strike",
        ["Chilean Teachers Union", "National Educators Federation"],
        ["Colegio de Profesores"],
        "Government of Chile",
        "teachers wage increase education sector pay",
        "education",
        "60000",
        "",
        "",
        "three-week strike concludes",
        None, True),
]

# ── Cluster B — French transport walkout (4 articles) ─────────────────────────
# day_number signals: entered 8th day → 8 days

cluster_b = [
    art("B1", "2023-06-14", "https://lemonde.fr/b1", "Paris metro workers walk out over pay",
        "France", "Paris", "Île-de-France",
        "strike",
        ["RATP Workers Union"],
        ["RATP Group", "CGT Transport"],
        "RATP Management",
        "pay dispute pension reform public transport",
        "transport",
        "8000",
        "June 14, 2023",
        "",
        "",
        True, None),

    art("B2", "2023-06-17", "https://reuters.com/b2", "French metro strike enters third day",
        "France", "Paris", "Île-de-France",
        "strike",
        ["RATP Workers Union", "SUD Rail"],
        ["CGT Transport"],
        "RATP Management",
        "pay dispute public transport pension",
        "transport",
        "7500",
        "",
        "third day of the Paris transport strike",
        "",
        True, None),

    art("B3", "2023-06-19", "https://france24.com/b3", "Commuters frustrated as Paris walkout continues",
        "France", "Paris", "Île-de-France",
        "walkout",
        ["RATP Workers Union"],
        ["RATP Group"],
        "RATP Management",
        "wage increase transport workers pension reform",
        "transport",
        "",
        "",
        "day 5 of the ongoing walkout",
        "",
        True, None),

    art("B4", "2023-06-21", "https://apnews.com/b4", "Paris transport strike entered its 8th day",
        "France", "Paris", "Île-de-France",
        "strike",
        ["RATP Workers Union", "SUD Rail"],
        ["CGT Transport", "RATP Group"],
        "French Government",
        "public transport workers pay pension rights",
        "transport",
        "10000",
        "",
        "entered its 8th day with no sign of ending",
        "",
        True, None),
]

# ── Cluster C — Indian farmers' march (5 articles, long span) ─────────────────
# Wide date range → lower_bound coverage

cluster_c = [
    art("C1", "2023-02-01", "https://thehindu.com/c1", "Farmers begin march toward Delhi",
        "India", "New Delhi", "Delhi",
        "march",
        ["All India Kisan Sabha", "Bharatiya Kisan Union"],
        [],
        "Central Government",
        "minimum support price farm laws subsidy",
        "agriculture",
        "200000",
        "February 1, 2023",
        "",
        "",
        True, None),

    art("C2", "2023-02-06", "https://ndtv.com/c2", "Farmers protest swells near Delhi border",
        "India", "New Delhi", "Delhi",
        "protest",
        ["All India Kisan Sabha"],
        ["Bharatiya Kisan Union"],
        "Central Government",
        "minimum support price MSP farmers rights",
        "agriculture",
        "250000",
        "",
        "sixth day of the farmers march",
        "",
        True, None),

    art("C3", "2023-02-12", "https://bbc.com/c3", "Indian farmer protest leaders meet government",
        "India", "New Delhi", "Delhi",
        "march",
        ["All India Kisan Sabha", "Samyukta Kisan Morcha"],
        ["Bharatiya Kisan Union"],
        "Ministry of Agriculture",
        "farm laws MSP crop price guarantee",
        "agriculture",
        "180000",
        "",
        "",
        "",
        True, None),

    art("C4", "2023-02-20", "https://reuters.com/c4", "Farmers protest continues into third week",
        "India", "New Delhi", "Delhi",
        "protest",
        ["Samyukta Kisan Morcha"],
        [],
        "Central Government",
        "minimum support price agricultural reform India",
        "agriculture",
        "150000",
        "",
        "twentieth day of demonstrations",
        "",
        True, None),

    art("C5", "2023-03-01", "https://aljazeera.com/c5", "India farm protest ends after government concessions",
        "India", "New Delhi", "Delhi",
        "march",
        ["All India Kisan Sabha", "Samyukta Kisan Morcha"],
        ["Bharatiya Kisan Union"],
        "Prime Minister Office",
        "farm subsidies MSP minimum support price",
        "agriculture",
        "100000",
        "",
        "",
        "",
        None, True),
]

# ── Cluster D — UK nurses' strike (2 articles) ────────────────────────────────
# explicit_duration "two-day strike" → 2 days

cluster_d = [
    art("D1", "2023-01-18", "https://guardian.com/d1", "NHS nurses walk out in historic strike action",
        "UK", "London", "England",
        "strike",
        ["Royal College of Nursing"],
        ["NHS England", "RCN"],
        "NHS Management",
        "nurse pay NHS funding healthcare workers",
        "healthcare",
        "100000",
        "January 18, 2023",
        "",
        "two-day strike over pay",
        True, None),

    art("D2", "2023-01-19", "https://bbc.co.uk/d2", "Second day of NHS nursing strike continues",
        "UK", "London", "England",
        "strike",
        ["Royal College of Nursing"],
        ["NHS England"],
        "Department of Health",
        "nurse salary NHS pay healthcare workers",
        "healthcare",
        "95000",
        "",
        "second day of the nursing strike",
        "two-day walkout",
        None, True),
]

# ── Cluster E — Brazilian general protest (3 articles) ────────────────────────
# event_start_reference absolute date → parsed_start_date

cluster_e = [
    art("E1", "2023-11-09", "https://globo.com/e1", "Brazilians protest against pension reform in Sao Paulo",
        "Brazil", "Sao Paulo", "São Paulo State",
        "protest",
        ["CUT Workers Union", "MST"],
        ["CUT", "PT Workers Party"],
        "Federal Government",
        "pension reform social security workers rights",
        "social",
        "500000",
        "November 8, 2023",
        "",
        "",
        True, None),

    art("E2", "2023-11-10", "https://folha.com/e2", "Anti-pension reform protests spread across Brazil",
        "Brazil", "Sao Paulo", "São Paulo State",
        "demonstration",
        ["CUT Workers Union"],
        ["CUT", "MTST"],
        "Federal Government",
        "pension reform retirement age workers social",
        "social",
        "400000",
        "8 November 2023",
        "second day of protests",
        "",
        True, None),

    art("E3", "2023-11-12", "https://bbc.com/e3", "Brazil protest movement continues amid crackdown",
        "Brazil", "Rio de Janeiro", "Rio de Janeiro State",
        "protest",
        ["MST", "CUT Workers Union"],
        ["PT Workers Party"],
        "President Lula",
        "pension reform social security retirement Brazil",
        "social",
        "300000",
        "",
        "",
        "",
        True, None),
]

# ── Cluster F — Egyptian political protests (4 articles) ─────────────────────
# Ordinal day references, continuation signals

cluster_f = [
    art("F1", "2023-09-20", "https://ahram.eg/f1", "Egyptians protest austerity measures in Cairo",
        "Egypt", "Cairo", "Cairo Governorate",
        "protest",
        ["April 6 Movement", "Egyptian opposition"],
        [],
        "Egyptian Government",
        "austerity economic reform IMF loan cuts",
        "economic",
        "10000",
        "September 20, 2023",
        "",
        "",
        True, None),

    art("F2", "2023-09-23", "https://madamasr.com/f2", "Cairo protests in their fourth day",
        "Egypt", "Cairo", "Cairo Governorate",
        "protest",
        ["April 6 Movement"],
        [],
        "President al-Sisi",
        "austerity measures economic IMF Egypt",
        "economic",
        "8000",
        "",
        "fourth day of protests",
        "",
        True, None),

    art("F3", "2023-09-26", "https://reuters.com/f3", "Egypt unrest: protest movement in its seventh day",
        "Egypt", "Cairo", "Cairo Governorate",
        "protest",
        ["April 6 Movement", "Egyptian opposition"],
        [],
        "Egyptian Government",
        "economic austerity cuts fuel subsidy Egypt",
        "economic",
        "5000",
        "",
        "entered its seventh day",
        "",
        True, None),

    art("F4", "2023-09-28", "https://bbc.com/f4", "Egyptian protests continue as arrests mount",
        "Egypt", "Alexandria", "Alexandria Governorate",
        "demonstration",
        ["Egyptian opposition"],
        [],
        "President al-Sisi",
        "austerity economy protest Egypt fuel price",
        "economic",
        "3000",
        "",
        "",
        "",
        True, None),
]

# ── Cluster G — Chilean miners' strike (DECOY, 2 articles) ───────────────────
# Same country as A (Chile), different sector/actors — should NOT merge with A

cluster_g = [
    art("G1", "2023-04-05", "https://cooperativa.cl/g1", "Copper miners strike at Codelco over bonuses",
        "Chile", "Antofagasta", "Antofagasta Region",
        "strike",
        ["Federacion de Trabajadores del Cobre"],
        ["Codelco"],
        "Codelco Management",
        "mining bonus copper worker salary mining rights",
        "mining",
        "3000",
        "April 5, 2023",
        "",
        "48-hour strike",
        True, None),

    art("G2", "2023-04-06", "https://mineria.cl/g2", "Codelco miners second day of strike action",
        "Chile", "Antofagasta", "Antofagasta Region",
        "strike",
        ["Federacion de Trabajadores del Cobre"],
        ["Codelco"],
        "Codelco Board",
        "copper mining workers bonus strike Codelco",
        "mining",
        "2800",
        "",
        "second day",
        "two-day strike",
        None, True),
]

# ── Cluster H — Isolated singletons (4 articles) ─────────────────────────────
# Each in a different country with no shared actors/cities → 4 separate clusters

cluster_h = [
    art("H1", "2023-07-15", "https://dailynation.ke/h1", "Kenyan teachers protest pay cuts in Nairobi",
        "Kenya", "Nairobi", "Nairobi County",
        "protest",
        ["Kenya National Union of Teachers"],
        ["TSC"],
        "Ministry of Education Kenya",
        "teacher pay salary cuts",
        "education",
        "5000",
        "",
        "",
        "",
        None, None),

    art("H2", "2023-08-22", "https://thenews.pk/h2", "Pakistan lawyers stage protest in Karachi",
        "Pakistan", "Karachi", "Sindh",
        "demonstration",
        ["Pakistan Bar Council"],
        [],
        "Supreme Court",
        "judicial independence rule of law Pakistan",
        "legal",
        "2000",
        "",
        "",
        "",
        None, None),

    art("H3", "2023-05-10", "https://nikkei.com/h3", "South Korean truckers protest fuel surcharge cuts",
        "South Korea", "Seoul", "Seoul Metropolitan",
        "protest",
        ["Korean Cargo Truckers Solidarity"],
        ["KCTU"],
        "Ministry of Transport",
        "fuel surcharge truckers logistics Korea",
        "transport",
        "10000",
        "",
        "",
        "",
        None, None),

    art("H4", "2023-03-18", "https://vanguard.ng/h4", "Nigeria labour unions protest rising fuel prices",
        "Nigeria", "Abuja", "FCT",
        "protest",
        ["Nigeria Labour Congress"],
        ["TUC Nigeria"],
        "NNPC",
        "fuel price subsidy removal oil Nigeria",
        "energy",
        "30000",
        "",
        "",
        "",
        None, None),
]

# ── Assemble and save ─────────────────────────────────────────────────────────

all_articles = (
    cluster_a + cluster_b + cluster_c + cluster_d +
    cluster_e + cluster_f + cluster_g + cluster_h
)

df = pd.DataFrame(all_articles)
df.to_csv(_OUT, index=False)

print(f"Written {len(df)} articles to {_OUT}")
print()
print("Expected clusters:")
print("  A - Chilean teachers' strike        3 articles  explicit_duration -> 21 days")
print("  B - French transport walkout         4 articles  reported_day_number -> 8 days")
print("  C - Indian farmers' march            5 articles  date_range -> 28 days")
print("  D - UK nurses' strike                2 articles  explicit_duration -> 2 days")
print("  E - Brazilian general protest        3 articles  event_start_reference")
print("  F - Egyptian political protests      4 articles  ordinal day refs -> 9 days")
print("  G - Chilean miners' strike (decoy)   2 articles  explicit_duration -> 2 days")
print("  H - 4 isolated singletons            1 each")
print()
print(f"Total: {len(df)} articles -> 11 clusters expected (A-G + 4 singletons)")


if __name__ == "__main__":
    pass  # runs on import for convenience; script is also directly executable
