"""
ESG Dataset Generator - Production Grade (v2 — Real-World Training)
==================================================================
Generates 55,000 realistic samples for training ML models on ESG metric extraction.

Data distribution:
  - Clean sentences: 30%
  - Tables: 25%
  - Broken text: 20%
  - Noisy paragraphs: 15%
  - Multi-value confusion: 10%

Output format:
  {"text": "...", "targets": [{"metric": "SCOPE_1", "value": ..., "unit": "..."}]}
"""

import json
import random
from typing import List, Dict, Tuple
from datetime import datetime

# ==================== METRIC DEFINITIONS ====================

METRIC_DEFINITIONS = {
    # Environmental Metrics
    "SCOPE_1": {
        "category": "Environmental",
        "synonyms": [
            "Scope 1 emissions", "direct emissions", "onsite emissions",
            "direct GHG emissions", "Scope 1 GHG", "direct greenhouse gas emissions",
            "stationary combustion emissions", "company-owned vehicle emissions",
            "direct carbon emissions", "facility emissions", "S1 emissions",
            "Scope 1 CO2e", "direct CO2 emissions"
        ],
        "units": ["tonnes CO2e", "tCO2e", "million tonnes CO2e", "Mt CO2e", "kg CO2e", "mtCO2e"],
        "value_range": (10000, 5000000),
        "distribution": "lognormal"
    },
    "SCOPE_2": {
        "category": "Environmental",
        "synonyms": [
            "Scope 2 emissions", "indirect emissions", "purchased electricity emissions",
            "Scope 2 GHG", "electricity-related emissions", "energy indirect emissions",
            "purchased energy emissions", "grid electricity emissions",
            "location-based emissions", "market-based emissions", "S2 emissions",
            "Scope 2 CO2e", "indirect CO2 emissions"
        ],
        "units": ["tonnes CO2e", "tCO2e", "million tonnes CO2e", "Mt CO2e", "kg CO2e", "mtCO2e"],
        "value_range": (5000, 3000000),
        "distribution": "lognormal"
    },
    "SCOPE_3": {
        "category": "Environmental",
        "synonyms": [
            "Scope 3 emissions", "value chain emissions", "supply chain emissions",
            "upstream emissions", "downstream emissions", "indirect value chain emissions",
            "Scope 3 GHG", "category 3 emissions", "other indirect emissions",
            "third-party emissions", "S3 emissions", "Scope 3 CO2e",
            "Cat 1-15 emissions", "value chain GHG"
        ],
        "units": ["tonnes CO2e", "tCO2e", "million tonnes CO2e", "Mt CO2e", "kg CO2e", "mtCO2e"],
        "value_range": (50000, 10000000),
        "distribution": "lognormal"
    },
    "ENERGY_CONSUMPTION": {
        "category": "Environmental",
        "synonyms": [
            "energy consumption", "total energy use", "energy usage",
            "power consumption", "energy demand", "total energy consumed",
            "operational energy", "facility energy use", "annual energy consumption",
            "electricity and fuel consumption"
        ],
        "units": ["MWh", "GWh", "GJ", "TJ", "kWh"],
        "value_range": (10000, 5000000),
        "distribution": "lognormal"
    },
    "WATER_USAGE": {
        "category": "Environmental",
        "synonyms": [
            "water usage", "water consumption", "water withdrawal", "freshwater use",
            "total water intake", "water demand", "water abstraction",
            "municipal water consumption", "water utilized", "annual water use"
        ],
        "units": ["m3", "cubic meters", "million m3", "liters", "ML", "gallons"],
        "value_range": (10000, 10000000),
        "distribution": "lognormal"
    },
    "WASTE_GENERATED": {
        "category": "Environmental",
        "synonyms": [
            "waste generated", "total waste", "waste production", "waste output",
            "waste created", "solid waste", "operational waste", "annual waste",
            "waste volumes", "waste produced"
        ],
        "units": ["tonnes", "metric tons", "kg", "tons"],
        "value_range": (100, 500000),
        "distribution": "lognormal"
    },
    "ESG_SCORE": {
        "category": "Governance",
        "synonyms": [
            "ESG score", "sustainability rating", "ESG rating", "overall ESG performance",
            "ESG assessment score", "sustainability score", "ESG index score",
            "composite ESG score", "ESG performance rating", "sustainability index",
            "CSR score", "corporate responsibility rating", "MSCI ESG rating",
            "Sustainalytics score", "CDP score"
        ],
        "units": ["score", "rating", "points", "out of 100", ""],
        "value_range": (30, 95),
        "distribution": "beta"
    },
}

# ==================== TEXT TEMPLATES ====================

SENTENCE_TEMPLATES = [
    # Environmental templates
    "In {year}, {metric1} {verb1} {value1} {unit1}, while {metric2} {verb2} {value2} {unit2}.",
    "Our {metric1} for the reporting period totaled {value1} {unit1}, with {metric2} at {value2} {unit2}.",
    "The company reported {metric1} of approximately {value1} {unit1} and {metric2} of {value2} {unit2} in {year}.",
    "{metric1} reached {value1} {unit1} during {year}, alongside {metric2} of {value2} {unit2}.",
    "For fiscal year {year}, {metric1} was recorded at {value1} {unit1}, and {metric2} stood at {value2} {unit2}.",
    "During the reporting period, {metric1} amounted to {value1} {unit1} while {metric2} totaled {value2} {unit2}.",
    "We achieved {metric1} of {value1} {unit1} in {year}. Additionally, {metric2} was {value2} {unit2}.",
    "{metric1}: {value1} {unit1} | {metric2}: {value2} {unit2} | {metric3}: {value3} {unit3} (FY {year})",
    "The organization's {metric1} decreased to {value1} {unit1}, with {metric2} improving to {value2} {unit2}.",
    "Performance metrics for {year}: {metric1} = {value1} {unit1}, {metric2} = {value2} {unit2}.",
    
    # Multi-metric dense templates
    "Sustainability highlights: {metric1} at {value1} {unit1}, {metric2} reaching {value2} {unit2}, and {metric3} of {value3} {unit3}.",
    "Key metrics - {metric1}: {value1} {unit1}; {metric2}: {value2} {unit2}; {metric3}: {value3} {unit3}; {metric4}: {value4} {unit4}.",
    "Environmental performance: {metric1} totaled {value1} {unit1}. {metric2} was {value2} {unit2}. {metric3} stood at {value3} {unit3}.",
    "In {year}, we recorded {metric1} of roughly {value1} {unit1}, {metric2} of about {value2} {unit2}, and {metric3} of nearly {value3} {unit3}.",
    "The following data was collected: {metric1} = {value1} {unit1}, {metric2} = {value2} {unit2}, {metric3} = {value3} {unit3}.",
    
    # Table-like templates
    "{metric1} | {value1} {unit1}\n{metric2} | {value2} {unit2}\n{metric3} | {value3} {unit3}",
    "Metric breakdown:\n- {metric1}: {value1} {unit1}\n- {metric2}: {value2} {unit2}\n- {metric3}: {value3} {unit3}",
    
    # Comparative templates
    "{metric1} increased from previous year to {value1} {unit1}, while {metric2} decreased to {value2} {unit2}.",
    "Compared to last year, {metric1} improved by reaching {value1} {unit1}, and {metric2} was maintained at {value2} {unit2}.",
    "Year-over-year: {metric1} rose to {value1} {unit1}; {metric2} remained steady at {value2} {unit2}.",
    
    # Narrative templates
    "Our sustainability efforts resulted in {metric1} of {value1} {unit1}. We also maintained {metric2} at {value2} {unit2} and achieved {metric3} of {value3} {unit3}.",
    "This year's performance included {metric1} totaling {value1} {unit1}, {metric2} of {value2} {unit2}, and {metric3} reaching {value3} {unit3}.",
    "As part of our commitment to transparency, we report {metric1} at {value1} {unit1} and {metric2} at {value2} {unit2} for the fiscal year {year}.",
    
    # Social templates
    "Workforce metrics: {metric1} of {value1} {unit1}, {metric2} at {value2} {unit2}, and {metric3} totaling {value3} {unit3}.",
    "Our people strategy delivered {metric1} of {value1} {unit1} with {metric2} standing at {value2} {unit2}.",
    "Employee-related KPIs: {metric1} = {value1} {unit1}, {metric2} = {value2} {unit2}.",
    
    # Governance templates
    "Governance indicators for {year}: {metric1} scored {value1} {unit1}, while {metric2} was rated at {value2} {unit2}.",
    "Board composition shows {metric1} at {value1} {unit1}. {metric2} was recorded at {value2} {unit2}.",
    "Compliance data: {metric1} of {value1} {unit1} and {metric2} of {value2} {unit2}.",
    
    # Technical/formal templates
    "According to our verified data, {metric1} equaled {value1} {unit1} and {metric2} was {value2} {unit2} in {year}.",
    "Audited results show {metric1}: {value1} {unit1}, {metric2}: {value2} {unit2}, {metric3}: {value3} {unit3}.",
    "Third-party assessment confirmed {metric1} at {value1} {unit1} and {metric2} at {value2} {unit2}.",
]

# ==================== BROKEN TEXT TEMPLATES (CHANGE 2) ====================

BROKEN_TEMPLATES = [
    "{metric1}\n{value1} {unit1}",
    "{metric1} – {value1}",
    "{metric1} ({unit1}) {value1}",
    "{metric1} : {value1}",
    "{metric1}\t{value1} {unit1}",
    "{metric1}  -  {value1} {unit1}",
    "{metric1}\n  {value1}\n  {unit1}",
    "{value1} {unit1} ({metric1})",
    "{metric1} .... {value1} {unit1}",
    "{metric1} | {value1}",
]

# ==================== NOISE INSERTIONS (CHANGE 3) ====================

NOISE_INSERTIONS = [
    "(Refer Note 12)",
    "(See page 45)",
    "(As per GHG Protocol)",
    "(Audited)",
    "(Unaudited)",
    "(Restated)",
    "(Provisional)",
    "(Scope as defined in GRI 305)",
    "(Including subsidiaries)",
    "(Excluding JVs)",
    "(Refer Annexure A)",
    "(As on 31 March 2023)",
    "*",
    "**",
    "(i)",
    "(ii)",
]

# ==================== MISLEADING DISTRACTOR NUMBERS (CHANGE 7) ====================

MISLEADING_DISTRACTORS = [
    "Total employees: {n}",
    "Revenue: INR {n} crore",
    "Operating profit margin: {pct}%",
    "Number of offices: {small}",
    "CSR spend: INR {n} lakh",
    "Page {page} of {pages}",
    "GRI {gri}-{sub}",
    "FY20{fy1}-{fy2}",
    "Board strength: {small} members",
    "EBITDA: {n} million",
    "{n} locations across {small} countries",
    "Dividend per share: INR {div}",
]

NOISE_PHRASES = [
    "As outlined in our annual report,",
    "Based on comprehensive data collection,",
    "Following industry best practices,",
    "In accordance with GRI standards,",
    "Aligned with our strategic objectives,",
    "As part of our ongoing commitment,",
    "Through rigorous measurement protocols,",
    "Demonstrating our leadership position,",
    "Reflecting our operational excellence,",
    "Consistent with regulatory requirements,",
    "Building on last year's progress,",
    "In line with stakeholder expectations,",
    "Supported by robust methodology,",
    "Validated through external assurance,",
    "As evidenced by our performance,",
    "Per SASB framework,",
    "According to TCFD guidelines,",
    "Following CDP disclosure requirements,",
    "As reported in our 10-K filing,",
    "In our sustainability report,",
    "Based on ISO 14001 standards,",
    "Per GHG Protocol guidelines,",
]

TRANSITION_PHRASES = [
    "Furthermore,", "Additionally,", "Moreover,", "In addition,",
    "Meanwhile,", "Simultaneously,", "Concurrently,", "At the same time,",
    "Notably,", "Importantly,", "Significantly,", "It should be noted that",
]

CONTEXTUAL_NOISE = [
    "This represents a significant milestone in our sustainability journey.",
    "We continue to monitor these metrics closely.",
    "These figures have been independently verified.",
    "Our reporting framework ensures transparency and accuracy.",
    "We remain committed to continuous improvement.",
    "This data reflects our global operations.",
    "Regional variations may apply.",
    "Further details are available in the appendix.",
    "Methodology notes can be found on page 47.",
    "All figures are reported on a calendar year basis.",
    "Data assured by third-party auditor PwC.",
    "Aligns with UN SDG targets.",
    "Verified under ISO 14064 standards.",
    "Reported per GHG Protocol corporate standard.",
    "Disclosure made in accordance with TCFD recommendations.",
    "Calculated using operational control approach.",
]

# ==================== VALUE GENERATION ====================

def generate_value(metric_key: str) -> float:
    """Generate realistic values based on metric type and distribution"""
    config = METRIC_DEFINITIONS[metric_key]
    min_val, max_val = config["value_range"]
    distribution = config["distribution"]
    
    if distribution == "uniform":
        value = random.uniform(min_val, max_val)
    elif distribution == "lognormal":
        mu = (min_val + max_val) / 2
        sigma = (max_val - min_val) / 6
        value = random.lognormvariate(0, 1) * sigma + min_val
        value = min(max(value, min_val), max_val)
    elif distribution == "beta":
        value = random.betavariate(2, 2) * (max_val - min_val) + min_val
    elif distribution == "poisson":
        lambda_param = (min_val + max_val) / 2
        value = min(random.expovariate(1/lambda_param) if lambda_param > 0 else 0, max_val)
    else:
        value = random.uniform(min_val, max_val)
    
    # Apply rounding based on magnitude
    if value > 10000:
        return round(value, -2)
    elif value > 100:
        return round(value, 0)
    else:
        return round(value, 2)


# ==================== CHANGE 5: FORMAT VARIATIONS ====================

def format_number(value: float) -> str:
    """Format numbers in varied ways to mimic real-world report diversity"""
    formats = [
        f"{int(value)}",              # plain integer
        f"{int(value):,}",             # comma separated
        f"{value:.2f}",               # two decimal places
        f"{value/1000:.1f}k",         # shorthand thousands
    ]
    # Add million/lakh shorthand for large numbers
    if value >= 1_000_000:
        formats.append(f"{value/1_000_000:.2f} million")
    if value >= 100_000:
        formats.append(f"{value/100_000:.2f} lakh")
    
    return random.choice(formats)


def format_value_with_uncertainty(value: float) -> Tuple[str, float]:
    """Add realistic uncertainty markers + number format variations"""
    # Use format_number for diverse representations
    value_str = format_number(value)
    
    uncertainty_type = random.choices(
        ["exact", "approx", "nearly", "around", "over", "under"],
        weights=[0.6, 0.15, 0.1, 0.1, 0.025, 0.025]
    )[0]
    
    if uncertainty_type == "exact":
        return value_str, value
    elif uncertainty_type == "approx":
        return f"approximately {value_str}", value
    elif uncertainty_type == "nearly":
        return f"nearly {value_str}", value
    elif uncertainty_type == "around":
        return f"around {value_str}", value
    elif uncertainty_type == "over":
        adjusted = value * 1.05
        return f"over {value_str}", adjusted
    else:  # under
        adjusted = value * 0.95
        return f"under {value_str}", adjusted


# ==================== CHANGE 6: OCR NOISE + BAD SPACING ====================

def corrupt_text(text: str) -> str:
    """Add realistic OCR/PDF extraction noise"""
    # Random extra spacing (30%)
    if random.random() < 0.3:
        words = text.split(' ')
        corrupt_idx = random.randint(0, max(0, len(words) - 2))
        space = random.choice(['  ', '   ', '\n'])
        words[corrupt_idx] = words[corrupt_idx] + space
        text = ' '.join(words)
    
    # Random character substitution — OCR errors (10%)
    if random.random() < 0.1:
        ocr_map = {'O': '0', 'l': '1', 'I': '1', 'S': '5', 'B': '8'}
        chars = list(text)
        for i, c in enumerate(chars):
            if c in ocr_map and random.random() < 0.05:  # 5% per char
                chars[i] = ocr_map[c]
                break  # Only one OCR error per text
        text = ''.join(chars)
    
    return text


# ==================== CHANGE 1: TABLE GENERATION ====================

def generate_table_format(metrics: List[Dict], year: int) -> str:
    """Generate markdown-style table format mimicking real ESG reports"""
    fy_current = f"FY{str(year)[2:]}"
    fy_previous = f"FY{str(year - 1)[2:]}"
    
    table_style = random.choice(['markdown', 'pipe', 'spaced', 'csv'])
    
    if table_style == 'markdown':
        header = f"| Metric | {fy_current} | {fy_previous} |\n|--------|------|------|\n"
        rows = ""
        for m in metrics:
            val1 = int(m["value"])
            val2 = int(val1 * random.uniform(0.8, 1.2))
            rows += f"| {m['metric']} | {val1:,} {m['unit']} | {val2:,} {m['unit']} |\n"
        return header + rows
    
    elif table_style == 'pipe':
        rows = ""
        for m in metrics:
            val1 = format_number(m["value"])
            rows += f"{m['metric']} | {val1} {m['unit']}\n"
        return rows
    
    elif table_style == 'spaced':
        rows = f"{'Indicator':<40} {fy_current:>15} {fy_previous:>15}\n"
        rows += "-" * 70 + "\n"
        for m in metrics:
            val1 = int(m["value"])
            val2 = int(val1 * random.uniform(0.85, 1.15))
            rows += f"{m['metric']:<40} {val1:>15,} {val2:>15,}\n"
        return rows
    
    else:  # csv
        header = f"Metric,{fy_current},Unit\n"
        rows = ""
        for m in metrics:
            rows += f"{m['metric']},{format_number(m['value'])},{m['unit']}\n"
        return header + rows


# ==================== CHANGE 4: MULTI-VALUE CONFUSION ====================

def add_multiple_values(text: str, value: float) -> str:
    """Add a second (older) value to create multi-value confusion"""
    old = int(value * random.uniform(1.05, 1.2))
    patterns = [
        f"{text} reduced from {old:,} to {int(value):,}",
        f"{text} (previous year: {old:,})",
        f"{text}, down from {old:,} in the prior period",
        f"{text} compared to {old:,} last year",
        f"{text} — a decrease from {old:,}",
    ]
    return random.choice(patterns)


# ==================== CHANGE 7: MISLEADING DISTRACTORS ====================

def generate_distractor() -> str:
    """Generate a misleading number that the model must learn to ignore"""
    template = random.choice(MISLEADING_DISTRACTORS)
    return template.format(
        n=random.randint(1000, 50000),
        pct=round(random.uniform(5, 40), 1),
        small=random.randint(5, 50),
        page=random.randint(1, 200),
        pages=random.randint(50, 300),
        gri=random.choice([305, 306, 302, 303, 401, 403, 404]),
        sub=random.randint(1, 5),
        fy1=random.randint(22, 24),
        fy2=random.randint(23, 25),
        div=round(random.uniform(5, 100), 2),
    )

# ==================== SAMPLE GENERATION ====================

def select_metrics_for_sample(num_metrics: int) -> List[str]:
    """Select coherent metric combinations"""
    
    # Define metric groups that often appear together
    metric_groups = [
        ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        ["ENERGY_CONSUMPTION", "WATER_USAGE", "WASTE_GENERATED"],
        ["SCOPE_1", "SCOPE_2", "ENERGY_CONSUMPTION"],
        ["WASTE_GENERATED", "WATER_USAGE", "ESG_SCORE"],
        ["SCOPE_1", "SCOPE_2", "SCOPE_3", "ESG_SCORE"],
    ]
    
    # 70% chance of using a coherent group, 30% random mix
    if random.random() < 0.7 and num_metrics <= 4:
        group = random.choice(metric_groups)
        return random.sample(group, min(num_metrics, len(group)))
    else:
        return random.sample(list(METRIC_DEFINITIONS.keys()), num_metrics)

def generate_metric_object(metric_key: str, year: int) -> Dict:
    """Generate a single metric object with all required fields"""
    config = METRIC_DEFINITIONS[metric_key]
    
    # Select random synonym
    metric_name = random.choice(config["synonyms"])
    
    # Generate value
    value = generate_value(metric_key)
    
    # Select unit (sometimes missing)
    if random.random() < 0.95:  # 95% have units
        unit = random.choice(config["units"])
    else:
        unit = ""
    
    return {
        "metric": metric_name,
        "normalized_metric": metric_key,
        "value": value,
        "unit": unit,
        "category": config["category"]
    }

def _generate_clean_sentence(metrics: List[Dict], year: int) -> str:
    """Generate a clean sentence-based text (original approach)"""
    num_metrics = len(metrics)
    
    if num_metrics == 2:
        templates = [t for t in SENTENCE_TEMPLATES if t.count("{metric") == 2]
    elif num_metrics == 3:
        templates = [t for t in SENTENCE_TEMPLATES if t.count("{metric") == 3]
    elif num_metrics == 4:
        templates = [t for t in SENTENCE_TEMPLATES if t.count("{metric") == 4]
    else:
        templates = [t for t in SENTENCE_TEMPLATES if t.count("{metric") <= 2]
    
    template = random.choice(templates) if templates else SENTENCE_TEMPLATES[0]
    
    subs = {"year": year}
    for i, metric in enumerate(metrics[:4], 1):
        value_str, actual_value = format_value_with_uncertainty(metric["value"])
        subs[f"metric{i}"] = metric["metric"]
        subs[f"value{i}"] = value_str
        subs[f"unit{i}"] = metric["unit"]
        subs[f"verb{i}"] = random.choice(["was", "totaled", "reached", "stood at", "amounted to"])
        metric["value"] = actual_value
    
    try:
        text = template.format(**subs)
    except KeyError:
        text = f"In {year}, "
        for i, m in enumerate(metrics):
            if i > 0:
                text += ", and " if i == len(metrics) - 1 else ", "
            text += f"{m['metric']} was {format_number(m['value'])} {m['unit']}"
        text += "."
    
    # Extra metrics beyond 4
    if len(metrics) > 4:
        for m in metrics[4:]:
            text += f" {random.choice(TRANSITION_PHRASES)} {m['metric']} reached {format_number(m['value'])} {m['unit']}."
    
    return text


def _generate_broken_text(metrics: List[Dict], year: int) -> str:
    """Generate broken/fragmented text mimicking PDF extraction artifacts"""
    parts = []
    for m in metrics:
        template = random.choice(BROKEN_TEMPLATES)
        subs = {
            "metric1": m["metric"],
            "value1": format_number(m["value"]),
            "unit1": m["unit"],
        }
        try:
            parts.append(template.format(**subs))
        except KeyError:
            parts.append(f"{m['metric']} {format_number(m['value'])} {m['unit']}")
    
    separator = random.choice(["\n", "\n\n", " | ", "   "])
    return separator.join(parts)


def generate_text_from_metrics(metrics: List[Dict], year: int) -> str:
    """
    Generate realistic text containing the metrics.
    
    Distribution:
      - 30% clean sentences
      - 25% tables
      - 20% broken text
      - 15% noisy paragraphs (clean + heavy noise + distractors)
      - 10% multi-value confusion
    """
    roll = random.random()
    
    # ---- 25% TABLE FORMAT (CHANGE 1) ----
    if roll < 0.25:
        text = generate_table_format(metrics, year)
        text_type = "table"
    
    # ---- 20% BROKEN TEXT (CHANGE 2) ----
    elif roll < 0.45:
        text = _generate_broken_text(metrics, year)
        text_type = "broken"
    
    # ---- 10% MULTI-VALUE CONFUSION (CHANGE 4) ----
    elif roll < 0.55:
        text = _generate_clean_sentence(metrics, year)
        # Add old/comparison values for the first metric
        text = add_multiple_values(text, metrics[0]["value"])
        text_type = "multi_value"
    
    # ---- 15% NOISY PARAGRAPHS (clean + heavy noise + distractors) ----
    elif roll < 0.70:
        text = _generate_clean_sentence(metrics, year)
        # Prefix noise
        text = random.choice(NOISE_PHRASES) + " " + text
        # Suffix noise
        text = text + " " + random.choice(CONTEXTUAL_NOISE)
        # Add distractor numbers (CHANGE 7)
        text += " " + generate_distractor()
        text_type = "noisy"
    
    # ---- 30% CLEAN SENTENCES ----
    else:
        text = _generate_clean_sentence(metrics, year)
        # Light noise (50% of clean gets a small addition)
        if random.random() < 0.5:
            noise_type = random.choice(["prefix", "suffix", "inline"])
            if noise_type == "prefix":
                text = random.choice(NOISE_PHRASES) + " " + text
            elif noise_type == "suffix":
                text = text + " " + random.choice(CONTEXTUAL_NOISE)
            else:
                parts = text.split(". ")
                if len(parts) > 1:
                    insert_pos = random.randint(0, len(parts) - 1)
                    parts.insert(insert_pos, random.choice(TRANSITION_PHRASES))
                    text = " ".join(parts)
        text_type = "clean"
    
    # ---- CHANGE 3: Noise annotations (30% of all) ----
    if random.random() < 0.3:
        text += " " + random.choice(NOISE_INSERTIONS)
    
    # ---- CHANGE 6: OCR/spacing corruption (15% of all) ----
    if random.random() < 0.15:
        text = corrupt_text(text)
    
    return text

def generate_sample(sample_id: int) -> Dict:
    """Generate a complete sample with text and labeled targets"""
    
    # Select year (2018-2023)
    year = random.randint(2018, 2023)
    
    # Determine number of metrics (2-5, weighted toward 3)
    num_metrics = random.choices([2, 3, 4, 5], weights=[0.25, 0.40, 0.25, 0.10])[0]
    
    # Select metrics
    metric_keys = select_metrics_for_sample(num_metrics)
    
    # Generate metric objects
    metrics = [generate_metric_object(key, year) for key in metric_keys]
    
    # Generate text
    text = generate_text_from_metrics(metrics, year)
    
    # Build targets (upgraded output structure)
    targets = []
    for m in metrics:
        targets.append({
            "metric": m["normalized_metric"],
            "value": m["value"],
            "unit": m["unit"],
        })
    
    return {
        "text": text,
        "targets": targets,
        # Keep full metrics for backward compatibility with phase1
        "metrics": metrics,
        "year": year
    }

# ==================== MAIN GENERATION ====================

def generate_dataset(num_samples: int = 55000, output_file: str = "esg_dataset.json"):
    """Generate the complete dataset"""
    
    print(f"Generating {num_samples:,} ESG samples...")
    print(f"Metrics covered: {len(METRIC_DEFINITIONS)}")
    print(f"Template variations: {len(SENTENCE_TEMPLATES)}")
    
    dataset = []
    
    # Progress tracking
    checkpoint = num_samples // 20
    
    for i in range(num_samples):
        sample = generate_sample(i)
        dataset.append(sample)
        
        if (i + 1) % checkpoint == 0:
            progress = ((i + 1) / num_samples) * 100
            print(f"Progress: {progress:.0f}% ({i+1:,}/{num_samples:,} samples)")
    
    # Save to JSON
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Generate statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(dataset):,}")
    print(f"File size: {len(json.dumps(dataset)) / 1024 / 1024:.2f} MB")
    
    # Metric distribution (from targets)
    metric_counts = {}
    for sample in dataset:
        for target in sample["targets"]:
            metric_counts[target["metric"]] = metric_counts.get(target["metric"], 0) + 1
    
    print("\nMetric coverage (from targets):")
    for metric, count in sorted(metric_counts.items()):
        print(f"  {metric}: {count:,} occurrences")
    
    # Category distribution
    category_counts = {"Environmental": 0, "Governance": 0}
    for sample in dataset:
        for metric in sample["metrics"]:
            cat = metric["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count:,} occurrences")
    
    # Metrics per sample
    metrics_per_sample = [len(s["targets"]) for s in dataset]
    print(f"\nMetrics per sample:")
    print(f"  Average: {sum(metrics_per_sample) / len(metrics_per_sample):.2f}")
    print(f"  Min: {min(metrics_per_sample)}")
    print(f"  Max: {max(metrics_per_sample)}")
    
    # Year distribution
    year_counts = {}
    for sample in dataset:
        year_counts[sample["year"]] = year_counts.get(sample["year"], 0) + 1
    
    print("\nYear distribution:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count:,} samples")
    
    # Text type distribution (approximate from structure)
    print("\nText type distribution (target):")
    print("  Clean sentences: ~30%")
    print("  Tables: ~25%")
    print("  Broken text: ~20%")
    print("  Noisy paragraphs: ~15%")
    print("  Multi-value confusion: ~10%")
    
    print("\n" + "="*60)
    print("Sample examples:")
    print("="*60)
    for i in range(5):
        sample = dataset[i]
        print(f"\n{'─'*60}")
        print(f"Sample {i+1}:")
        print(f"Text: {sample['text'][:300]}")
        print(f"Targets ({len(sample['targets'])}):")
        for t in sample['targets']:
            print(f"  → {t['metric']}: {t['value']} {t['unit']}")
    
    print("\n" + "="*60)
    print(f"✓ Dataset generation complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Format: {{text, targets[{{metric, value, unit}}], metrics, year}}")
    print("="*60)

if __name__ == "__main__":
    # Generate 55,000 samples (middle of 50k-60k range)
    generate_dataset(num_samples=55000, output_file="esg_dataset.json")
