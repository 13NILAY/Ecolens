"""
ESG Dataset Generator - Production Grade
Generates 50,000-60,000 realistic samples for training ML models on ESG metric extraction
"""

import json
import random
from typing import List, Dict, Tuple
from datetime import datetime

# ==================== METRIC DEFINITIONS ====================

METRIC_DEFINITIONS = {
    # Environmental Metrics
    "GHG_SCOPE_1": {
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
    "GHG_SCOPE_2": {
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
    "GHG_SCOPE_3": {
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
    "TOTAL_GHG": {
        "category": "Environmental",
        "synonyms": [
            "total GHG emissions", "total emissions", "total greenhouse gas emissions",
            "combined emissions", "overall carbon footprint", "aggregate emissions",
            "total carbon emissions", "total CO2 equivalent", "total carbon footprint",
            "net emissions"
        ],
        "units": ["tonnes CO2e", "tCO2e", "million tonnes CO2e", "Mt CO2e", "kg CO2e"],
        "value_range": (100000, 15000000),
        "distribution": "lognormal"
    },
    "CARBON_INTENSITY": {
        "category": "Environmental",
        "synonyms": [
            "carbon intensity", "emissions intensity", "CO2 intensity",
            "carbon footprint per unit", "GHG intensity", "emission factor",
            "specific emissions", "normalized emissions", "emissions per revenue",
            "emissions per employee"
        ],
        "units": ["tCO2e/million USD", "kg CO2e/unit", "tonnes/revenue", "tCO2e/FTE", "kg/m2"],
        "value_range": (0.5, 500),
        "distribution": "uniform"
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
    "RENEWABLE_ENERGY_PCT": {
        "category": "Environmental",
        "synonyms": [
            "renewable energy percentage", "renewable energy share", "clean energy ratio",
            "renewable power percentage", "green energy proportion", "renewable mix",
            "clean energy percentage", "renewables share", "sustainable energy ratio",
            "green power percentage", "RE percentage", "RE%", "renewable energy %",
            "RES percentage", "clean power mix"
        ],
        "units": ["%", "percent", "percentage", "% of total energy"],
        "value_range": (0, 100),
        "distribution": "beta"
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
    "WASTE_RECYCLED_PCT": {
        "category": "Environmental",
        "synonyms": [
            "waste recycled percentage", "recycling rate", "waste diverted from landfill",
            "recycling percentage", "waste recovery rate", "diversion rate",
            "recycled waste ratio", "waste reused percentage", "circular waste percentage",
            "landfill diversion rate"
        ],
        "units": ["%", "percent", "percentage"],
        "value_range": (10, 95),
        "distribution": "beta"
    },
    "EMPLOYEE_COUNT": {
        "category": "Social",
        "synonyms": [
            "employee count", "total employees", "workforce size", "headcount",
            "staff count", "full-time equivalents", "FTE count", "total workforce",
            "number of employees", "team size"
        ],
        "units": ["employees", "FTE", "headcount", "people", ""],
        "value_range": (100, 100000),
        "distribution": "lognormal"
    },
    "GENDER_DIVERSITY_PCT": {
        "category": "Social",
        "synonyms": [
            "gender diversity percentage", "female representation", "women in workforce",
            "gender balance", "female employees percentage", "women representation",
            "gender parity", "female participation rate", "women in leadership",
            "female workforce ratio", "D&I gender metric", "DEI gender ratio",
            "women %", "female %", "gender diversity ratio"
        ],
        "units": ["%", "percent", "percentage", "% female"],
        "value_range": (15, 65),
        "distribution": "beta"
    },
    "TRAINING_HOURS": {
        "category": "Social",
        "synonyms": [
            "training hours", "employee development hours", "learning hours",
            "training time", "professional development hours", "average training hours",
            "hours of training per employee", "L&D hours", "skill development hours",
            "total training hours", "learning and development hours", "training hours/FTE",
            "avg training hours", "development hours per employee"
        ],
        "units": ["hours", "hours/employee", "hours per FTE", "training hours", "hrs/FTE", ""],
        "value_range": (10, 120),
        "distribution": "uniform"
    },
    "INJURY_RATE": {
        "category": "Social",
        "synonyms": [
            "injury rate", "lost time injury rate", "LTIR", "TRIR",
            "total recordable incident rate", "workplace injury frequency",
            "accident rate", "safety incident rate", "injury frequency rate",
            "recordable injuries per million hours", "OSHA rate", "TRI rate",
            "LTIFR", "lost time injury frequency rate", "recordable injury rate"
        ],
        "units": ["per 200,000 hours", "incidents/million hours", "rate", "per 100 FTE", ""],
        "value_range": (0.1, 5.0),
        "distribution": "uniform"
    },
    "EMPLOYEE_TURNOVER_PCT": {
        "category": "Social",
        "synonyms": [
            "employee turnover percentage", "turnover rate", "attrition rate",
            "staff turnover", "employee attrition", "voluntary turnover",
            "retention rate inverse", "churn rate", "separation rate",
            "employee departure rate"
        ],
        "units": ["%", "percent", "percentage"],
        "value_range": (5, 35),
        "distribution": "beta"
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
    "ENVIRONMENTAL_SCORE": {
        "category": "Governance",
        "synonyms": [
            "environmental score", "E score", "environmental rating",
            "environmental pillar score", "environmental performance score",
            "environmental assessment", "ecological rating", "green score",
            "environmental index", "E pillar rating"
        ],
        "units": ["score", "rating", "points", ""],
        "value_range": (25, 95),
        "distribution": "beta"
    },
    "SOCIAL_SCORE": {
        "category": "Governance",
        "synonyms": [
            "social score", "S score", "social rating", "social pillar score",
            "social performance score", "social responsibility score",
            "S pillar rating", "social impact score", "people score",
            "social assessment"
        ],
        "units": ["score", "rating", "points", ""],
        "value_range": (25, 95),
        "distribution": "beta"
    },
    "GOVERNANCE_SCORE": {
        "category": "Governance",
        "synonyms": [
            "governance score", "G score", "corporate governance rating",
            "governance pillar score", "governance performance score",
            "G pillar rating", "governance index", "corporate governance score",
            "governance assessment", "leadership score"
        ],
        "units": ["score", "rating", "points", ""],
        "value_range": (30, 95),
        "distribution": "beta"
    },
    "BOARD_INDEPENDENCE_PCT": {
        "category": "Governance",
        "synonyms": [
            "board independence percentage", "independent directors ratio",
            "board independence", "independent board members percentage",
            "non-executive directors percentage", "independent director ratio",
            "board independence rate", "outside directors percentage",
            "independent board composition", "external directors ratio"
        ],
        "units": ["%", "percent", "percentage"],
        "value_range": (40, 95),
        "distribution": "beta"
    },
    "ETHICS_VIOLATIONS": {
        "category": "Governance",
        "synonyms": [
            "ethics violations", "compliance breaches", "code of conduct violations",
            "ethical incidents", "misconduct cases", "integrity violations",
            "reported ethics cases", "compliance incidents", "policy violations",
            "ethical breaches reported"
        ],
        "units": ["cases", "incidents", "violations", ""],
        "value_range": (0, 50),
        "distribution": "poisson"
    }
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
        # Lognormal for skewed distributions (emissions, energy)
        mu = (min_val + max_val) / 2
        sigma = (max_val - min_val) / 6
        value = random.lognormvariate(0, 1) * sigma + min_val
        value = min(max(value, min_val), max_val)
    elif distribution == "beta":
        # Beta distribution for percentages
        value = random.betavariate(2, 2) * (max_val - min_val) + min_val
    elif distribution == "poisson":
        # Poisson for count data
        lambda_param = (min_val + max_val) / 2
        value = min(random.expovariate(1/lambda_param) if lambda_param > 0 else 0, max_val)
    else:
        value = random.uniform(min_val, max_val)
    
    # Apply rounding based on magnitude
    if value > 10000:
        return round(value, -2)  # Round to nearest 100
    elif value > 100:
        return round(value, 0)
    else:
        return round(value, 2)

def format_value_with_uncertainty(value: float) -> Tuple[str, float]:
    """Add realistic uncertainty markers"""
    uncertainty_type = random.choices(
        ["exact", "approx", "nearly", "around", "over", "under"],
        weights=[0.6, 0.15, 0.1, 0.1, 0.025, 0.025]
    )[0]
    
    if uncertainty_type == "exact":
        return str(value), value
    elif uncertainty_type == "approx":
        return f"approximately {value}", value
    elif uncertainty_type == "nearly":
        return f"nearly {value}", value
    elif uncertainty_type == "around":
        return f"around {value}", value
    elif uncertainty_type == "over":
        adjusted = value * 1.05
        return f"over {value}", adjusted
    else:  # under
        adjusted = value * 0.95
        return f"under {value}", adjusted

# ==================== SAMPLE GENERATION ====================

def select_metrics_for_sample(num_metrics: int) -> List[str]:
    """Select coherent metric combinations"""
    
    # Define metric groups that often appear together
    metric_groups = [
        ["GHG_SCOPE_1", "GHG_SCOPE_2", "GHG_SCOPE_3", "TOTAL_GHG"],
        ["ENERGY_CONSUMPTION", "RENEWABLE_ENERGY_PCT", "CARBON_INTENSITY"],
        ["WATER_USAGE", "WASTE_GENERATED", "WASTE_RECYCLED_PCT"],
        ["EMPLOYEE_COUNT", "GENDER_DIVERSITY_PCT", "TRAINING_HOURS"],
        ["INJURY_RATE", "EMPLOYEE_TURNOVER_PCT", "TRAINING_HOURS"],
        ["ESG_SCORE", "ENVIRONMENTAL_SCORE", "SOCIAL_SCORE", "GOVERNANCE_SCORE"],
        ["BOARD_INDEPENDENCE_PCT", "ETHICS_VIOLATIONS", "GOVERNANCE_SCORE"],
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

def generate_text_from_metrics(metrics: List[Dict], year: int) -> str:
    """Generate realistic text containing the metrics"""
    
    # Select template based on number of metrics
    num_metrics = len(metrics)
    
    if num_metrics == 2:
        template = random.choice([t for t in SENTENCE_TEMPLATES if t.count("{metric") == 2])
    elif num_metrics == 3:
        template = random.choice([t for t in SENTENCE_TEMPLATES if t.count("{metric") == 3])
    elif num_metrics == 4:
        template = random.choice([t for t in SENTENCE_TEMPLATES if t.count("{metric") == 4])
    else:
        # For 5+ metrics, use multiple sentences
        template = random.choice([t for t in SENTENCE_TEMPLATES if t.count("{metric") <= 2])
    
    # Prepare substitution dict
    subs = {"year": year}
    
    for i, metric in enumerate(metrics[:4], 1):  # Limit to 4 metrics per template
        value_str, actual_value = format_value_with_uncertainty(metric["value"])
        
        subs[f"metric{i}"] = metric["metric"]
        subs[f"value{i}"] = value_str
        subs[f"unit{i}"] = metric["unit"]
        subs[f"verb{i}"] = random.choice(["was", "totaled", "reached", "stood at", "amounted to"])
        
        # Update the actual value in case uncertainty changed it
        metric["value"] = actual_value
    
    # Generate base text
    try:
        text = template.format(**subs)
    except KeyError:
        # Fallback for complex templates
        text = f"In {year}, "
        for i, m in enumerate(metrics):
            if i > 0:
                text += ", and " if i == len(metrics) - 1 else ", "
            text += f"{m['metric']} was {m['value']} {m['unit']}"
        text += "."
    
    # Add noise (50% chance)
    if random.random() < 0.5:
        noise_type = random.choice(["prefix", "suffix", "inline"])
        
        if noise_type == "prefix":
            text = random.choice(NOISE_PHRASES) + " " + text
        elif noise_type == "suffix":
            text = text + " " + random.choice(CONTEXTUAL_NOISE)
        else:  # inline
            parts = text.split(". ")
            if len(parts) > 1:
                insert_pos = random.randint(0, len(parts) - 1)
                parts.insert(insert_pos, random.choice(TRANSITION_PHRASES))
                text = " ".join(parts)
    
    # Add extra metrics in separate sentences (for 5-metric samples)
    if len(metrics) > 4:
        extra_metrics = metrics[4:]
        for m in extra_metrics:
            text += f" {random.choice(TRANSITION_PHRASES)} {m['metric']} reached {m['value']} {m['unit']}."
    
    return text

def generate_sample(sample_id: int) -> Dict:
    """Generate a complete sample with text and labeled metrics"""
    
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
    
    return {
        "text": text,
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
    
    # Metric distribution
    metric_counts = {}
    for sample in dataset:
        for metric in sample["metrics"]:
            norm_metric = metric["normalized_metric"]
            metric_counts[norm_metric] = metric_counts.get(norm_metric, 0) + 1
    
    print("\nMetric coverage:")
    for metric, count in sorted(metric_counts.items()):
        print(f"  {metric}: {count:,} occurrences")
    
    # Category distribution
    category_counts = {"Environmental": 0, "Social": 0, "Governance": 0}
    for sample in dataset:
        for metric in sample["metrics"]:
            category_counts[metric["category"]] += 1
    
    print("\nCategory distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count:,} occurrences")
    
    # Metrics per sample distribution
    metrics_per_sample = [len(s["metrics"]) for s in dataset]
    print(f"\nMetrics per sample:")
    print(f"  Average: {sum(metrics_per_sample) / len(metrics_per_sample):.2f}")
    print(f"  Min: {min(metrics_per_sample)}")
    print(f"  Max: {max(metrics_per_sample)}")
    
    # Year distribution
    year_counts = {}
    for sample in dataset:
        year = sample["year"]
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print("\nYear distribution:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count:,} samples")
    
    print("\n" + "="*60)
    print("Sample examples:")
    print("="*60)
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"Text: {sample['text'][:200]}...")
        print(f"Metrics: {len(sample['metrics'])} extracted")
        for m in sample['metrics']:
            print(f"  - {m['normalized_metric']}: {m['value']} {m['unit']}")
    
    print("\n" + "="*60)
    print(f"✓ Dataset generation complete!")
    print(f"✓ Output saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    # Generate 55,000 samples (middle of 50k-60k range)
    generate_dataset(num_samples=55000, output_file="esg_dataset.json")
