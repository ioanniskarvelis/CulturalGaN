"""Region normalization mapping for Greek locations.

This standardizes region names to avoid duplicates.
"""

import re

# Master region mapping
REGION_MAPPING = {
    # Lesvos/Lesbos (all variations map to Lesvos)
    r'lesvos|lesbos': 'Lesvos',
    
    # Rhodes
    r'rhodes': 'Rhodes',
    
    # Dodecanese
    r'dodecanese': 'Dodecanese',
    
    # Cyclades
    r'cyclades': 'Cyclades',
    
    # Crete
    r'crete|kriti': 'Crete',
    
    # Thessaly
    r'thessaly|thessalia': 'Thessaly',
    
    # Epirus
    r'epirus|epiros': 'Epirus',
    
    # Macedonia
    r'macedonia|makedonia': 'Macedonia',
    
    # Thrace
    r'thrace|thraki': 'Thrace',
    
    # Peloponnese
    r'peloponnese|peloponnisos': 'Peloponnese',
    
    # Attica
    r'attica|attiki|athens': 'Attica',
    
    # Central Greece
    r'central greece|sterea ellada': 'Central_Greece',
    
    # Ionian Islands
    r'ionian|ionia': 'Ionian_Islands',
    
    # North Aegean (when standalone)
    r'north aegean': 'North_Aegean',
    
    # South Aegean
    r'south aegean': 'South_Aegean',
    
    # Aegean Islands (general)
    r'aegean islands': 'Aegean_Islands',
    
    # Turkey
    r'turkey|asia minor': 'Turkey',
}

# Subregion indicators (these get appended if present)
SUBREGION_INDICATORS = {
    'karditsa': 'Karditsa',
    'ioannina': 'Ioannina',
    'mytilene': 'Mytilene',
}

def normalize_region(region_str):
    """
    Normalize a region string to a standardized format.
    
    Examples:
        "Lesvos (Lesbos), North Aegean, Greece" → "Lesvos"
        "Rhodes, Dodecanese, Greece" → "Rhodes_Dodecanese"
        "\tGreece" → "Greece"
        "Rural Greece" → "Greece"
    """
    if not region_str or region_str.strip() == '':
        return 'Unknown'
    
    # Clean up the string
    region = region_str.strip().replace('\t', '')
    region_lower = region.lower()
    
    # Handle special cases
    if region_lower in ['rural greece', 'greece', 'greek']:
        return 'Greece'
    
    if region_lower == 'unknown' or region_lower == 'misc':
        return 'Unknown'
    
    # Initialize result parts
    main_region = None
    subregion = None
    
    # Try to match main regions
    for pattern, standard_name in REGION_MAPPING.items():
        if re.search(pattern, region_lower):
            main_region = standard_name
            break
    
    # If no specific region found, default to Greece
    if not main_region:
        main_region = 'Greece'
    
    # Check for subregions
    for indicator, subregion_name in SUBREGION_INDICATORS.items():
        if indicator in region_lower:
            subregion = subregion_name
            break
    
    # Special case: Rhodes in Dodecanese
    if main_region == 'Rhodes' and 'dodecanese' in region_lower:
        return 'Rhodes_Dodecanese'
    
    # Special case: Lesvos in North Aegean
    if main_region == 'Lesvos' and 'north aegean' in region_lower:
        return 'Lesvos_North_Aegean'
    
    # Construct final region name
    if subregion and main_region != 'Greece':
        return f"{main_region}_{subregion}"
    
    return main_region

def get_region_statistics(regions_list):
    """
    Get statistics on region normalization.
    
    Args:
        regions_list: List of original region strings
        
    Returns:
        dict with original → normalized mapping and counts
    """
    from collections import defaultdict
    
    mapping = {}
    normalized_counts = defaultdict(int)
    
    for original in regions_list:
        normalized = normalize_region(original)
        mapping[original] = normalized
        normalized_counts[normalized] += 1
    
    return {
        'mapping': mapping,
        'counts': dict(sorted(normalized_counts.items(), key=lambda x: x[1], reverse=True)),
        'original_count': len(set(regions_list)),
        'normalized_count': len(normalized_counts)
    }

def test_normalization():
    """Test the normalization with example cases."""
    test_cases = [
        "Lesvos (Lesbos), Greece",
        "Lesvos (Lesbos), North Aegean, Greece",
        "Lesbos, Greece",
        "Lesbos (Lesvos), North Aegean, Greece",
        "Rhodes, Dodecanese, Greece",
        "Rhodes, Greece",
        "\tGreece",
        "Rural Greece",
        "Greece",
        "Thessaly",
        "Epirus",
        "North Aegean, Greece",
        "Aegean Islands, Greece",
        "Turkey",
        "Thrace",
        "Cyclades",
    ]
    
    print("="*70)
    print("REGION NORMALIZATION TEST")
    print("="*70)
    
    for original in test_cases:
        normalized = normalize_region(original)
        print(f"{original:50} → {normalized}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_normalization()

