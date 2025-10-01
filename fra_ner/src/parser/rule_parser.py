from rapidfuzz import process, fuzz

# Define the fields you want to extract and their possible labels on the form
LABEL_MAP = {
    "CLAIMANT_NAME": ["Name of Claimant", "Claimant Name", "Name"],
    "FATHER_NAME": ["Name of Father/Spouse", "Father Name", "Spouse Name"],
    "VILLAGE": ["Village", "Village Name"],
    "DISTRICT": ["District"],
    "CLAIM_ID": ["Claim ID", "Claim No", "Claim Number"],
    "DATE": ["Date of Submission", "Date"],
    "LAND_AREA": ["Area (hectares)", "Area"],
    "AGE": ["Age"],
    "GENDER": ["Gender"],
    "TRIBE": ["Tribe / Community", "Tribe", "Community"],
    "GRAM_PANCHAYAT": ["Gram Panchayat"],
    "STATE": ["State"],
    "CLAIM_TYPE": ["Type of Claim"],
    "GPS_COORDINATES": ["GPS Coordinates (Lat, Lon)", "GPS Coordinates"],
    "CLAIM_STATUS": ["Claim Status", "Status"]
}


def extract_fields_by_rules(lines: list) -> dict:
    """
    Extracts information using fuzzy matching on labels and simple adjacency rules.
    """
    extracted_data = {}
    line_texts = [line["text"] for line in lines]

    for field_key, patterns in LABEL_MAP.items():
        # Find the line that best matches one of the field's patterns
        best_match = process.extractOne(
            patterns[0], 
            line_texts, 
            scorer=fuzz.partial_ratio, 
            score_cutoff=75
        )

        if not best_match:
            continue

        matched_line_text = best_match[0]
        
        # Simple rule: if the line contains a colon, take the text after it.
        if ":" in matched_line_text:
            value = matched_line_text.split(":", 1)[1].strip()
            extracted_data[field_key] = value

    return extracted_data