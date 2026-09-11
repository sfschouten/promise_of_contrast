"""The `cities` dataset: "The city of X is (not) in Y", with Y sometimes wrong.

Built from the GeoNames export in ``external/geometry_of_truth/geonames.csv`` — cities
with population over 500,000, excluding city-states (where the city name occurs in the
country name) and ambiguous names shared by more than one city.  Each surviving city
yields a true-country and a false-country statement, and each of those is contrasted
against its own negation, so truth is the XOR of "is the country right" and "is the
sentence affirmative".

``COUNTRY_DISPLAY_NAMES`` maps the GeoNames country field to the form used in the
sentence, which is also the whitelist: a city whose country is absent is dropped.  The
mapping is mostly identity and exists for the handful of countries whose GeoNames name is
unusable in running text ("Iran, Islamic Rep. of") or which need an article
("the United Kingdom").

The city sample and the choice of wrong country are drawn with a seed, so the dataset is
reproducible.
"""
from __future__ import annotations

import csv
from pathlib import Path

COUNTRY_DISPLAY_NAMES: dict[str, str] = {
    "Kenya": "Kenya",
    "Croatia": "Croatia",
    "Austria": "Austria",
    "Peru": "Peru",
    "Zambia": "Zambia",
    "Tajikistan": "Tajikistan",
    "Niger": "Niger",
    "Congo, Democratic Republic of the": "the Democratic Republic of the Congo",
    "Algeria": "Algeria",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Cyprus": "Cyprus",
    "Mauritania": "Mauritania",
    "Uruguay": "Uruguay",
    "Slovenia": "Slovenia",
    "Saint Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Bolivia": "Bolivia",
    "Malawi": "Malawi",
    "Bangladesh": "Bangladesh",
    "Turkey": "Turkey",
    "Vanuatu": "Vanuatu",
    "Madagascar": "Madagascar",
    "Hungary": "Hungary",
    "Kyrgyzstan": "Kyrgyzstan",
    "New Zealand": "New Zealand",
    "Uzbekistan": "Uzbekistan",
    "Iran, Islamic Rep. of": "Iran",
    "Togo": "Togo",
    "Tonga": "Tonga",
    "Monaco": "Monaco",
    "Luxembourg": "Luxembourg",
    "Costa Rica": "Costa Rica",
    "Belize": "Belize",
    "Montenegro": "Montenegro",
    "Spain": "Spain",
    "Lebanon": "Lebanon",
    "Poland": "Poland",
    "Tanzania, United Republic of": "Tanzania",
    "Australia": "Australia",
    "Angola": "Angola",
    "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    "Panama": "Panama",
    "Samoa": "Samoa",
    "Switzerland": "Switzerland",
    "Burundi": "Burundi",
    "Mozambique": "Mozambique",
    "Papua New Guinea": "Papua New Guinea",
    "Bulgaria": "Bulgaria",
    "Chile": "Chile",
    "Mali": "Mali",
    "Venezuela, Bolivarian Rep. of": "Venezuela",
    "South Africa": "South Africa",
    "United Kingdom": "the United Kingdom",
    "Comoros": "Comoros",
    "Japan": "Japan",
    "Albania": "Albania",
    "Senegal": "Senegal",
    "Guatemala": "Guatemala",
    "Guinea": "Guinea",
    "Malaysia": "Malaysia",
    "Yemen": "Yemen",
    "Nauru": "Nauru",
    "Syrian Arab Republic": "Syria",
    "Slovakia": "Slovakia",
    "Germany": "Germany",
    "Ecuador": "Ecuador",
    "Lithuania": "Lithuania",
    "Dominica": "Dominica",
    "Azerbaijan": "Azerbaijan",
    "Sudan, The Republic of": "Sudan",
    "Seychelles": "Seychelles",
    "Kiribati": "Kiribati",
    "Iraq": "Iraq",
    "Namibia": "Namibia",
    "Congo": "the Republic of the Congo",
    "Andorra": "Andorra",
    "Canada": "Canada",
    "Korea, Republic of": "South Korea",
    "Bahamas": "the Bahamas",
    "Sierra Leone": "Sierra Leone",
    "Brazil": "Brazil",
    "Finland": "Finland",
    "Ukraine": "Ukraine",
    "Norway": "Norway",
    "Russian Federation": "Russia",
    "Cuba": "Cuba",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "Estonia": "Estonia",
    "Portugal": "Portugal",
    "Greece": "Greece",
    "Bhutan": "Bhutan",
    "Latvia": "Latvia",
    "Central African Republic": "the Central African Republic",
    "Zimbabwe": "Zimbabwe",
    "Lesotho": "Lesotho",
    "Moldova, Republic of": "Moldova",
    "Mauritius": "Mauritius",
    "Palau": "Palau",
    "Nicaragua": "Nicaragua",
    "Djibouti": "Djibouti",
    "Gabon": "Gabon",
    "Dominican Republic": "the Dominican Republic",
    "Qatar": "Qatar",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Kazakhstan": "Kazakhstan",
    "Maldives": "the Maldives",
    "China": "China",
    "Viet Nam": "Vietnam",
    "Korea, Dem. People's Rep. of": "North Korea",
    "Myanmar": "Myanmar",
    "Turkmenistan": "Turkmenistan",
    "Barbados": "Barbados",
    "San Marino": "San Marino",
    "Romania": "Romania",
    "Armenia": "Armenia",
    "United Arab Emirates": "the United Arab Emirates",
    "Malta": "Malta",
    "Uganda": "Uganda",
    "United States": "the United States",
    "Saudi Arabia": "Saudi Arabia",
    "Ethiopia": "Ethiopia",
    "Guyana": "Guyana",
    "Benin": "Benin",
    "India": "India",
    "Macedonia, The former Yugoslav Rep. of": "Macedonia",
    "Philippines": "the Philippines",
    "Mexico": "Mexico",
    "Fiji": "Fiji",
    "Bahrain": "Bahrain",
    "Belarus": "Belarus",
    "Afghanistan": "Afghanistan",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "France": "France",
    "Kuwait": "Kuwait",
    "Czech Republic": "the Czech Republic",
    "Egypt": "Egypt",
    "Jordan": "Jordan",
    "Gambia": "the Gambia",
    "Equatorial Guinea": "Equatorial Guinea",
    "Oman": "Oman",
    "Denmark": "Denmark",
    "Haiti": "Haiti",
    "El Salvador": "El Salvador",
    "Liberia": "Liberia",
    "Tuvalu": "Tuvalu",
    "Burkina Faso": "Burkina Faso",
    "Chad": "Chad",
    "Guinea-Bissau": "Guinea-Bissau",
    "Cape Verde": "Cape Verde",
    "Somalia": "Somalia",
    "Indonesia": "Indonesia",
    "Tunisia": "Tunisia",
    "Belgium": "Belgium",
    "Liechtenstein": "Liechtenstein",
    "Colombia": "Colombia",
    "Lao People's Dem. Rep.": "Laos",
    "Timor-Leste": "Timor-Leste",
    "Honduras": "Honduras",
    "Italy": "Italy",
    "Serbia": "Serbia",
    "Netherlands": "the Netherlands",
    "Jamaica": "Jamaica",
    "Eritrea": "Eritrea",
    "Nepal": "Nepal",
    "Swaziland": "Swaziland",
    "Antigua and Barbuda": "Antigua and Barbuda",
    "Rwanda": "Rwanda",
    "Paraguay": "Paraguay",
    "Sri Lanka": "Sri Lanka",
    "Iceland": "Iceland",
    "Morocco": "Morocco",
    "Suriname": "Suriname",
    "Argentina": "Argentina",
    "Mongolia": "Mongolia",
    "Botswana": "Botswana",
    "Thailand": "Thailand",
    "Cameroon": "Cameroon",
    "Ireland": "Ireland",
    "Nigeria": "Nigeria",
    "Cambodia": "Cambodia",
    "Sweden": "Sweden",
    "Pakistan": "Pakistan",
    "Ghana": "Ghana",
    "Singapore": "Singapore"
}


def load_cities(geonames_csv: Path, min_population: int = 500_000) -> list[tuple[str, str]]:
    """Return (city, country_display_name) for every city passing the filters."""
    with open(geonames_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    name_counts: dict[str, int] = {}
    for row in rows:
        name_counts[row["ASCII Name"]] = name_counts.get(row["ASCII Name"], 0) + 1

    out: list[tuple[str, str]] = []
    for row in rows:
        name, country = row["ASCII Name"], row["Country name EN"]
        if country not in COUNTRY_DISPLAY_NAMES:
            continue
        try:
            if int(float(row["Population"] or 0)) <= min_population:
                continue
        except ValueError:
            continue
        if name in country:                 # city-states: "Singapore" in "Singapore"
            continue
        if name_counts[name] > 1:           # ambiguous city name
            continue
        out.append((name, COUNTRY_DISPLAY_NAMES[country]))
    return out
