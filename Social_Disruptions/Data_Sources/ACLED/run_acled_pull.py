import os

from acled_auth import ACLEDAuth
from acled_client import ACLEDClient, ACLEDClientConfig
from acled_indicators import filter_social_disruption, country_month_panel

# Set these as environment variables (recommended)
# Windows PowerShell:
#   setx ACLED_EMAIL "you@cam.ac.uk"
#   setx ACLED_PASSWORD "your_password"
ACLED_EMAIL = os.environ["ACLED_EMAIL"]
ACLED_PASSWORD = os.environ["ACLED_PASSWORD"]

auth = ACLEDAuth(email=ACLED_EMAIL, password=ACLED_PASSWORD)
client = ACLEDClient(auth, ACLEDClientConfig())

events = client.fetch_events(
    countries=["Chile"],
    start_date="2022-01-01",
    end_date="2023-01-01",
    fields=[
        "event_id_cnty", "event_date", "country", "iso3",
        "admin1", "admin2", "location",
        "event_type", "sub_event_type",
        "actor1", "actor2", "assoc_actor_1", "assoc_actor_2",
        "fatalities", "latitude", "longitude",
        "geo_precision", "time_precision", "source_scale",
    ],
)

social = filter_social_disruption(events)
panel = country_month_panel(social, use_iso3=True, severity="count_plus_fatalities")

print("events:", events.shape)
print("social:", social.shape)
print(panel.head(10))
