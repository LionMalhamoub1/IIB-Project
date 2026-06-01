import glob
import os

import pandas as pd
import plotly.graph_objects as go

EVENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "events")
OUT_DIR = os.path.dirname(__file__)
HTML_OUT = os.path.join(OUT_DIR, "acled_events_animated.html")
MP4_OUT = os.path.join(OUT_DIR, "acled_events_animated.mp4")

EVENT_COLOURS = {
    "Protests": "#5b9bd5",
    "Riots": "#ed7d31",
    "Violence against civilians": "#c00000",
    "Battles": "#7030a0",
    "Explosions/Remote violence": "#ff0000",
    "Strategic developments": "#70ad47",
}
DEFAULT_COLOUR = "#999999"

parquet_files = glob.glob(os.path.join(EVENTS_DIR, "**", "*.parquet"), recursive=True)
if not parquet_files:
    csv_files = glob.glob(os.path.join(EVENTS_DIR, "**", "*.csv"), recursive=True)
    dfs = [pd.read_csv(f, usecols=["event_date", "event_type", "latitude", "longitude",
                                    "country", "location", "fatalities", "iso3"])
           for f in csv_files]
else:
    dfs = [pd.read_parquet(f, columns=["event_date", "event_type", "latitude", "longitude",
                                        "country", "location", "fatalities", "iso3"])
           for f in parquet_files]

df = pd.concat(dfs, ignore_index=True)
df["event_date"] = pd.to_datetime(df["event_date"])
df["month"] = df["event_date"].dt.to_period("M").dt.to_timestamp()
df["fatalities"] = df["fatalities"].fillna(0).astype(int)

months = sorted(df["month"].unique())
event_types = list(EVENT_COLOURS.keys())


def make_trace(subset, name, colour):
    marker_size = (subset["fatalities"].clip(upper=50) / 10 + 4).tolist()
    return go.Scattergeo(
        lat=subset["latitude"].tolist(),
        lon=subset["longitude"].tolist(),
        mode="markers",
        name=name,
        marker=dict(
            color=colour,
            size=marker_size,
            opacity=0.7,
            line=dict(width=0.3, color="white"),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Fatalities: %{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(subset["location"].fillna(""),
                            subset["event_type"],
                            subset["fatalities"])),
    )


frames = []
for month in months:
    frame_data = []
    subset_month = df[df["month"] == month]
    for etype in event_types:
        sub = subset_month[subset_month["event_type"] == etype]
        frame_data.append(make_trace(sub, etype, EVENT_COLOURS[etype]))
    frames.append(go.Frame(data=frame_data, name=str(month.date())))

initial_month = months[0]
initial_data = []
for etype in event_types:
    sub = df[(df["month"] == initial_month) & (df["event_type"] == etype)]
    initial_data.append(make_trace(sub, etype, EVENT_COLOURS[etype]))

sliders = [dict(
    active=0,
    steps=[dict(
        method="animate",
        args=[[str(m.date())], dict(mode="immediate", frame=dict(duration=300, redraw=True),
                                     transition=dict(duration=0))],
        label=str(m.strftime("%b %Y")),
    ) for m in months],
    x=0, y=0,
    len=1.0,
    currentvalue=dict(prefix="Month: ", visible=True, xanchor="center"),
    transition=dict(duration=0),
)]

updatemenus = [dict(
    type="buttons",
    showactive=False,
    y=0.08, x=0.5, xanchor="center",
    buttons=[
        dict(label="Play",
             method="animate",
             args=[None, dict(frame=dict(duration=300, redraw=True),
                              fromcurrent=True, transition=dict(duration=0))]),
        dict(label="Pause",
             method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate", transition=dict(duration=0))]),
    ],
)]

layout = go.Layout(
    title=dict(text="ACLED Disruption Events", x=0.5, font=dict(size=20)),
    geo=dict(
        showland=True, landcolor="#f0f0f0",
        showocean=True, oceancolor="#cce5f0",
        showcoastlines=True, coastlinecolor="#aaaaaa",
        showcountries=True, countrycolor="#cccccc",
        showlakes=False,
        projection_type="natural earth",
    ),
    legend=dict(
        orientation="v", x=1.01, y=0.5,
    ),
    margin=dict(l=0, r=150, t=50, b=100),
    sliders=sliders,
    updatemenus=updatemenus,
)

fig = go.Figure(data=initial_data, layout=layout, frames=frames)
fig.write_html(HTML_OUT, include_plotlyjs="cdn")
print(f"Saved {HTML_OUT}")

try:
    import imageio
    import kaleido  # noqa: F401

    frames_dir = os.path.join(OUT_DIR, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    for i, month in enumerate(months):
        subset_month = df[df["month"] == month]
        frame_data = []
        for etype in event_types:
            sub = subset_month[subset_month["event_type"] == etype]
            frame_data.append(make_trace(sub, etype, EVENT_COLOURS[etype]))
        frame_fig = go.Figure(data=frame_data, layout=layout)
        frame_fig.update_layout(
            title=dict(text=f"ACLED Disruption Events - {month.strftime('%B %Y')}"),
            sliders=[], updatemenus=[],
        )
        path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame_fig.write_image(path, width=1920, height=1080, scale=1)
        frame_paths.append(path)

    writer = imageio.get_writer(MP4_OUT, fps=4, codec="libx264", quality=8)
    for path in frame_paths:
        writer.append_data(imageio.imread(path))
    writer.close()

    for path in frame_paths:
        os.remove(path)
    os.rmdir(frames_dir)

    print(f"Saved {MP4_OUT}")

except ImportError:
    print("MP4 export requires: pip install kaleido imageio[ffmpeg]")
