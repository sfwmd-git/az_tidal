# Station Metadata Automation

This document describes how station metadata is automatically fetched and calculated for the SMART Tide Predictions system.

## Overview

The `noaa_station_metadata.py` module fetches station metadata from NOAA APIs and calculates derived values. This automates the population of most fields in `station_data.csv`.

## Data Sources

| Source | API Endpoint | Fields Provided |
|--------|--------------|-----------------|
| NOAA Metadata API | `api.tidesandcurrents.noaa.gov/mdapi/prod/webapi` | Station info, datums, flood levels |
| NOAA Data API | `api.tidesandcurrents.noaa.gov/api/prod/datagetter` | Historical data for start_year |
| Calculated | N/A | Bounding box, coast type |

---

## Automated Fields

### 1. Basic Station Information

**Source:** NOAA Metadata API (`/stations/{id}.json`)

| Field | Description | Example |
|-------|-------------|---------|
| `station_ID` | NOAA station identifier | `8723214` |
| `station_name` | Official station name | `Virginia Key` |
| `state` | State abbreviation | `FL` |
| `Latitude` | Station latitude | `25.7314` |
| `Longitude` | Station longitude | `-80.1618` |

### 2. Tidal Datum Offsets

**Source:** NOAA Datums API (`/stations/{id}/datums.json`)

All offsets are calculated relative to Mean Higher High Water (MHHW).

| Field | Calculation | Description |
|-------|-------------|-------------|
| `MHHW_MLLW_offset` | MHHW - MLLW | Tidal range from MLLW to MHHW |
| `navd_offset` | MHHW - NAVD88 | Offset to convert NAVD88 to MHHW datum |
| `ngvd_offset` | MHHW - NGVD29 | Offset to convert NGVD29 to MHHW datum |

**Note:** `ngvd_offset` returns `null` for many stations because NGVD29 (National Geodetic Vertical Datum of 1929) is being phased out and is not available in the NOAA API for all stations.

### 3. Flood Thresholds

**Source:** NOAA Flood Levels API (`/stations/{id}/floodlevels.json`)

Flood thresholds are fetched from NWS (National Weather Service) values and converted to MHHW-relative values to match the format used in predictions.

| Field | Calculation | Description |
|-------|-------------|-------------|
| `minor_flood` | NWS minor threshold - MHHW | Minor flood level above MHHW |
| `moderate_flood` | NWS moderate threshold - MHHW | Moderate flood level above MHHW |
| `major_flood` | NWS major threshold - MHHW | Major flood level above MHHW |

**Example:** Virginia Key
- Raw NWS minor threshold: 13.66 ft (station datum)
- MHHW: 12.38 ft (station datum)
- Result: `minor_flood = 1.28 ≈ 1.3 ft` (above MHHW)

### 4. Bounding Box

**Source:** Calculated from station coordinates

A 1° × 1° bounding box centered on the station location.

| Field | Calculation | Description |
|-------|-------------|-------------|
| `wlon` | Longitude - 0.5° | Western boundary |
| `elon` | Longitude + 0.5° | Eastern boundary |
| `slat` | Latitude - 0.5° | Southern boundary |
| `nlat` | Latitude + 0.5° | Northern boundary |

**Note:** These may need manual adjustment based on coastline orientation and specific modeling needs.

### 5. Start Year

**Source:** NOAA Data API (binary search for earliest available data)

The script queries the NOAA hourly height data API to find the earliest year with available water level data.

| Field | Method | Description |
|-------|--------|-------------|
| `start_year` | Binary search 1850-2025 | First year with hourly water level data |

**Algorithm:**
1. Check if data exists for 2025 (or recent years)
2. Binary search between 1850 and max_year
3. Query `hourly_height` product for January 1st of each test year
4. Return earliest year with valid data

**Performance:** ~12 API calls vs 175 for linear search

### 6. Coast Type

**Source:** Calculated from coordinates

Approximate determination of coast orientation for US stations.

| Value | Criteria |
|-------|----------|
| `west` | Longitude < -115° and Latitude < 50° |
| `east` | Longitude > -85° |
| `south` | Latitude < 32° and -97° < Longitude < -85° |
| `north` | Latitude > 50° |
| `island` | All other cases |

---

## Fields NOT Automated (Manual Entry Required)

### User-Defined Fields

| Field | Reason | Example |
|-------|--------|---------|
| `site_name` | Custom identifier for internal use | `virginia_key` |
| `station_abbr` | User-defined abbreviation | `VK` |

### Calculated Elsewhere

| Field | Where Calculated | Description |
|-------|------------------|-------------|
| `slr_adjustment` | `py_3_Clean Tide Data.py` | 3-year rolling mean of departure (observed - predicted) |

The `slr_adjustment` is **automatically updated** each time the tide data cleaning script runs. It represents the current cumulative sea level rise plus any systematic bias in NOAA predictions.

### Grid Indices (Future Automation)

The 32 grid index fields for GFS and ECMWF models are currently manual but could be automated given known grid specifications:

| Model | Resolution | Domain |
|-------|------------|--------|
| GFS Deterministic Forecast | 0.25° | 20-30°N, 275-285°E |
| GFS Deterministic Wave | 0.16° | Atlantic Ocean (`atlocn`) |
| GFS Ensemble Forecast | 0.25° | 20-30°N, 275-285°E |
| GFS Ensemble Wave | 0.25° | 20-30°N, 275-285°E |
| ECMWF (all products) | TBD | TBD |

---

## Usage

### Fetch Single Station

```python
from noaa_station_metadata import fetch_station_metadata

metadata = fetch_station_metadata('8723214')
print(metadata)
```

### Fetch Multiple Stations

```python
from noaa_station_metadata import fetch_multiple_stations

station_ids = ['8723214', '8724580', '8722670']
all_metadata = fetch_multiple_stations(station_ids, delay=0.5)
```

### Fetch All NOAA Stations

```python
from noaa_station_metadata import NOAAStationMetadata, fetch_multiple_stations

all_ids = NOAAStationMetadata.fetch_all_station_ids()
all_metadata = fetch_multiple_stations(all_ids, delay=0.5)

# Save to JSON
import json
with open('all_stations_metadata.json', 'w') as f:
    json.dump(all_metadata, f, indent=2)
```

---

## Output Format

Example output for Virginia Key (8723214):

```json
{
  "station_ID": "8723214",
  "station_name": "Virginia Key",
  "state": "FL",
  "Latitude": 25.7314,
  "Longitude": -80.1618,
  "MHHW_MLLW_offset": 2.25,
  "navd_offset": 0.23,
  "ngvd_offset": null,
  "minor_flood": 1.3,
  "moderate_flood": 1.7,
  "major_flood": 2.5,
  "wlon": -80.66,
  "elon": -79.66,
  "slat": 25.23,
  "nlat": 26.23,
  "coast_type": "east",
  "start_year": 1995
}
```

---

## Validation Results

Comparison of automated values vs existing `station_data.csv`:

| Station | Field | Automated | CSV | Match |
|---------|-------|-----------|-----|-------|
| Virginia Key | MHHW_MLLW_offset | 2.25 | 2.25 | ✓ |
| Virginia Key | minor_flood | 1.3 | 1.3 | ✓ |
| Virginia Key | moderate_flood | 1.7 | 1.7 | ✓ |
| Virginia Key | major_flood | 2.5 | 2.5 | ✓ |
| Virginia Key | navd_offset | 0.23 | 0.2 | ~✓ |
| Key West | minor_flood | 1.1 | 1.1 | ✓ |
| Key West | start_year | 1914 | 1996 | * |
| Port Everglades | MHHW_MLLW_offset | 2.78 | (empty) | NEW |
| Naples | MHHW_MLLW_offset | 2.87 | (empty) | NEW |

\* Start year differences may be intentional (using only modern data for model training)

---

## API Rate Limiting

The module implements rate limiting to avoid overwhelming NOAA servers:

- **Between stations:** 0.5 second delay (configurable)
- **During start_year search:** 0.2 second delay between binary search queries

---

## Error Handling

- Network errors return empty dictionaries or `None` values
- Missing datums (e.g., NGVD29) return `null`
- Stations without flood threshold data return `null` for flood fields
- Stations without recent data return `null` for start_year

---

## Dependencies

- `requests` - HTTP client for API calls
- Python 3.7+ (for type hints)

No numpy/pandas required for the metadata fetcher.
