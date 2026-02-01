"""
NOAA Tides and Currents Station Metadata Fetcher

This module provides functions to automatically fetch station metadata
from the NOAA Tides and Currents API.
"""

import requests
from typing import Dict, List, Optional, Tuple
import json
import time


class NOAAStationMetadata:
    """Fetch and process NOAA station metadata."""
    
    BASE_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
    
    def __init__(self, station_id: str):
        """
        Initialize with a station ID.

        Args:
            station_id: NOAA station ID (e.g., '9414290')
        """
        self.station_id = station_id
        self.metadata = {}

    @classmethod
    def fetch_all_station_ids(cls) -> List[str]:
        """
        Fetch all available NOAA station IDs.

        Returns:
            List of station ID strings
        """
        url = f"{cls.BASE_URL}/stations.json"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [s['id'] for s in data.get('stations', [])]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching station list: {e}")
            return []

    def fetch_station_info(self) -> Dict:
        """
        Fetch basic station information.

        Returns:
            Dictionary with station_ID, station_name, state, Latitude, Longitude
        """
        url = f"{self.BASE_URL}/stations/{self.station_id}.json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract relevant fields
            stations = data.get('stations', [])
            if stations:
                station = stations[0]
                return {
                    'station_ID': self.station_id,
                    'station_name': station.get('name', ''),
                    'state': station.get('state', ''),
                    'Latitude': float(station.get('lat', 0)),
                    'Longitude': float(station.get('lng', 0))
                }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching station info: {e}")

        return {}
    
    def fetch_datums(self) -> Dict:
        """
        Fetch datum information for offset calculations.

        Returns:
            Dictionary with MHHW_MLLW_offset, navd_offset, and ngvd_offset
        """
        datums, _ = self.fetch_datums_with_mhhw()
        return datums

    def fetch_datums_with_mhhw(self) -> Tuple[Dict, float]:
        """
        Fetch datum information and return both offsets and raw MHHW.

        Returns:
            Tuple of (datum offsets dict, MHHW value)
        """
        url = f"{self.BASE_URL}/stations/{self.station_id}/datums.json"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse datums
            datums = data.get('datums', [])
            datum_dict = {}

            for datum in datums:
                name = datum.get('name', '')
                value = datum.get('value')
                if value is not None:
                    datum_dict[name] = float(value)

            # Calculate offsets (all relative to MHHW)
            mhhw = datum_dict.get('MHHW', 0)
            mllw = datum_dict.get('MLLW', 0)
            navd88 = datum_dict.get('NAVD88')
            ngvd29 = datum_dict.get('NGVD')  # NGVD29 is stored as 'NGVD' in NOAA API

            result = {
                'MHHW_MLLW_offset': round(mhhw - mllw, 3) if mllw else None,
            }

            # NAVD offset (MHHW - NAVD88)
            if navd88 is not None:
                result['navd_offset'] = round(mhhw - navd88, 3)
            else:
                result['navd_offset'] = None

            # NGVD offset (MHHW - NGVD29)
            if ngvd29 is not None:
                result['ngvd_offset'] = round(mhhw - ngvd29, 3)
            else:
                result['ngvd_offset'] = None

            return result, mhhw

        except requests.exceptions.RequestException as e:
            print(f"Error fetching datums: {e}")

        return {}, 0
    
    def fetch_flood_levels(self, mhhw: float = None) -> Dict:
        """
        Fetch NWS flood level thresholds relative to MHHW.

        The flood levels are stored in station datum units (STND) and need
        to be converted to MHHW-relative values for use in predictions.

        Args:
            mhhw: MHHW datum value. If not provided, will be fetched.

        Returns:
            Dictionary with minor_flood, moderate_flood, major_flood (relative to MHHW)
        """
        # Fetch MHHW if not provided
        if mhhw is None:
            datums_url = f"{self.BASE_URL}/stations/{self.station_id}/datums.json"
            try:
                response = requests.get(datums_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                datum_dict = {d['name']: d['value'] for d in data.get('datums', [])}
                mhhw = datum_dict.get('MHHW', 0)
            except requests.exceptions.RequestException:
                mhhw = 0

        # Fetch flood levels from dedicated endpoint
        flood_url = f"{self.BASE_URL}/stations/{self.station_id}/floodlevels.json"

        try:
            response = requests.get(flood_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Use NWS flood levels (preferred for consistency with CSV data)
            nws_minor = data.get('nws_minor')
            nws_moderate = data.get('nws_moderate')
            nws_major = data.get('nws_major')

            # Convert to MHHW-relative values
            result = {
                'minor_flood': round(nws_minor - mhhw, 1) if nws_minor is not None else None,
                'moderate_flood': round(nws_moderate - mhhw, 1) if nws_moderate is not None else None,
                'major_flood': round(nws_major - mhhw, 1) if nws_major is not None else None,
            }

            return result

        except requests.exceptions.RequestException as e:
            print(f"Error fetching flood levels: {e}")

        return {
            'minor_flood': None,
            'moderate_flood': None,
            'major_flood': None
        }
    
    def calculate_bounding_box(self, lat: float, lon: float) -> Dict:
        """
        Calculate 1°x1° bounding box around the station.
        
        This is a simple implementation that centers the box on the station.
        You may need to adjust based on coast orientation.
        
        Args:
            lat: Station latitude
            lon: Station longitude
            
        Returns:
            Dictionary with Wlon, Elon, Slat, Nlat
        """
        # Simple centered box - adjust as needed for your use case
        return {
            'Wlon': round(lon - 0.5, 2),
            'Elon': round(lon + 0.5, 2),
            'Slat': round(lat - 0.5, 2),
            'Nlat': round(lat + 0.5, 2)
        }
    
    def determine_coast_type(self, lat: float, lon: float) -> str:
        """
        Determine coast type based on coordinates.
        
        Args:
            lat: Station latitude
            lon: Station longitude
            
        Returns:
            Coast type: 'east', 'west', 'north', 'south', or 'island'
        """
        # Continental US rough boundaries
        # West Coast: lon < -115
        # East Coast: lon > -85
        # Gulf Coast (South): lat < 32 and lon between -85 and -97
        # Alaska (North): lat > 50
        
        if lon < -115 and lat < 50:
            return 'west'
        elif lon > -85:
            return 'east'
        elif lat < 32 and -97 < lon < -85:
            return 'south'
        elif lat > 50:
            return 'north'
        else:
            return 'island'
    
    def fetch_data_inventory(self) -> Dict:
        """
        Determine the earliest year of available water level data.

        Uses binary search to efficiently find the start year by querying
        the NOAA CO-OPS data API.

        Returns:
            Dictionary with start_year
        """
        DATA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

        def has_data_for_year(year: int) -> bool:
            """Check if data exists for January of the given year."""
            params = {
                'begin_date': f'{year}0101',
                'end_date': f'{year}0102',
                'station': self.station_id,
                'product': 'hourly_height',
                'datum': 'MLLW',
                'units': 'metric',
                'time_zone': 'gmt',
                'format': 'json',
            }
            try:
                response = requests.get(DATA_API_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                # If 'data' key exists and has entries, data is available
                return 'data' in data and len(data['data']) > 0
            except requests.exceptions.RequestException:
                return False

        # Binary search for earliest year with data
        # NOAA tide data generally starts from 1850s for oldest stations
        min_year = 1850
        max_year = 2025
        start_year = None

        # First, check if there's any recent data
        if not has_data_for_year(max_year):
            # Try a few recent years
            for year in [2024, 2023, 2022, 2020]:
                if has_data_for_year(year):
                    max_year = year
                    break
            else:
                # No recent data found
                return {'start_year': None}

        # Binary search to find the earliest year
        while min_year < max_year:
            mid_year = (min_year + max_year) // 2
            time.sleep(0.2)  # Rate limiting

            if has_data_for_year(mid_year):
                max_year = mid_year
            else:
                min_year = mid_year + 1

        # Verify we found valid data
        if has_data_for_year(min_year):
            start_year = min_year

        return {'start_year': start_year}
    
    def get_all_metadata(self) -> Dict:
        """
        Fetch all available metadata for the station.

        Returns:
            Complete metadata dictionary with keys matching station_data.csv format
        """
        metadata = {}

        # Fetch basic info
        station_info = self.fetch_station_info()
        metadata.update(station_info)

        # Fetch datums (includes MHHW_MLLW_offset, navd_offset, ngvd_offset)
        # Also get raw MHHW for flood level conversion
        datums, mhhw = self.fetch_datums_with_mhhw()
        metadata.update(datums)

        # Fetch flood levels (relative to MHHW)
        flood_levels = self.fetch_flood_levels(mhhw=mhhw)
        metadata.update(flood_levels)

        # Calculate bounding box if we have coordinates
        if 'Latitude' in metadata and 'Longitude' in metadata:
            bbox = self.calculate_bounding_box(
                metadata['Latitude'],
                metadata['Longitude']
            )
            # Use lowercase keys to match CSV format
            metadata['wlon'] = bbox['Wlon']
            metadata['elon'] = bbox['Elon']
            metadata['slat'] = bbox['Slat']
            metadata['nlat'] = bbox['Nlat']

            # Determine coast type
            coast_type = self.determine_coast_type(
                metadata['Latitude'],
                metadata['Longitude']
            )
            metadata['coast_type'] = coast_type

        # Fetch data inventory (start_year)
        print(f"  Searching for earliest data year (this may take a moment)...")
        inventory = self.fetch_data_inventory()
        metadata.update(inventory)

        return metadata


def fetch_station_metadata(station_id: str) -> Dict:
    """
    Convenience function to fetch all metadata for a station.
    
    Args:
        station_id: NOAA station ID
        
    Returns:
        Dictionary with all metadata fields
    """
    fetcher = NOAAStationMetadata(station_id)
    return fetcher.get_all_metadata()


def fetch_multiple_stations(station_ids: list, delay: float = 0.5) -> Dict[str, Dict]:
    """
    Fetch metadata for multiple stations.

    Args:
        station_ids: List of NOAA station IDs
        delay: Seconds to wait between requests (rate limiting)

    Returns:
        Dictionary mapping station_id to metadata
    """
    results = {}
    total = len(station_ids)

    for i, station_id in enumerate(station_ids, 1):
        print(f"[{i}/{total}] Fetching metadata for station {station_id}...")
        results[station_id] = fetch_station_metadata(station_id)
        if i < total:
            time.sleep(delay)

    return results


# Example usage
if __name__ == "__main__":
    # Example: Fetch metadata for a single station
    station_id = "8724580"

    metadata = fetch_station_metadata(station_id)
    print(json.dumps(metadata, indent=2))

    # Example: Fetch all station IDs
    # all_ids = NOAAStationMetadata.fetch_all_station_ids()
    # print(f"Found {len(all_ids)} stations")

    # Example: Fetch metadata for ALL stations (takes ~5-10 minutes with rate limiting)
    all_ids = NOAAStationMetadata.fetch_all_station_ids()
    all_metadata = fetch_multiple_stations(all_ids, delay=0.5)
    with open('all_stations_metadata.json', 'w') as f:
        print(json.dump(all_metadata, f, indent=2))

