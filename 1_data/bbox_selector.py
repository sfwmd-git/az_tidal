"""
Interactive Bounding Box Selector for Station Grid Indices

Displays a simple map with the station location and allows you to drag/resize
a bounding box to select the ocean area for weather model data extraction.

Usage:
    python bbox_selector.py                 # Iterate through all stations in station_data.csv
    python bbox_selector.py <station_id>    # Single station mode

Example:
    python bbox_selector.py
    python bbox_selector.py 8723214

Controls:
    - Click and drag inside box to move it
    - Click and drag edges/corners to resize
    - Press 's' to save and go to NEXT station
    - Press 'n' to skip to next station without saving
    - Press 'q' to quit entirely
    - Press 'r' to reset to default 1x1 degree box
"""

import os
import sys
import json
import csv
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.widgets import Button


# Grid specifications for each model/product
GRID_SPECS = {
    'gfs_det_forecast': {
        'resolution': 0.25,
        'origin_lat': 20.0,
        'origin_lon': 275.0,  # -85 in 0-360 format
    },
    'gfs_det_wave': {
        'resolution': 0.16,
        # atlocn domain - need to verify these
        'origin_lat': 0.0,
        'origin_lon': 260.0,
    },
    'gfs_ens_forecast': {
        'resolution': 0.25,
        'origin_lat': 20.0,
        'origin_lon': 275.0,
    },
    'gfs_ens_wave': {
        'resolution': 0.25,
        'origin_lat': 20.0,
        'origin_lon': 275.0,
    },
    'ecmwf_det_forecast': {
        'resolution': 0.1,
        'origin_lat': 90.0,  # ECMWF starts from north
        'origin_lon': 0.0,
        'lat_direction': -1,  # decreasing latitude
    },
    'ecmwf_det_wave': {
        'resolution': 0.125,
        'origin_lat': 90.0,
        'origin_lon': 0.0,
        'lat_direction': -1,
    },
    'ecmwf_ens_forecast': {
        'resolution': 0.1,
        'origin_lat': 90.0,
        'origin_lon': 0.0,
        'lat_direction': -1,
    },
    'ecmwf_ens_wave': {
        'resolution': 0.125,
        'origin_lat': 90.0,
        'origin_lon': 0.0,
        'lat_direction': -1,
    },
}


def calculate_grid_indices(wlon, elon, slat, nlat, grid_spec):
    """Calculate grid indices from bounding box coordinates."""
    res = grid_spec['resolution']
    lat_dir = grid_spec.get('lat_direction', 1)

    # Convert longitude to 0-360 if needed
    wlon_360 = wlon + 360 if wlon < 0 else wlon
    elon_360 = elon + 360 if elon < 0 else elon

    if lat_dir == -1:
        # ECMWF style (latitude decreases with index)
        nlat_idx = int((grid_spec['origin_lat'] - nlat) / res)
        slat_idx = int((grid_spec['origin_lat'] - slat) / res)
    else:
        # GFS style (latitude increases with index)
        slat_idx = int((slat - grid_spec['origin_lat']) / res)
        nlat_idx = int((nlat - grid_spec['origin_lat']) / res)

    wlon_idx = int((wlon_360 - grid_spec['origin_lon']) / res)
    elon_idx = int((elon_360 - grid_spec['origin_lon']) / res)

    return {
        'nlat_idx': nlat_idx,
        'slat_idx': slat_idx,
        'wlon_idx': wlon_idx,
        'elon_idx': elon_idx,
    }


class BoundingBoxSelector:
    def __init__(self, station_id, station_name, lat, lon, existing_bbox=None,
                 current_idx=0, total_count=1):
        self.station_id = station_id
        self.station_name = station_name
        self.station_lat = lat
        self.station_lon = lon
        self.current_idx = current_idx
        self.total_count = total_count

        # Use existing bbox if provided, otherwise default 1x1 degree box
        if existing_bbox:
            self.bbox = existing_bbox.copy()
        else:
            self.bbox = {
                'wlon': round(lon - 0.5, 2),
                'elon': round(lon + 0.5, 2),
                'slat': round(lat - 0.5, 2),
                'nlat': round(lat + 0.5, 2),
            }

        self.fig = None
        self.ax = None
        self.rect = None
        self.info_text = None
        self.dragging = False
        self.resize_mode = None
        self.drag_start = None
        self.drag_bbox_start = None
        self.saved = False
        self.skip_to_next = False
        self.quit_all = False

    def setup_plot(self):
        """Create the map with station marker and bounding box."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Set extent around station (3 degrees padding)
        padding = 3
        self.ax.set_xlim(self.station_lon - padding, self.station_lon + padding)
        self.ax.set_ylim(self.station_lat - padding, self.station_lat + padding)

        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_axisbelow(True)

        # Color background as ocean (light blue)
        self.ax.set_facecolor('lightblue')

        # Plot station location
        self.ax.plot(self.station_lon, self.station_lat, 'r*', markersize=20,
                     label=f'Station: {self.station_name}', zorder=10)

        # Create draggable rectangle
        self.rect = Rectangle(
            (self.bbox['wlon'], self.bbox['slat']),
            self.bbox['elon'] - self.bbox['wlon'],
            self.bbox['nlat'] - self.bbox['slat'],
            fill=True, facecolor='yellow', alpha=0.3,
            edgecolor='red', linewidth=2, zorder=5
        )
        self.ax.add_patch(self.rect)

        # Add corner handles for visual feedback
        self.update_handles()

        # Labels
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        progress = f"[{self.current_idx + 1}/{self.total_count}] " if self.total_count > 1 else ""
        self.ax.set_title(
            f"{progress}{self.station_name} ({self.station_id})\n"
            "'s'=save & next | 'n'=skip | 'r'=reset | 'q'=quit all",
            fontsize=11
        )
        self.ax.legend(loc='upper right')

        # Info text
        self.info_text = self.ax.text(
            0.02, 0.02,
            self.get_info_string(),
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update_handles(self):
        """Update corner handle markers."""
        # Remove old handles if they exist
        for artist in self.ax.patches[:]:
            if isinstance(artist, plt.Circle):
                artist.remove()

    def get_info_string(self):
        return (f"W: {self.bbox['wlon']:.2f}  E: {self.bbox['elon']:.2f}\n"
                f"S: {self.bbox['slat']:.2f}  N: {self.bbox['nlat']:.2f}")

    def update_rect(self):
        """Update rectangle position from bbox."""
        self.rect.set_xy((self.bbox['wlon'], self.bbox['slat']))
        self.rect.set_width(self.bbox['elon'] - self.bbox['wlon'])
        self.rect.set_height(self.bbox['nlat'] - self.bbox['slat'])
        self.info_text.set_text(self.get_info_string())
        self.fig.canvas.draw_idle()

    def get_resize_mode(self, x, y):
        """Determine if click is on edge/corner for resizing."""
        # Calculate margin based on view extent
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        margin_x = (xlim[1] - xlim[0]) * 0.02
        margin_y = (ylim[1] - ylim[0]) * 0.02

        on_left = abs(x - self.bbox['wlon']) < margin_x
        on_right = abs(x - self.bbox['elon']) < margin_x
        on_bottom = abs(y - self.bbox['slat']) < margin_y
        on_top = abs(y - self.bbox['nlat']) < margin_y

        in_x = self.bbox['wlon'] - margin_x <= x <= self.bbox['elon'] + margin_x
        in_y = self.bbox['slat'] - margin_y <= y <= self.bbox['nlat'] + margin_y

        if on_left and on_top and in_x and in_y:
            return 'nw'
        elif on_right and on_top and in_x and in_y:
            return 'ne'
        elif on_left and on_bottom and in_x and in_y:
            return 'sw'
        elif on_right and on_bottom and in_x and in_y:
            return 'se'
        elif on_left and in_y:
            return 'w'
        elif on_right and in_y:
            return 'e'
        elif on_top and in_x:
            return 'n'
        elif on_bottom and in_x:
            return 's'
        return None

    def point_in_rect(self, x, y):
        """Check if point is inside rectangle."""
        return (self.bbox['wlon'] <= x <= self.bbox['elon'] and
                self.bbox['slat'] <= y <= self.bbox['nlat'])

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        x, y = event.xdata, event.ydata

        # Check for resize mode first
        self.resize_mode = self.get_resize_mode(x, y)
        if self.resize_mode:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_bbox_start = self.bbox.copy()
            return

        # Check for drag (click inside box)
        if self.point_in_rect(x, y):
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_bbox_start = self.bbox.copy()
            self.resize_mode = 'move'

    def on_release(self, event):
        if self.dragging:
            # Round to 2 decimal places
            self.bbox = {k: round(v, 2) for k, v in self.bbox.items()}
            self.update_rect()
        self.dragging = False
        self.resize_mode = None
        self.drag_start = None
        self.drag_bbox_start = None

    def on_motion(self, event):
        if not self.dragging or event.xdata is None:
            return

        x, y = event.xdata, event.ydata
        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]

        if self.resize_mode == 'move':
            self.bbox['wlon'] = self.drag_bbox_start['wlon'] + dx
            self.bbox['elon'] = self.drag_bbox_start['elon'] + dx
            self.bbox['slat'] = self.drag_bbox_start['slat'] + dy
            self.bbox['nlat'] = self.drag_bbox_start['nlat'] + dy
        else:
            # Resize modes
            if 'w' in self.resize_mode:
                self.bbox['wlon'] = min(x, self.bbox['elon'] - 0.1)
            if 'e' in self.resize_mode:
                self.bbox['elon'] = max(x, self.bbox['wlon'] + 0.1)
            if 's' in self.resize_mode:
                self.bbox['slat'] = min(y, self.bbox['nlat'] - 0.1)
            if 'n' in self.resize_mode:
                self.bbox['nlat'] = max(y, self.bbox['slat'] + 0.1)

        self.update_rect()

    def on_key(self, event):
        if event.key == 's':
            self.save_indices()
            self.saved = True
            plt.close(self.fig)
        elif event.key == 'n':
            # Skip to next without saving
            self.skip_to_next = True
            plt.close(self.fig)
        elif event.key == 'q':
            # Quit entirely
            self.quit_all = True
            plt.close(self.fig)
        elif event.key == 'r':
            # Reset to default
            self.bbox = {
                'wlon': round(self.station_lon - 0.5, 2),
                'elon': round(self.station_lon + 0.5, 2),
                'slat': round(self.station_lat - 0.5, 2),
                'nlat': round(self.station_lat + 0.5, 2),
            }
            self.update_rect()

    def save_indices(self):
        """Calculate and save grid indices for all model products."""
        print("\n" + "="*60)
        print(f"GRID INDICES FOR {self.station_name} ({self.station_id})")
        print("="*60)
        print(f"\nBounding Box:")
        print(f"  wlon: {self.bbox['wlon']:.2f}")
        print(f"  elon: {self.bbox['elon']:.2f}")
        print(f"  slat: {self.bbox['slat']:.2f}")
        print(f"  nlat: {self.bbox['nlat']:.2f}")

        results = {
            'station_id': self.station_id,
            'station_name': self.station_name,
            'bbox': self.bbox.copy(),
            'indices': {}
        }

        print(f"\nGrid Indices:")
        for product, spec in GRID_SPECS.items():
            indices = calculate_grid_indices(
                self.bbox['wlon'], self.bbox['elon'],
                self.bbox['slat'], self.bbox['nlat'],
                spec
            )
            results['indices'][product] = indices
            print(f"\n  {product}:")
            print(f"    nlat_idx: {indices['nlat_idx']}")
            print(f"    slat_idx: {indices['slat_idx']}")
            print(f"    wlon_idx: {indices['wlon_idx']}")
            print(f"    elon_idx: {indices['elon_idx']}")

        # Save to JSON file
        output_file = f"{self.station_id}_grid_indices.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_file}")

        # Print CSV format
        print("\n" + "-"*60)
        print("CSV column values (add to station_data.csv):")
        print("-"*60)
        csv_values = {}
        for product, indices in results['indices'].items():
            for key, val in indices.items():
                col_name = f"{product}_{key}"
                csv_values[col_name] = val

        # Print in CSV column order
        for col in sorted(csv_values.keys()):
            print(f"{col},{csv_values[col]}")

    def run(self):
        """Run the interactive selector."""
        self.setup_plot()
        plt.show()


def get_station_info(station_id):
    """Fetch station info from NOAA API."""
    url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json"
    response = requests.get(url, timeout=10)
    data = response.json()

    stations = data.get('stations', [])
    if stations:
        station = stations[0]
        return {
            'name': station.get('name', 'Unknown'),
            'lat': float(station.get('lat', 0)),
            'lon': float(station.get('lng', 0)),
        }
    return None


def load_stations_from_csv(csv_path):
    """Load station info from station_data.csv."""
    stations = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append({
                'station_id': row['station_ID'],
                'name': row['station_name'],
                'bbox': {
                    'wlon': float(row['wlon']),
                    'elon': float(row['elon']),
                    'slat': float(row['slat']),
                    'nlat': float(row['nlat']),
                }
            })
    return stations


def run_single_station(station_id):
    """Run selector for a single station."""
    print(f"Fetching station info for {station_id}...")
    info = get_station_info(station_id)

    if not info:
        print(f"Error: Could not find station {station_id}")
        return

    print(f"Station: {info['name']}")
    print(f"Location: {info['lat']:.4f}, {info['lon']:.4f}")

    selector = BoundingBoxSelector(
        station_id,
        info['name'],
        info['lat'],
        info['lon']
    )
    selector.run()


def run_all_stations():
    """Iterate through all stations in station_data.csv."""
    # Find station_data.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', '0_station_info', 'station_data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)

    # Load stations from CSV
    stations = load_stations_from_csv(csv_path)
    total = len(stations)

    print(f"\nFound {total} stations in station_data.csv")
    print("\nControls:")
    print("  - Drag inside box to move it")
    print("  - Drag edges/corners to resize")
    print("  - 's' = save & go to next station")
    print("  - 'n' = skip to next (no save)")
    print("  - 'r' = reset to default box")
    print("  - 'q' = quit entirely")
    print("\n" + "="*60)

    saved_count = 0
    skipped_count = 0

    for i, station in enumerate(stations):
        station_id = station['station_id']
        station_name = station['name']
        existing_bbox = station['bbox']

        # Fetch coordinates from NOAA API
        print(f"\n[{i+1}/{total}] Fetching info for {station_name} ({station_id})...")
        info = get_station_info(station_id)

        if not info:
            print(f"  Warning: Could not fetch from NOAA API, skipping...")
            skipped_count += 1
            continue

        selector = BoundingBoxSelector(
            station_id,
            station_name,
            info['lat'],
            info['lon'],
            existing_bbox=existing_bbox,
            current_idx=i,
            total_count=total
        )
        selector.run()

        if selector.quit_all:
            print("\nQuitting...")
            break
        elif selector.saved:
            saved_count += 1
            print(f"  Saved! ({saved_count} saved so far)")
        elif selector.skip_to_next:
            skipped_count += 1
            print(f"  Skipped.")

    print("\n" + "="*60)
    print(f"Complete! Saved: {saved_count}, Skipped: {skipped_count}")
    print(f"JSON files saved in: {os.getcwd()}")


def main():
    print("\n" + "="*60)
    print("  BOUNDING BOX SELECTOR FOR GRID INDICES")
    print("="*60)

    if len(sys.argv) >= 2:
        # Single station mode
        station_id = sys.argv[1]
        run_single_station(station_id)
    else:
        # Iterate through all stations
        run_all_stations()


if __name__ == "__main__":
    main()
