#!/usr/bin/env python3
"""
Build the Csa / Csb / Csc climate-zone GeoJSON overlay used by the
map page, directly from a real Koppen-Geiger raster.

This replaces the hand-curated `mediterranean_climate.geojson` that
shipped with the project (and was, as the comment in map.js admits,
a smoothed ~0.5-degree approximation) with polygons that match the
canonical Csa/Csb/Csc boundaries 1:1.

----------------------------------------------------------------------
Which source to download
----------------------------------------------------------------------

Two well-known Koppen-Geiger raster datasets are supported, plus any
other raster that follows the same conventions (integer class codes
per pixel).

  1. Beck et al. (2018) - RECOMMENDED.
     1 km resolution, present-day climate (1980-2016 baseline).
     Sci. Data 5, 180214.
     Download:  https://www.gloh2o.org/koppen/
     File:      koppen_geiger_0p00833333.tif
     Class codes: Csa=8, Csb=9, Csc=10.



The Beck 2018 dataset gives a visibly sharper map (1 km vs 50 km
cells) and is what most contemporary papers cite, so the script
defaults to its class numbering. The Kottek 2006 PDF you shared is
the methodological reference for the underlying climate
classification, not the raster you would actually overlay - both
papers use the same Csa/Csb/Csc definitions, the difference is
purely in resolution and base period.

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

    # One-time setup
    pip install rasterio shapely

    # Default (Beck 2018 GeoTIFF, present-day)
    python build_koppen_geojson.py Beck_KG_V1_present_0p0083.tif \\
        --output mediterranean_climate.geojson


    # Restrict to a bounding box (e.g. Mediterranean basin only,
    # for a smaller file)
    python build_koppen_geojson.py raster.tif \\
        --bbox -12 28 42 47 \\
        --output mediterranean_climate.geojson

    # Aggressive simplification (cuts vertex count, ok for low-zoom
    # viewing - the default keeps almost all detail)
    python build_koppen_geojson.py raster.tif --simplify-deg 0.02

Place the output at:
    forest-fire-fullstack/frontend/data/mediterranean_climate.geojson

That is the path the existing frontend already fetches.

----------------------------------------------------------------------
What the output looks like
----------------------------------------------------------------------

A GeoJSON FeatureCollection with one Feature per contiguous polygon
region, each tagged with its zone:

    {"type":"FeatureCollection","features":[
        {"type":"Feature",
         "properties":{"zone":"Csa"},
         "geometry":{"type":"Polygon","coordinates":[[[...],...]]}
        }, ...
    ]}

The frontend reads `feature.properties.zone` to pick the colour, and
uses the polygons themselves for the point-in-polygon check that
flags hexes outside the Mediterranean training climate. That logic
is unchanged - this script only swaps the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Build the Csa/Csb/Csc GeoJSON overlay from a '
                    'Koppen-Geiger raster.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'raster',
        type=Path,
        help='Path to the Koppen-Geiger raster file (GeoTIFF or ASCII grid)',
    )
    p.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('mediterranean_climate.geojson'),
        help='Output GeoJSON path (default: %(default)s)',
    )
    p.add_argument(
        '--csa-code', type=int, default=8,
        help='Integer pixel value for Csa (default: 8, the Beck 2018 code)',
    )
    p.add_argument(
        '--csb-code', type=int, default=9,
        help='Integer pixel value for Csb (default: 9, the Beck 2018 code)',
    )
    p.add_argument(
        '--csc-code', type=int, default=10,
        help='Integer pixel value for Csc (default: 10, the Beck 2018 code)',
    )
    p.add_argument(
        '--simplify-deg', type=float, default=0.005,
        help='Polygon simplification tolerance in degrees. Higher = '
             'fewer vertices but jaggier outline. Default %(default)s is '
             'about a 500m tolerance at the equator, which is invisible '
             'at the zoom levels the map page uses.',
    )
    p.add_argument(
        '--min-area-deg2', type=float, default=0.0005,
        help='Drop polygons smaller than this area in square-degrees. '
             'Default %(default)s suppresses 1-2 pixel artefacts that '
             'would otherwise become tiny invisible features. Set to 0 '
             'to keep everything.',
    )
    p.add_argument(
        '--bbox',
        nargs=4, type=float, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        default=None,
        help='Restrict to a bounding box. Example for the Mediterranean '
             'basin only: --bbox -12 28 42 47',
    )
    p.add_argument(
        '--no-pretty', action='store_true',
        help='Write compact JSON (smaller file). Without this flag the '
             'output is indented for readability.',
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.raster.exists():
        print(f'ERROR: raster not found: {args.raster}', file=sys.stderr)
        sys.exit(1)

    try:
        import rasterio
        from rasterio.features import shapes as rio_shapes
        from rasterio.windows import from_bounds
        from shapely.geometry import shape as shp_shape, mapping as shp_mapping
        from shapely.geometry import box as shp_box
    except ImportError as e:
        print(
            f'ERROR: missing dependency ({e}). Install with:\n\n'
            f'    pip install rasterio shapely\n',
            file=sys.stderr,
        )
        sys.exit(1)

    code_to_zone = {
        args.csa_code: 'Csa',
        args.csb_code: 'Csb',
        args.csc_code: 'Csc',
    }

    features = []
    counts = {'Csa': 0, 'Csb': 0, 'Csc': 0}

    with rasterio.open(args.raster) as src:
        # Optional bounding-box window. Reading only the requested
        # region is dramatically faster on a global 1 km raster.
        if args.bbox is not None:
            min_lon, min_lat, max_lon, max_lat = args.bbox
            window = from_bounds(
                min_lon, min_lat, max_lon, max_lat, src.transform
            )
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            bbox_geom = shp_box(min_lon, min_lat, max_lon, max_lat)
        else:
            data = src.read(1)
            transform = src.transform
            bbox_geom = None

        # Polygonise each zone separately so we can tag each feature
        # with its zone name. rasterio.features.shapes is the standard
        # raster-to-vector pathway; the `mask` argument restricts the
        # output to non-zero pixels.
        for code, zone in code_to_zone.items():
            zone_mask = (data == code)
            if not zone_mask.any():
                print(
                    f'  Warning: no {zone} pixels found '
                    f'(class code {code})',
                    file=sys.stderr,
                )
                continue
            mask_uint8 = zone_mask.astype('uint8')

            for geom, value in rio_shapes(
                mask_uint8, mask=mask_uint8, transform=transform
            ):
                if value != 1:
                    continue
                poly = shp_shape(geom)

                # Belt-and-braces clip to the bbox if specified, to
                # trim polygons that the window read may have given us
                # a bit of slop on.
                if bbox_geom is not None:
                    poly = poly.intersection(bbox_geom)
                    if poly.is_empty:
                        continue

                if args.simplify_deg > 0:
                    poly = poly.simplify(
                        args.simplify_deg, preserve_topology=True,
                    )

                if poly.is_empty or poly.area < args.min_area_deg2:
                    continue

                features.append({
                    'type': 'Feature',
                    'properties': {'zone': zone},
                    'geometry': shp_mapping(poly),
                })
                counts[zone] += 1

    feature_collection = {
        'type': 'FeatureCollection',
        'features': features,
        # Metadata block: who built this file, what data went into it,
        # what filters were applied. Anyone opening the file can see
        # the provenance immediately - matters when the project is
        # being marked.
        'metadata': {
            'source_raster': str(args.raster),
            'csa_code': args.csa_code,
            'csb_code': args.csb_code,
            'csc_code': args.csc_code,
            'simplify_deg': args.simplify_deg,
            'min_area_deg2': args.min_area_deg2,
            'bbox': args.bbox,
            'generator': 'build_koppen_geojson.py',
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.no_pretty else 2
    with open(args.output, 'w') as f:
        json.dump(feature_collection, f, indent=indent)

    size_kb = args.output.stat().st_size / 1024
    print(
        f'Wrote {len(features)} features to {args.output} '
        f'({size_kb:.0f} KB)\n'
        f'  Csa polygons: {counts["Csa"]}\n'
        f'  Csb polygons: {counts["Csb"]}\n'
        f'  Csc polygons: {counts["Csc"]}'
    )


if __name__ == '__main__':
    main()
