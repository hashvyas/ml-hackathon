"""Per-trip projected geometry helpers."""

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class TripProjection:
    """A local UTM projection for one trip."""

    forward: Transformer
    inverse: Transformer

    @classmethod
    def from_coordinates(cls, lon: np.ndarray, lat: np.ndarray) -> "TripProjection":
        if len(lon) == 0:
            raise ValueError("At least one coordinate is required to select a projection")
        center_lon = float(np.nanmean(lon))
        center_lat = float(np.nanmean(lat))
        zone = int((center_lon + 180.0) // 6.0) + 1
        epsg = (32600 if center_lat >= 0 else 32700) + zone
        crs = CRS.from_epsg(epsg)
        return cls(
            Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        )

    def project(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.forward.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def unproject(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self.inverse.transform(x, y)
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def point_to_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and fractional offset for each point-segment pair."""
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    relative = points - starts
    safe_lengths_sq = np.where(lengths_sq > 0.0, lengths_sq, 1.0)
    offset = np.einsum("ij,ij->i", relative, vectors) / safe_lengths_sq
    offset = np.clip(offset, 0.0, 1.0)
    closest = starts + offset[:, None] * vectors
    distance = np.linalg.norm(points - closest, axis=1)
    return distance, offset
