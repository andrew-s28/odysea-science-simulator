"""Coordinate system conversions.
For several relevant conversions, see https://airsar.jpl.nasa.gov/documents/workshop2002/papers/T3.pdf
"""
import numexpr as ne
import numpy as np


class WGS84:
    """Definition of WGS84 ellipsoid and calculation of local radii."""

    # The Earth's constants
    SEMIMAJOR_AXIS = 6378137.  # in meters

    # SEMIMAJOR_AXIS*sqrt(1-ECCENTRICITY_SQ)
    SEMIMINOR_AXIS = 6356752.3135930374

    ECCENTRICITY_SQ = 0.00669437999015

    # ECCENTRICITY_SQ/(1-ECCENTRICITY_SQ)
    EP_SQUARED = 0.0067394969488402

    CENTER_SCALE = 0.9996

    # Auxiliary Functions
    @staticmethod
    def east_radius(lat):
        """radius of curvature in the east direction"""
        deg2rad = np.pi/180.
        r = ne.evaluate(
            'SEMIMAJOR_AXIS / sqrt(1. - ECCENTRICITY_SQ*sin(lat*deg2rad)**2)',
            local_dict={'lat': lat, 'deg2rad': deg2rad},
            global_dict={'SEMIMAJOR_AXIS': WGS84.SEMIMAJOR_AXIS, 'ECCENTRICITY_SQ': WGS84.ECCENTRICITY_SQ}
        )
        return r

    @staticmethod
    def north_radius(lat):
        """radius of curvature in the north direction"""
        deg2rad = np.pi/180.
        r = ne.evaluate(
            'SEMIMAJOR_AXIS*(1. - ECCENTRICITY_SQ) / (1. - ECCENTRICITY_SQ*sin(lat*deg2rad)**2)**1.5',
            local_dict={'lat': lat, 'deg2rad': deg2rad},
            global_dict={'SEMIMAJOR_AXIS': WGS84.SEMIMAJOR_AXIS, 'ECCENTRICITY_SQ': WGS84.ECCENTRICITY_SQ}
        )
        return r

    @staticmethod
    def local_radius(azimuth, lat):
        """Local radius of curvature along an azimuth direction measured
        clockwise from north. Azimuth in radians
        """
        r = ne.evaluate(
            'east_radius*north_radius / (east_radius*cos(azimuth)**2 + north_radius*sin(azimuth)**2)',
            local_dict={'east_radius': WGS84.east_radius(lat), 'north_radius': WGS84.north_radius(lat), 'azimuth': azimuth}
        )
        return r


def sch_to_llh(s, c, h, peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Transform spherical cross-track height (s, c, h) coordinates to WGS84 (lat, lon, h) coordinates."""
    m, ov = get_sphere_to_ellipsoid_transform_matrices(peg_lat, peg_lon, peg_hdg, peg_local_radius)
    x, y, z = sch_to_xyz(s, c, h, peg_local_radius, m, ov)
    lat, lon, h = xyz_to_llh(x, y, z)
    return lat, lon, h


def sch_to_xyz(s, c, h, peg_local_radius, m, ov):
    """Transform spherical cross-track height (s, c, h) coordinates to geocentric WGS-84 (x, y, z) coordinates."""
    # x = np.full((s.shape[0], c.shape[0]), np.nan)
    # y = np.full((s.shape[0], c.shape[0]), np.nan)
    # z = np.full((s.shape[0], c.shape[0]), np.nan)

    c_lat = np.outer(1/peg_local_radius, c)
    s_lon = np.outer(1/peg_local_radius, s)
    r = np.add.outer(peg_local_radius, h)

    # Get geocentric x, y, z coordinates based on sphere approximation
    x_prime = ne.evaluate(
        'r*cos(c_lat)*cos(s_lon)',
        local_dict={'r': r, 'c_lat': c_lat, 's_lon': s_lon}
    )
    y_prime = ne.evaluate(
        'r*cos(c_lat)*sin(s_lon)',
        local_dict={'r': r, 'c_lat': c_lat, 's_lon': s_lon}
    )
    z_prime = ne.evaluate(
        'r*sin(c_lat)',
        local_dict={'r': r, 'c_lat': c_lat}
    )

    # Apply affine transformation from sphereical geocentric coords to WGS84 ellipsoid geocentric coords
    x = m[:, 0, 0, np.newaxis]*x_prime + m[:, 0, 1, np.newaxis]*y_prime + m[:, 0, 2, np.newaxis]*z_prime + ov[:, 0, np.newaxis]
    y = m[:, 1, 0, np.newaxis]*x_prime + m[:, 1, 1, np.newaxis]*y_prime + m[:, 1, 2, np.newaxis]*z_prime + ov[:, 1, np.newaxis]
    z = m[:, 2, 0, np.newaxis]*x_prime + m[:, 2, 1, np.newaxis]*y_prime + m[:, 2, 2, np.newaxis]*z_prime + ov[:, 2, np.newaxis]

    return x, y, z


def get_sphere_to_ellipsoid_transform_matrices(peg_lat, peg_lon, peg_hdg, peg_local_radius):
    """Get transformation matrices for converting from spherical to ellipsoidal geocentric coordinates."""
    m = np.full((peg_lat.shape[0], 3, 3), np.nan)
    up = np.full((peg_lat.shape[0], 3), np.nan)  # local up vector in geocentric coordinates */

    # Calculate transformation matrix
    deg2rad = np.pi/180.
    clt = ne.evaluate('cos(peg_lat*deg2rad)', local_dict={'peg_lat': peg_lat, 'deg2rad': deg2rad})
    slt = ne.evaluate('sin(peg_lat*deg2rad)', local_dict={'peg_lat': peg_lat, 'deg2rad': deg2rad})
    clo = ne.evaluate('cos(peg_lon*deg2rad)', local_dict={'peg_lon': peg_lon, 'deg2rad': deg2rad})
    slo = ne.evaluate('sin(peg_lon*deg2rad)', local_dict={'peg_lon': peg_lon, 'deg2rad': deg2rad})
    chg = ne.evaluate('cos(peg_hdg*deg2rad)', local_dict={'peg_hdg': peg_hdg, 'deg2rad': deg2rad})
    shg = ne.evaluate('sin(peg_hdg*deg2rad)', local_dict={'peg_hdg': peg_hdg, 'deg2rad': deg2rad})
    m[:, 0, 0] = clt*clo
    m[:, 0, 1] = -shg*slo - slt*clo*chg
    m[:, 0, 2] = slo*chg - slt*clo*shg
    m[:, 1, 0] = clt*slo
    m[:, 1, 1] = clo*shg - slt*slo*chg
    m[:, 1, 2] = -clo*chg - slt*slo*shg
    m[:, 2, 0] = slt
    m[:, 2, 1] = clt*chg
    m[:, 2, 2] = clt*shg

    # Calculate peg point vector from center of ellipsoid to peg point
    p = np.full((peg_lat.shape[0], 3), np.nan)
    east_radius = WGS84.east_radius(peg_lat)
    # displacement vector
    p[:, 0] = east_radius*clt*clo
    p[:, 1] = east_radius*clt*slo
    p[:, 2] = east_radius*(1. - WGS84.ECCENTRICITY_SQ)*slt
    # Calculate the local upward vector in geocentric coordinates
    up[:, 0] = peg_local_radius*clt*clo
    up[:, 1] = peg_local_radius*clt*slo
    up[:, 2] = peg_local_radius*slt
    # Calculate the translation vector for the sch -> xyz transformation
    ov = p - up

    return m, ov


def xyz_to_llh(x, y, z):
    """Transform geocentric WGS-84 (x, y, z) coordinates to spherical WGS-84 (lat, lon, h) coordinates."""
    lon = ne.evaluate('arctan2(y, x)', local_dict={'x': x, 'y': y})
    sa = ne.evaluate(
        'sin(arctan(z/(sqrt(x**2 + y**2)*sqrt(1. - ECCENTRICITY_SQ))))',
        local_dict={'x': x, 'y': y, 'z': z},
        global_dict={'ECCENTRICITY_SQ': WGS84.ECCENTRICITY_SQ})
    ca = ne.evaluate(
        'cos(arctan(z/(sqrt(x**2 + y**2)*sqrt(1. - ECCENTRICITY_SQ))))',
        local_dict={'x': x, 'y': y, 'z': z},
        global_dict={'ECCENTRICITY_SQ': WGS84.ECCENTRICITY_SQ})
    lat = ne.evaluate(
        'arctan((z + EP_SQUARED*SEMIMINOR_AXIS*sa**3) / (sqrt(x**2 + y**2) - ECCENTRICITY_SQ*SEMIMAJOR_AXIS*ca**3))',
        local_dict={'x': x, 'y': y, 'z': z, 'sa': sa, 'ca': ca},
        global_dict={
            'EP_SQUARED': WGS84.EP_SQUARED,
            'SEMIMAJOR_AXIS': WGS84.SEMIMAJOR_AXIS,
            'SEMIMINOR_AXIS': WGS84.SEMIMINOR_AXIS,
            'ECCENTRICITY_SQ': WGS84.ECCENTRICITY_SQ,
        }
    )
    h = ne.evaluate(
        'sqrt(x**2 + y**2)/cos(lat)',
        local_dict={'x': x, 'y': y, 'lat': lat}
    ) - WGS84.east_radius(lat)
    return np.degrees(lat), np.degrees(lon), h
