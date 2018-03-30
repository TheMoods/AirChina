from folium.plugins import HeatMap
from folium import Map, Marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def map_points(df, lat_col='latitude', lon_col='longitude', zoom_start=11,
               plot_points=False, pt_radius=15,
               draw_heatmap=False, heat_map_weights_col=None,
               heat_map_weights_normalize=True, heat_map_radius=15):
    """Creates a map given a dataframe of points. Can also produce a
    heatmap overlay
    Arguments:
        df: dataframe containing points to maps
        lat_col: Column containing latitude (string)
        lon_col: Column containing longitude (string)
        zoom_start: Integer representing the initial zoom of the map
        plot_points: Add points to map (boolean)
        pt_radius: Size of each point
        draw_heatmap: Add heatmap to map (boolean)
        heat_map_weights_col: Column containing heatmap weights
        heat_map_weights_normalize: Normalize heatmap weights (boolean)
        heat_map_radius: Size of heatmap point

    Returns:
        folium map object
    """

    #  center map in the middle of points
    middle_lat = df[lat_col].mean() + 0.2
    middle_lon = df[lon_col].mean()

    curr_map = Map(location=[middle_lat, middle_lon],
                   tiles='Cartodb Positron',
                   zoom_start=zoom_start)

    # add points to map
    if plot_points:
        for _, row in df.iterrows():
            Marker([row[lat_col], row[lon_col]],
                   popup=row['station']).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            # cols_to_pull = [lat_col, lon_col]
            heat_df = [[row[lat_col], row[lon_col]]
                       for index, row in df.iterrows()]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                df[heat_map_weights_col] = \
                    df[heat_map_weights_col] / df[heat_map_weights_col].sum()

            heat_df = [[row[lat_col], row[lon_col], row[heat_map_weights_col]]
                       for index, row in df.iterrows()]

        HeatMap(heat_df).add_to(curr_map)

    return curr_map


def distance_on_sphere_numpy(coordinate_array):
    """
    Compute a distance matrix of the coordinates using a spherical metric.
    :param coordinate_array: numpy.ndarray with shape (n,2);
        latitude is in 1st col, longitude in 2nd.
    :returns distance_mat: numpy.ndarray with shape (n, n)
        containing distance in km between coords.
    """

    # Radius of the earth in km (GRS 80-Ellipsoid)
    EARTH_RADIUS = 6371.007176
    latitudes = coordinate_array.iloc[:, 1]
    longitudes = coordinate_array.iloc[:, 2]

    # convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
    phi_values = (90.0 - latitudes)*degrees_to_radians
    theta_values = longitudes*degrees_to_radians

    # expand phi_values and theta_values into grids
    theta_1, theta_2 = np.meshgrid(theta_values, theta_values)
    theta_diff_mat = theta_1 - theta_2
    phi_1, phi_2 = np.meshgrid(phi_values, phi_values)

    # compute spherical distance from spherical coordinates
    angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) +
             np.cos(phi_1) * np.cos(phi_2))
    arc = np.arccos(angle)

    # Multiply by earth's radius to obtain distance in km
    return arc * EARTH_RADIUS


def plot_dist_stations(df):
    dists = distance_on_sphere_numpy(df)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(dists, interpolation='None')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.xaxis.tick_top()
    ax.set_xticklabels(df['station'], minor=False)
    ax.set_yticklabels(df['station'], minor=False)
    plt.show()
