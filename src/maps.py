from folium.plugins import HeatMap
from folium import Map, Marker, CircleMarker
from branca.colormap import LinearColormap
from pandas import DataFrame, concat
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


def map_points_circles(df, feature, date):

     print(feature.title(),'at time:' , date)

     # Create a map
     this_map = Map(prefer_canvas=True)

     # Check for the inputs to be on the dataframe
     # Getting data
     data = df[df['datetime'] == date]

     # Create a color map
     color_var = str(feature) #what variable will determine the color
     cmap = LinearColormap(['blue', 'red'],
                              vmin=data[color_var].quantile(0.05), vmax=data[color_var].quantile(0.95),
                              caption = color_var)

     # Add the color map legend to your map
     this_map.add_child(cmap)

     def plotDot(point):

         '''Takes a series that contains a list of floats named latitude and longitude and
         this function creates a CircleMarker and adds it to your this_map'''

         CircleMarker(location=[point.latitude, point.longitude],
                        fill_color=cmap(point[color_var]),
                            fill=True,
                            fill_opacity=0.4,
                            radius=40,
                         weight=0).add_to(this_map)

            # Iterate over rows
     data.apply(plotDot, axis=1)

     # Set the boundaries
     this_map.fit_bounds(this_map.get_bounds())

     # Show plot
     return this_map



def map_points_series(df, feature_name):
     map_time = Map(location=[df.latitude.values[0],df.longitude.values[0]], zoom_start=11)
     df['latitude'] = df['latitude'].astype(float)
     df['longitude'] = df['longitude'].astype(float)
     assert feature_name in df.columns
     # map_df = df[['latitude','longitude',str(feature_name)]]
     dataframe = DataFrame(columns=['latitude','longitude',feature_name])
     station_names = df.station.columns
     for station in station_names:
         print(station)
         dataframe = concat([dataframe,
                             df[df['station'] == station]\
                 .reset_index().set_index('datetime')\
                 .resample('M').mean()\
                 .reset_index()[['latitude','longitude',str(feature_name)]]],
                             axis=0)
     return map_time
