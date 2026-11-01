import numpy as np
import pyvista as pv

def third_mov_avg(data, resolution = 10, avg_range = 10, current_val_weight = 1):

    def equation(distance, avg_range, current_val_weight):

        result = - current_val_weight / avg_range * distance + current_val_weight

        return result

    x_raw = data[:, 0]
    y_raw = data[:, 1]

    min_x = np.min(x_raw)
    max_x = np.max(x_raw)
    min_y = np.min(y_raw)
    max_y = np.max(y_raw)

    interp_x = np.linspace(min_x, max_x, int(np.round((max_x - min_x) * resolution + 1)))
    interp_y = np.linspace(min_y, max_y, int(np.round((max_y - min_y) * resolution + 1)))

    N = np.size(interp_x) * np.size(interp_y)

    x_vals = np.repeat(interp_x, np.size(interp_y))
    y_vals = np.tile(interp_y, np.size(interp_x))
    z_vals = np.empty(N)

    averaged_array = np.column_stack((x_vals, y_vals, z_vals))
    
    for idx, (x, y, _) in enumerate(averaged_array):

        counter = 0
        added_height = 0

        for x1, y1, z1 in data:

            distance = np.sqrt(np.square(x - x1) + np.square(y - y1))

            if distance < avg_range:

                added_height += equation(distance, avg_range, current_val_weight) * z1
                counter += equation(distance, avg_range, current_val_weight)

        if counter != 0:

            averaged_array[idx, 2] = added_height / counter

        else:

            averaged_array[idx, 2] = 0

    return averaged_array

if __name__ == "__main__":

    """

    data = np.array([
        [1,3,8],
        [1,2,6],
        [2,3,10],
        [0,1,3],
        [2,3,9.7],
        [3,3,18],
        [0,3,1],
        [4,5,9],
        [6,0,4],
        [5,0,3],
        [6,4,8],
        [2,6,6],
        [1,5,7],
        [0,5,3],
        [3,4,10],
        [3,6,4],
        [5,1,7],
        [4,3,12],
        [3.5,2,14],
        [3,0.7,0.9]
    ])

    """

    data = np.array([
        [0.0,0,-0.12],
        [0.1,0,1.84],
        [0.2,0,2.29],
        [0.3,0,3.11],
        [0.4,0,3.96],
        [0.5,0,3.73],
        [0.6,0,4.28],
        [0.7,0,4.12],
        [0.8,0,4.17],
        [0.9,0,4.47],
        [1.0,0,4.7],
        [0.0,0.1,-1.96],
        [0.1,0.1,-0.11],
        [0.2,0.1,0.49],
        [0.3,0.1,1.54],
        [0.4,0.1,1.94],
        [0.5,0.1,2.76],
        [0.6,0.1,2.95],
        [0.7,0.1,2.8],
        [0.8,0.1,3.14],
        [0.9,0.1,2.84],
        [1.0,0.1,2.62],
        [0.0,0.2,-2.07],
        [0.1,0.2,-0.37],
        [0.2,0.2,0.26],
        [0.3,0.2,0.98],
        [0.4,0.2,1.86],
        [0.5,0.2,1.45],
        [0.6,0.2,1.86],
        [0.7,0.2,1.91],
        [0.8,0.2,1.34],
        [0.9,0.2,0.87],
        [1.0,0.2,0.85],
        [0.0,0.3,-2.66],
        [0.1,0.3,-1.26],
        [0.2,0.3,-0.86],
        [0.3,0.3,0.12],
        [0.4,0.3,0.63],
        [0.5,0.3,0.71],
        [0.6,0.3,0.81],
        [0.7,0.3,0.54],
        [0.8,0.3,0.04],
        [0.9,0.3,0],
        [1.0,0.3,-0.87],
        [0.0,0.4,-3.31],
        [0.1,0.4,-2.32],
        [0.2,0.4,-1.45],
        [0.3,0.4,-0.7],
        [0.4,0.4,-0.32],
        [0.5,0.4,0.09],
        [0.6,0.4,0.08],
        [0.7,0.4,0.07],
        [0.8,0.4,-0.91],
        [0.9,0.4,-1.29],
        [1.0,0.4,-2],
        [0.0,0.5,-3.81],
        [0.1,0.5,-2.7],
        [0.2,0.5,-1.52],
        [0.3,0.5,-0.53],
        [0.4,0.5,0.02],
        [0.5,0.5,-0.11],
        [0.6,0.5,0],
        [0.7,0.5,-0.68],
        [0.8,0.5,-1.3],
        [0.9,0.5,-1.62],
        [1.0,0.5,-2.68],
        [0.0,0.6,-4.14],
        [0.1,0.6,-2.92],
        [0.2,0.6,-1.72],
        [0.3,0.6,-0.89],
        [0.4,0.6,-0.24],
        [0.5,0.6,-0.11],
        [0.6,0.6,-0.12],
        [0.7,0.6,-0.34],
        [0.8,0.6,-1.25],
        [0.9,0.6,-2.11],
        [1.0,0.6,-2.32],
        [0.0,0.7,-4.02],
        [0.1,0.7,-3.06],
        [0.2,0.7,-1.37],
        [0.3,0.7,-0.55],
        [0.4,0.7,-0.38],
        [0.5,0.7,0.26],
        [0.6,0.7,0.55],
        [0.7,0.7,-0.12],
        [0.8,0.7,-0.37],
        [0.9,0.7,-1.5],
        [1.0,0.7,-1.36],
        [0.0,0.8,-4.67],
        [0.1,0.8,-3.02],
        [0.2,0.8,-1.35],
        [0.3,0.8,0.11],
        [0.4,0.8,0.75],
        [0.5,0.8,0.87],
        [0.6,0.8,0.68],
        [0.7,0.8,0.4],
        [0.8,0.8,-0.03],
        [0.9,0.8,-0.61],
        [1.0,0.8,-1.63],
        [0.0,0.9,-4.57],
        [0.1,0.9,-2.48],
        [0.2,0.9,-0.48],
        [0.3,0.9,0.25],
        [0.4,0.9,1.65],
        [0.5,0.9,1.56],
        [0.6,0.9,2.07],
        [0.7,0.9,0.94],
        [0.8,0.9,0.79],
        [0.9,0.9,0.04],
        [1.0,0.9,-1.56],
        [0.0,1,-4.88],
        [0.1,1,-3.01],
        [0.2,1,-0.66],
        [0.3,1,0.7],
        [0.4,1,2.39],
        [0.5,1,2.78],
        [0.6,1,2.85],
        [0.7,1,2.31],
        [0.8,1,1.94],
        [0.9,1,1.03],
        [1.0,1,-0.07],
    ])

    #heat_map = True # If true, plot a heat map instead of a 3d graph
    original_points = True # If true, show the original points which created the average
    resolution = 200 # This number's square will be the total number of points in a single square
    avg_range = 0.2 # The range which a point takes its info from, anything outside of this range (circular / radius) will have no effect on this point
    plot_mesh = False # If true, the plot will *try* to form a solid body for the averaged array. "Sometimes maybe good, sometimes maybe sheet!"
    heat_map = False # If true, plot a heat map instead of a 3d graph :)

    averaged_array = third_mov_avg(data, resolution, avg_range)

    if not heat_map:
        pv_raw_array = pv.PolyData(data)
        pv_averaged_array = pv.PolyData(averaged_array)

        mesh = pv_averaged_array.delaunay_2d()

        plt = pv.Plotter()

        if original_points:
            plt.add_points(pv_raw_array, scalars = data[:, 2], cmap = 'reds', point_size = 10, label = "Original Data")

        if plot_mesh == True:
            plt.add_mesh(mesh, scalars = averaged_array[:, 2], cmap = 'viridis', show_edges = False, label = "Averaged Data")

        else:
            plt.add_points(mesh, scalars = averaged_array[:, 2], cmap = 'viridis', label = "Averaged Data")

        if original_points:
            legend_entries = [
                ["Original Data", "red"],
                ["Averaged Data", "blue"]
            ]

        else:
            legend_entries = [
                ["Averaged Data", "blue"]
            ]

        pv_raw_array.scale((1, 1, 0.2), inplace=True)
        mesh.scale((1, 1, 0.2), inplace=True)

        # labels = dict(
        #     xlabel="Player 1 Confidence Level (0-1)", ylabel="Player 2 Confidence Level (0-1)", zlabel="Average Score by Point Difference (Positive - P1 Wins, Negative - P2 Wins)"
        # )

        # plt.show_grid(**labels)
        # plt.add_axes(**labels)

        plt.add_legend(legend_entries)
        plt.add_axes()
        plt.show_grid()

        plt.show()

    else:
        import matplotlib.pyplot as plt

        plt.figure()

        scatter = plt.scatter(averaged_array[:,0], averaged_array[:,1], c = averaged_array[:,2], cmap = 'viridis', s = 30)

        bar = plt.colorbar(scatter)

        plt.title('Player Win Rate by Point Final Point Value (negative indicates p2 wins instead)', fontsize=12, pad=10)
        plt.xlabel("Player 1 Confidence Level (0-1)")
        plt.ylabel("Player 2 Confidence Level (0-1)")
        plt.show()