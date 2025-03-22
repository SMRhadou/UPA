import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString


# Generate random user points efficiently
def generate_random_points(min_distance, max_distance, R):
    locTx = np.random.uniform(0.02*R, (0.98*R), (1, 2))
    phi = 2 * np.pi * np.random.uniform()
    r = np.sqrt(np.random.uniform(low=min_distance ** 2, high=max_distance ** 2, size=(1,)))
    locRx = np.clip(locTx + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= 0, a_max=R)
    return np.concat((locTx, locRx))

# Check if a new link intersects existing ones
def is_valid_link(new_link, existing_links, min_link_distance):
    new_line = LineString(new_link)
    return all(not (new_line.intersects(LineString(link)) or new_line.distance(LineString(link)) < min_link_distance) for link in existing_links)

# Drop links ensuring no interference
def drop_links(num_links, min_distance=10, max_distance=50, R=500, min_link_distance=10, plot=True):
    links = []
    attempts = 0
    while len(links) < num_links and attempts < 1000:  # Limit attempts to prevent infinite loops
        points = generate_random_points(min_distance, max_distance, R)  # Generate two random points
        p1, p2 = tuple(points[0]), tuple(points[1])  # Convert NumPy arrays to tuples

        if np.any(p1 != p2) and is_valid_link((p1, p2), links, min_link_distance):
            links.append(points)
        attempts += 1

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        for link in links:
            x_values, y_values = np.array(link).T  # Transpose to get x and y separately
            ax.plot(x_values, y_values, 'b-')
        ax.set_xlim(0, R)
        ax.set_ylim(0, R)
        ax.set_title("Random Links")
        fig.savefig('links.png')
    return np.stack(links)[:,0], np.stack(links)[:,1]


if __name__ == '__main__':
    # Constants
    AREA_SIZE = 500
    NUM_LINKS = 100
    MIN_DISTANCE = 10
    # Generate non-interfering links
    links = drop_links(NUM_LINKS)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 8))
    for link in links:
        x_values, y_values = np.array(link).T  # Transpose to get x and y separately
        ax.plot(x_values, y_values, 'b-')

    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_title("Random Links")
    fig.savefig('links.png')

    print('ok!')