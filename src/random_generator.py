import random
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import csv

def plot_values(random_values, start, end, save = False):
    # Create the x-axis values
    x_values = list(range(len(random_values)))

    # Plot the random values
    plt.plot(x_values, random_values, marker='o')

    # Set the x-axis limits
    plt.xlim(start, end)

    # Set the x-axis tick labels
    plt.xticks(x_values)

    # Set the axis labels
    plt.xlabel('Index')
    plt.ylabel('Random Value')


    # Save the figure as an image file
    if save:
        plt.savefig('random_values_plot.png')

    # Close the plot
    plt.close()



def random_times(atk_params, filename):
    start_time = atk_params['start'].timestamp()+atk_params['wave_freq']+10
    end_time = (atk_params['start'] + timedelta(minutes=atk_params['duration'])).timestamp()

    times = sorted([random.uniform(0,end_time-start_time) for _ in range(atk_params['nb_benign'])])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Time:{atk_params["start"].timestamp()}'])
        writer.writerows([[time] for time in times])

    print(f"Random times saved to {filename}")


    return times


if __name__ == "__main__":
       
    atk_params = {
        'start': datetime.now(timezone.utc),
        'duration': 5,                  # in minutes
        'collection_extra_time': 2,     # in minutes
        'collection_step': 1,           # in seconds
        'wave_freq': 60,                # in seconds
        'cooldown': 1,                  # in seconds
        'nb_ues': 30,
        'nb_benign': 220,
        'deployments': 1,       
        'restart_pod': True,           
        'deregistration': 'normal',
        'integrity_protection': '',
        'ciphering_algorithm': ''
    }
    times = random_times(atk_params, 'random_times.csv')
    start_time = atk_params['start'].timestamp()
    end_time = (atk_params['start'] + timedelta(minutes=atk_params['duration'])).timestamp()
    plot_values(times, start_time, end_time)
    print(times)
