import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from plotting_pipelines import curly_brace
from nun_setup import value_calculations, net_testing_value, find_tradeoff_points
from example_with_maximum import exported_parameters

def redraw_plots(base_price, density_function, subsidy, subsidy_vals, planner_values, max_index, biased_update):
    # calculate values based on the current subsidy
    test_value, non_test_value, testing_frac = value_calculations(base_price - subsidy, density_function, biased_update=biased_update)
    
    # impact of subsidy on total value
    axs[0,0].cla()
    axs[0,0].plot(subsidy_vals, planner_values)
    axs[0,0].scatter(subsidy_vals[max_index], planner_values[max_index], color="red", label="optimal subsidy")
    axs[0,0].scatter(subsidy, test_value+non_test_value-testing_frac*subsidy, color="green", label="selected subsidy")
    axs[0,0].set_xlabel("Subsidy per test ($)")
    axs[0,0].set_ylabel("Value/patient - planner costs/patient ($)")
    axs[0,0].legend()
    
    # private testing incentives
    axs[0, 1].cla()
    t_vals = np.linspace(0,1,1000)
    net_testing_vals = [net_testing_value(t) for t in t_vals]
    first_crossing, last_crossing = find_tradeoff_points(base_price)
    subs_first_crossing, subs_last_crossing = find_tradeoff_points(base_price-subsidy)
    axs[0, 1].plot(t_vals, net_testing_vals)

    # unsubsidized
    axs[0,1].hlines(base_price, 0, last_crossing+0.1, color="black")
    axs[0,1].text(last_crossing+0.11, base_price, "Cost of testing", fontsize=8, verticalalignment='center')
    axs[0,1].vlines(first_crossing, 0, base_price, color="black", linestyles="dashed")
    axs[0,1].vlines(last_crossing, 0, base_price, color="black", linestyles="dashed")

    #subsidized
    axs[0,1].hlines(base_price-subsidy, 0, subs_last_crossing, color="black")
    axs[0,1].text(subs_last_crossing+0.01, base_price-subsidy, "Cost subs. testing", fontsize=8, verticalalignment='center')
    axs[0,1].vlines(subs_first_crossing, 0, base_price-subsidy, color="black", linestyles="dashed")
    axs[0,1].vlines(subs_last_crossing, 0, base_price-subsidy, color="black", linestyles="dashed")

    axs[0,1].set_ylim(0, max(net_testing_vals)+1)
    axs[0,1].set_xlim(0,1)
    axs[0,1].set_ylabel("Net benefit of testing ($)")
    axs[0,1].set_xlabel("Patient risk for disease one")
    subscript = "\u209B"
    axs[0,1].set_xticks([0,first_crossing, last_crossing, subs_first_crossing, subs_last_crossing, 1], labels=[0,"L", "U", f"{"L"}{subscript}", f"{"U"}{subscript}", 1])
    
    # breakdown of value between testing and non-testing vs. cost
    axs[1, 0].cla()
    # test_value, non_test_value, testing_frac = value_calculations(base_price-subsidy_vals[max_index], density_function)
    bar_locs = [-0.4, 0.0, 0.4]
    
    axs[1,0].set_xticks(bar_locs, labels=["Value to testers", "Value to non-testers", "Cost to planner"])
    axs[1,0].set_ylabel("Value/patient ($)")
    axs[1,0].bar(bar_locs[0:-1], [test_value, non_test_value], width=0.2)
    axs[1,0].bar([0.4], [subsidy*testing_frac], width=0.2, color="red")

    # population density that will undergo testing
    axs[1, 1].cla()
    fill_t_vals = np.linspace(first_crossing, last_crossing,1000)
    density_values = [density_function(t) for t in t_vals]
    axs[1,1].plot(t_vals, density_values)
    axs[1,1].set_xlim(0,1)
    axs[1,1].set_xticks([0,first_crossing, last_crossing, subs_first_crossing, subs_last_crossing, 1], labels=[0,"L", "U", f"{"L"}{subscript}", f"{"U"}{subscript}", 1])
    axs[1,1].set_xlabel("Patient risk for disease one")
    axs[1,1].set_ylabel("Number of patients")
    axs[1,1].set_yticks(())
    # unsubsidized
    axs[1,1].vlines(first_crossing, 0, density_function(first_crossing), color="black", linestyles="dashed")
    axs[1,1].vlines(last_crossing, 0, density_function(last_crossing), color="black", linestyles="dashed")
    axs[1,1].fill_between(fill_t_vals, [density_function(t) for t in fill_t_vals], alpha=0.3, color="blue")
    midpoint = (last_crossing+first_crossing)/2
    axs[1,1].text(midpoint, density_function(midpoint)/2, f"Population \ndemand for testing", horizontalalignment='center', fontsize=9)
    # subsidized
    subs_fill_t_vals = np.linspace(subs_first_crossing, subs_last_crossing, 1000)
    axs[1,1].vlines(subs_first_crossing, 0, density_function(subs_first_crossing), color="black", linestyles="dashed")
    axs[1,1].vlines(subs_last_crossing, 0, density_function(subs_last_crossing), color="black", linestyles="dashed")
    axs[1,1].fill_between(subs_fill_t_vals, [density_function(t) for t in subs_fill_t_vals], alpha=0.3, color="blue")
    axs[1,1].set_ylim(0,1.1*max(density_values))
    
    plt.draw()

def nun_philipson(base_price, density_function, subsidy_vals=None, biased_update=False):
    if subsidy_vals is None:
        subsidy_vals = np.linspace(0, base_price, 100)

    global fig, axs
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(bottom=0.25)

    planner_values = []
    for subsidy in subsidy_vals:
        test_value, non_test_value, testing_frac = value_calculations(base_price-subsidy, density_function, biased_update=biased_update)
        total_value = test_value + non_test_value - testing_frac*subsidy
        planner_values.append(total_value)
    
    max_index = np.argmax(np.array(planner_values))
    
    # set initial subsidy to be equal to the optimum
    initial_subsidy = subsidy_vals[max_index]
    redraw_plots(base_price, density_function, initial_subsidy, subsidy_vals, planner_values, max_index, biased_update)

    # slider on bottom of screen
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    subsidy_slider = Slider(ax_slider, 'Subsidy', subsidy_vals[0], subsidy_vals[-1], valinit=initial_subsidy)

    # adjust when slider is moved
    subsidy_slider.on_changed(lambda value: redraw_plots(base_price, density_function, value, subsidy_vals, planner_values, max_index, biased_update))

    plt.show()

if __name__ == "__main__":
    base_price = 20
    def edge_cases(t, edge_width=0.1, edge_height=0.1):
        # chosen so that the integral of this function on [0,1] is equal to 1
        mid_height = (1-2*edge_width*edge_height)/(1-2*edge_width)

        if t <= edge_width or t >= 1-edge_width:
            return edge_height
        else:
            return mid_height

    non_uniform = lambda t: edge_cases(t, edge_width=0.1, edge_height=0)
    uniform = lambda t: 1
    non_uniform = lambda t: 6*t*(1-t)

    def triangular(t):
        if 0<=t<=0.5:
            return 4*t
        else:
            return 4*(1-t)

    exported_parameters["cost_res"] *= 1.5
    nun_philipson(base_price, non_uniform)