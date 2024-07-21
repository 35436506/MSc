import random
from openpyxl import Workbook
import matplotlib.pyplot as plt
import math
import numpy as np

# Set seed for reproducibility
random.seed(42)

# Define the recipes as a list from 1 to 65
recipes = list(range(1, 66))

# Define the recipe eligibility for each factory
factory_eligibility = {
    'F1': random.sample(recipes, int(0.3 * len(recipes))),  # 30% of recipes
    'F2': random.sample(recipes, int(0.6 * len(recipes))),  # 60% of recipes
    'F3': recipes  # All recipes
}

# Define total orders and factory capacities
total_orders = 10000
F1_cap = int(0.3 * total_orders)  # 30% of total orders
F2_cap = int(0.5 * total_orders)  # 50% of total orders
factory_capacities = {
    'F1': F1_cap,
    'F2': F2_cap,
    'F3': float('inf')  # F3 has unlimited capacity
}

def generate_orders(num_orders, is_real=True, factory=None, prioritize_f1_f2=False):
    orders = []
    while len(orders) < num_orders:
        if prioritize_f1_f2:
            if factory == 'F1':
                recipe_ids = random.sample(factory_eligibility['F1'], random.randint(1, 4))
            elif factory == 'F2':
                recipe_ids = random.sample(factory_eligibility['F2'], random.randint(1, 4))
            else:
                recipe_ids = random.sample(recipes, random.randint(1, 4))
        else:
            recipe_ids = random.sample(recipes, random.randint(1, 4))

        if factory is None or all(recipe_id in factory_eligibility[factory] for recipe_id in recipe_ids):
            order = {'id': len(orders) + 1, 'recipe_ids': recipe_ids, 'is_real': is_real}
            order['eligible_factories'] = get_eligible_factories(order)
            orders.append(order)
    return orders

def get_eligible_factories(order):
    return [factory for factory, eligible_recipes in factory_eligibility.items()
            if all(recipe_id in eligible_recipes for recipe_id in order['recipe_ids'])]

def generate_orders_for_day(day, total_orders, previous_real_orders=None):
    if day >= -3:
        real_proportion = 1.0
    elif day <= -18:
        real_proportion = 0.1
    else:
        real_proportion = 0.1 + (0.9 / 15) * (18 + day)
    
    num_real = int(real_proportion * total_orders)
    
    if previous_real_orders:
        orders = previous_real_orders.copy()
        additional_real_orders = num_real - len(previous_real_orders)
        if additional_real_orders > 0:
            orders.extend(generate_orders(additional_real_orders, is_real=True))
    else:
        orders = generate_orders(num_real, is_real=True)
    
    num_simulated = total_orders - len(orders)
    orders.extend(generate_orders(num_simulated, is_real=False))
    
    for factory in ['F1', 'F2']:
        eligible_orders = [order for order in orders if factory in order['eligible_factories']]
        while len(eligible_orders) < factory_capacities[factory]:
            new_orders = generate_orders(factory_capacities[factory] - len(eligible_orders), 
                                         is_real=False, factory=factory, prioritize_f1_f2=True)
            orders.extend(new_orders)
            eligible_orders.extend(new_orders)
    
    for i, order in enumerate(orders):
        order['id'] = i + 1
    
    return orders

def allocate_orders(orders, factory_capacities):
    allocation = {factory: [] for factory in factory_capacities}
    remaining_orders = orders.copy()

    for factory in ['F1', 'F2', 'F3']:
        capacity = factory_capacities[factory]
        eligible_orders = [order for order in remaining_orders if factory in order['eligible_factories']]
        eligible_orders.sort(key=lambda x: x['is_real'], reverse=True)  # Prioritize real orders
        
        if factory == 'F3':
            allocated_orders = eligible_orders  # Allocate all remaining orders to F3
        else:
            allocated_orders = eligible_orders[:int(capacity)]  # Convert capacity to int for slicing
        
        allocation[factory].extend(allocated_orders)
        remaining_orders = [order for order in remaining_orders if order not in allocated_orders]

    return allocation

def calculate_wmape_site(allocation_t_minus_1, allocation_t, sheet_wmape):
    total_abs_diff = 0
    total_items_t = 0
    
    for factory in ['F1', 'F2', 'F3']:
        recipe_counts_t_minus_1 = {}
        recipe_counts_t = {}
        
        for order in allocation_t_minus_1[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t_minus_1[recipe_id] = recipe_counts_t_minus_1.get(recipe_id, 0) + 1
        
        for order in allocation_t[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t[recipe_id] = recipe_counts_t.get(recipe_id, 0) + 1
                total_items_t += 1
        
        for recipe_id in set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys()):
            t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
            t_count = recipe_counts_t.get(recipe_id, 0)
            abs_diff = abs(t_count - t_minus_1_count)
            total_abs_diff += abs_diff
    
    if total_items_t == 0:
        wmape_site = float('inf')
    else:
        wmape_site = total_abs_diff / total_items_t
    
    return wmape_site

def calculate_wmape_global(allocation_t_minus_1, allocation_t, sheet_wmape):
    total_abs_diff = 0
    total_t_items = 0
    recipe_counts_t_minus_1 = {}
    recipe_counts_t = {}
    for factory in allocation_t_minus_1:
        for order in allocation_t_minus_1[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t_minus_1[recipe_id] = recipe_counts_t_minus_1.get(recipe_id, 0) + 1
    for factory in allocation_t:
        for order in allocation_t[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t[recipe_id] = recipe_counts_t.get(recipe_id, 0) + 1
                total_t_items += 1
    for recipe_id in set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys()):
        t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
        t_count = recipe_counts_t.get(recipe_id, 0)
        abs_diff = abs(t_minus_1_count - t_count)
        total_abs_diff += abs_diff
    wmape_global = total_abs_diff / total_t_items
    return wmape_global

def simulated_annealing_with_swap(allocation_t_minus_1, allocation_t, factory_capacities, initial_temp=1000, cooling_rate=0.995, iterations=10000):
    current_allocation = {factory: orders[:] for factory, orders in allocation_t.items()}
    current_wmape_site = calculate_wmape_site(allocation_t_minus_1, current_allocation, [])
    best_allocation = current_allocation.copy()
    best_wmape_site = current_wmape_site
    temp = initial_temp

    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1)

    for i in range(iterations):
        factory1, factory2 = random.sample(list(factory_capacities.keys()), 2)
        if current_allocation[factory1] and current_allocation[factory2]:
            idx1 = random.randint(0, len(current_allocation[factory1]) - 1)
            idx2 = random.randint(0, len(current_allocation[factory2]) - 1)
            order1 = current_allocation[factory1][idx1]
            order2 = current_allocation[factory2][idx2]
            
            if factory2 in order1['eligible_factories'] and factory1 in order2['eligible_factories']:
                new_allocation = {f: orders[:] for f, orders in current_allocation.items()}
                new_allocation[factory1][idx1], new_allocation[factory2][idx2] = order2, order1
                
                new_wmape_site = calculate_wmape_site(allocation_t_minus_1, new_allocation, [])
                
                if new_wmape_site < current_wmape_site or random.random() < math.exp((current_wmape_site - new_wmape_site) / temp):
                    current_allocation = new_allocation
                    current_wmape_site = new_wmape_site
                    if current_wmape_site < best_wmape_site:
                        best_wmape_site = current_wmape_site
                        best_allocation = {f: orders[:] for f, orders in current_allocation.items()}
        
        # Local search
        if i % 100 == 0:
            current_allocation = local_search(allocation_t_minus_1, current_allocation, factory_capacities, recipe_counts_t_minus_1)
            current_wmape_site = calculate_wmape_site(allocation_t_minus_1, current_allocation, [])
            if current_wmape_site < best_wmape_site:
                best_wmape_site = current_wmape_site
                best_allocation = current_allocation.copy()

        temp *= cooling_rate

    return best_allocation, best_wmape_site

def get_recipe_counts(allocation):
    recipe_counts = {}
    for factory in allocation:
        for order in allocation[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts[recipe_id] = recipe_counts.get(recipe_id, 0) + 1
    return recipe_counts

def local_search(allocation_t_minus_1, current_allocation, factory_capacities, recipe_counts_t_minus_1):
    recipe_counts_current = get_recipe_counts(current_allocation)
    
    for l in range(100):  # Perform 100 local search iterations
        factory1, factory2 = random.sample(list(factory_capacities.keys()), 2)
        if current_allocation[factory1] and current_allocation[factory2]:
            order1 = max(current_allocation[factory1], key=lambda o: sum(abs(recipe_counts_current.get(r, 0) - recipe_counts_t_minus_1.get(r, 0)) for r in o['recipe_ids']))
            order2 = max(current_allocation[factory2], key=lambda o: sum(abs(recipe_counts_current.get(r, 0) - recipe_counts_t_minus_1.get(r, 0)) for r in o['recipe_ids']))
            
            if factory2 in order1['eligible_factories'] and factory1 in order2['eligible_factories']:
                idx1, idx2 = current_allocation[factory1].index(order1), current_allocation[factory2].index(order2)
                current_allocation[factory1][idx1], current_allocation[factory2][idx2] = order2, order1
                recipe_counts_current = get_recipe_counts(current_allocation)
    
    return current_allocation

def run_allocation_process_over_time(start_day, end_day, total_orders):
    allocations = {}
    wmape_site_values = []
    wmape_global_values = []
    real_orders_proportions = []
    previous_real_orders = None
    
    for day in range(start_day, end_day + 1):
        orders = generate_orders_for_day(day, total_orders, previous_real_orders)
        allocation = allocate_orders(orders, factory_capacities)
        
        if day > start_day:
            # Apply Simulated Annealing optimization
            optimized_allocation, _ = simulated_annealing_with_swap(allocations[day-1], allocation, factory_capacities)
            allocation = optimized_allocation
            
            wmape_site = calculate_wmape_site(allocations[day-1], allocation, [])
            wmape_global = calculate_wmape_global(allocations[day-1], allocation, [])
            wmape_site_values.append(wmape_site)
            wmape_global_values.append(wmape_global)
        
        allocations[day] = allocation
        real_orders_proportions.append(len([o for o in orders if o['is_real']]) / total_orders)
        
        # Update previous_real_orders for the next iteration
        previous_real_orders = [order for order in orders if order['is_real']]
    
    return allocations, wmape_site_values, wmape_global_values, real_orders_proportions

def plot_temporal_component(days, real_orders_proportions):
    plt.figure(figsize=(15, 8))
    
    # Calculate real and simulated orders for each day
    morning_real = []
    afternoon_real = []
    morning_simu = []
    afternoon_simu = []
    
    for day, prop in zip(days, real_orders_proportions):
        total_real = int(prop * total_orders)
        
        # Morning allocation
        if day == -3:
            morning_real_count = int(total_real * 0.95)  # 95% of total real orders in the morning on the last day
        else:
            morning_real_count = int(total_real * 0.90)  # 90% of total real orders in the morning for other days
        
        # Afternoon allocation (slightly higher than morning)
        afternoon_real_count = total_real
        
        morning_real.append(morning_real_count)
        afternoon_real.append(afternoon_real_count)
        morning_simu.append(total_orders - morning_real_count)
        afternoon_simu.append(total_orders - afternoon_real_count)
    
    # Create x-axis positions
    x = np.arange(len(days))
    width = 0.35
    
    # Plot bars
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width/2, morning_real, width, label='Morning Real', color='lightblue')
    ax.bar(x - width/2, morning_simu, width, bottom=morning_real, label='Morning Simulated', color='gold')
    ax.bar(x + width/2, afternoon_real, width, label='Afternoon Real', color='blue')
    ax.bar(x + width/2, afternoon_simu, width, bottom=afternoon_real, label='Afternoon Simulated', color='orange')
    
    # Customize the plot
    ax.set_xlabel('Days to delivery', fontsize=12)
    ax.set_ylabel('Order quantity', fontsize=12)
    ax.set_title('Temporal component of order allocation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(day) for day in days], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Set y-axis limit
    ax.set_ylim(0, 1.1 * total_orders)
    
    # Add vertical lines and annotations
    ax.axvline(x=16, color='purple', linestyle='--', linewidth=1)  # -2 corresponds to index 16
    ax.text(16, ax.get_ylim()[1], 'Delivery date', va='bottom', ha='left', color='purple')
    ax.axvline(x=-1, color='purple', linestyle=':', linewidth=1)  # -19 corresponds to index -1
    ax.text(-1, ax.get_ylim()[1], 'Menu opens', va='bottom', ha='left', color='purple')
    
    # Move legend outside of the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_wmape_over_time(days, wmape_site_values, wmape_global_values):
    plt.figure(figsize=(12, 6))
    plt.plot(days[1:], wmape_site_values, label='WMAPE site', marker='o')
    plt.plot(days[1:], wmape_global_values, label='WMAPE global', marker='s')
    
    plt.xlabel('Days to delivery')
    plt.ylabel('WMAPE')
    plt.title('WMAPE site and global over time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution code
start_day = -18
end_day = -3

allocations, wmape_site_values, wmape_global_values, real_orders_proportions = run_allocation_process_over_time(start_day, end_day, total_orders)

days = list(range(start_day, end_day + 1))

# Plot temporal component of order allocation
plot_temporal_component(days, real_orders_proportions)

# Plot WMAPE over time
plot_wmape_over_time(days, wmape_site_values, wmape_global_values)

# Create Excel workbook
workbook = Workbook()
sheet_results = workbook.active
sheet_results.title = "Results"

# Write headers
sheet_results.append(["Day", "Real Orders Proportion", "WMAPE Site", "WMAPE Global"])

# Write data
for i, day in enumerate(days):
    row = [day, real_orders_proportions[i]]
    if i > 0:
        row.extend([wmape_site_values[i-1], wmape_global_values[i-1]])
    else:
        row.extend(["N/A", "N/A"])
    sheet_results.append(row)

# Save the workbook
workbook.save("allocation_results_over_time.xlsx")

# Print final WMAPE values
print(f"Final WMAPE site: {wmape_site_values[-1]:.3f}")
print(f"Final WMAPE global: {wmape_global_values[-1]:.3f}")