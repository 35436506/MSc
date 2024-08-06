import random
import math
import time
import matplotlib.pyplot as plt

# Set seed for reproducibility
random.seed(42)

# Define the recipes with their eligibility
recipes_f1_only = list(range(1, 20))
recipes_f2_only = list(range(30, 66))
recipes_f1_f2 = list(range(20, 30))
all_recipes = recipes_f1_only + recipes_f1_f2 + recipes_f2_only

# Global variables that will be set in run_allocation_process
total_orders = None
factory_capacities = None

def generate_order_recipes(eligible_recipes, num_recipes):
    return random.sample(eligible_recipes, random.randint(1, min(4, len(eligible_recipes))))

def generate_orders_for_day(real_proportion, simulated_proportion, existing_real_orders=None):
    orders = []
    f1_target = int(0.3 * total_orders)
    f2_target = int(0.6 * total_orders)
    f1_count = 0
    f2_count = 0
    total_real_orders = int(real_proportion * total_orders)
    
    # Handle existing real orders
    if existing_real_orders:
        for order in existing_real_orders:
            orders.append(order)
            if 'F1' in order['eligible_factories']:
                f1_count += 1
            if 'F2' in order['eligible_factories']:
                f2_count += 1

    # Generate Group 1 (F1 eligible) orders
    while f1_count < f1_target:
        recipe_ids = generate_order_recipes(all_recipes, 4)
        # Bias towards F1,F3 instead of F1,F2,F3
        eligible_factories = ['F1', 'F3'] if random.random() < 0.7 else ['F1', 'F2', 'F3']
        orders.append({
            'recipe_ids': recipe_ids,
            'eligible_factories': eligible_factories,
            'is_real': len([o for o in orders if o['is_real']]) < total_real_orders
        })
        f1_count += 1
        if 'F2' in eligible_factories:
            f2_count += 1

    # Generate Group 2 (F2 eligible) orders
    while f2_count < f2_target:
        recipe_ids = generate_order_recipes(all_recipes, 4)
        # Bias towards F2,F3 instead of F1,F2,F3
        if f1_count < f1_target and random.random() < 0.3:
            eligible_factories = ['F1', 'F2', 'F3']
            f1_count += 1
        else:
            eligible_factories = ['F2', 'F3']
        orders.append({
            'recipe_ids': recipe_ids,
            'eligible_factories': eligible_factories,
            'is_real': len([o for o in orders if o['is_real']]) < total_real_orders
        })
        f2_count += 1

    # Generate F3-only orders for the remainder
    while len(orders) < total_orders:
        recipe_ids = generate_order_recipes(all_recipes, 4)
        orders.append({
            'recipe_ids': recipe_ids,
            'eligible_factories': ['F3'],
            'is_real': len([o for o in orders if o['is_real']]) < total_real_orders
        })

    # Ensure exact number of real orders
    real_order_count = sum(1 for order in orders if order['is_real'])
    if real_order_count < total_real_orders:
        for order in random.sample([o for o in orders if not o['is_real']], total_real_orders - real_order_count):
            order['is_real'] = True
    elif real_order_count > total_real_orders:
        for order in random.sample([o for o in orders if o['is_real']], real_order_count - total_real_orders):
            order['is_real'] = False

    # Update order IDs
    for i, order in enumerate(orders, start=1):
        order['id'] = i

    return orders

def allocate_orders(orders, factory_capacities):
    allocation = {factory: [] for factory in factory_capacities}
    remaining_orders = orders.copy()

    # Allocate to F1 first, prioritizing F1,F3 and F1,F2,F3 orders
    f1_eligible = sorted([order for order in remaining_orders if 'F1' in order['eligible_factories']],
                         key=lambda x: len(x['eligible_factories']))  # Prioritize F1,F3 over F1,F2,F3
    allocation['F1'] = f1_eligible[:factory_capacities['F1']]
    remaining_orders = [order for order in remaining_orders if order not in allocation['F1']]

    # Allocate to F2, using remaining F1,F2,F3 orders and F2,F3 orders
    f2_eligible = [order for order in remaining_orders if 'F2' in order['eligible_factories']]
    allocation['F2'] = f2_eligible[:factory_capacities['F2']]
    remaining_orders = [order for order in remaining_orders if order not in allocation['F2']]

    # Allocate remaining to F3
    allocation['F3'] = remaining_orders

    return allocation

def calculate_wmape_site(allocation_t_minus_1, allocation_t, sheet_wmape):
    total_abs_diff = 0
    total_items_t = 0
    
    # Get all unique recipe IDs from both days
    all_recipes = set()
    for allocation in [allocation_t_minus_1, allocation_t]:
        for factory in allocation:
            for order in allocation[factory]:
                all_recipes.update(order['recipe_ids'])
    
    # Initialize recipe counts for each factory on both days
    recipe_counts_t_minus_1 = {factory: {recipe: 0 for recipe in all_recipes} for factory in ['F1', 'F2', 'F3']}
    recipe_counts_t = {factory: {recipe: 0 for recipe in all_recipes} for factory in ['F1', 'F2', 'F3']}
    
    # Count the recipes for each factory on day t-1
    for factory, orders in allocation_t_minus_1.items():
        for order in orders:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t_minus_1[factory][recipe_id] += 1
    
    # Count the recipes for each factory on day t
    for factory, orders in allocation_t.items():
        for order in orders:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t[factory][recipe_id] += 1
                total_items_t += 1
                
    # Create the table headers
    sheet_wmape.append(['Recipe', 'Factory', 'Day t-1', 'Day t', 'Absolute recipe-site error'])
    
    # Calculate the absolute difference for each recipe-site combination
    for recipe_id in all_recipes:
        for factory in ['F1', 'F2', 'F3']:
            t_minus_1_count = recipe_counts_t_minus_1[factory][recipe_id]
            t_count = recipe_counts_t[factory][recipe_id]
            abs_diff = abs(t_count - t_minus_1_count)
            total_abs_diff += abs_diff
            sheet_wmape.append([recipe_id, factory, t_minus_1_count, t_count, abs_diff])
    
    if total_items_t == 0:
        wmape_site = float('inf')
    else:
        wmape_site = total_abs_diff / total_items_t
        
    sheet_wmape.append(['', '', 'SUM', total_items_t, total_abs_diff])
    sheet_wmape.append(['WMAPE site', wmape_site])
    sheet_wmape.append([''])
    
    return wmape_site

def calculate_wmape_global(allocation_t_minus_1, allocation_t, sheet_wmape):
    total_abs_diff = 0
    total_t_items = 0
    recipe_counts_t_minus_1 = {}
    recipe_counts_t = {}
    
    # Count the recipes for each factory on day t-1
    for factory in allocation_t_minus_1:
        for order in allocation_t_minus_1[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t_minus_1[recipe_id] = recipe_counts_t_minus_1.get(recipe_id, 0) + 1
    
    # Count the recipes for each factory on day t
    for factory in allocation_t:
        for order in allocation_t[factory]:
            for recipe_id in order['recipe_ids']:
                recipe_counts_t[recipe_id] = recipe_counts_t.get(recipe_id, 0) + 1
                total_t_items += 1
    
    # Get all unique recipe IDs from both days
    all_recipes = set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys())
    
    # Create the table headers
    sheet_wmape.append(['Recipe', 'Day t-1', 'Day t', 'Absolute recipe error'])
    
    # Calculate the absolute difference for each recipe
    for recipe_id in all_recipes:
        t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
        t_count = recipe_counts_t.get(recipe_id, 0)
        abs_diff = abs(t_minus_1_count - t_count)
        total_abs_diff += abs_diff
        sheet_wmape.append([recipe_id, t_minus_1_count, t_count, abs_diff])
    
    # Add the total absolute difference and total items on day t to the table
    sheet_wmape.append(['', 'SUM', total_t_items, total_abs_diff])
    
    if total_t_items == 0:
        wmape_global = float('inf')
    else:
        wmape_global = total_abs_diff / total_t_items
    
    # Add the WMAPE global to the table
    sheet_wmape.append(['WMAPE global', wmape_global])
    
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

# Test scalability
def run_allocation_process(quantity):
    global total_orders, factory_capacities
    
    # Set total orders and factory capacities
    total_orders = quantity
    F1_cap = int(0.25 * total_orders)
    F2_cap = int(0.5 * total_orders)
    factory_capacities = {
        'F1': F1_cap,
        'F2': F2_cap,
        'F3': float('inf')  # F3 has unlimited capacity
    }
    
    # Generate orders for day t-1
    orders_t_minus_1 = generate_orders_for_day(0.3, 0.7)
    # Generate orders for day t
    orders_t = generate_orders_for_day(0.4, 0.6, existing_real_orders=[o for o in orders_t_minus_1 if o['is_real']])
    # Perform allocation for day t-1 and day t
    allocation_t_minus_1 = allocate_orders(orders_t_minus_1, factory_capacities)
    allocation_t = allocate_orders(orders_t, factory_capacities)
    # Calculate WMAPE site before SA and WMAPE global
    wmape_site_before_sa = calculate_wmape_site(allocation_t_minus_1, allocation_t, [])
    wmape_global = calculate_wmape_global(allocation_t_minus_1, allocation_t, [])
    # Apply Simulated Annealing with swapping to optimize allocation on day t
    optimization_start_time = time.time()
    optimized_allocation_t, optimized_wmape_site = simulated_annealing_with_swap(allocation_t_minus_1, allocation_t, factory_capacities)
    optimization_time = time.time() - optimization_start_time
        
    return optimization_time, wmape_site_before_sa, optimized_wmape_site, wmape_global

# Main execution code for scalability test
print("\nScalability test")
order_quantities = [1000, 3000, 5000, 7000, 10000]
results = []
for quantity in order_quantities:
    print(f"Running optimization for {quantity} orders...")
    optimization_time, wmape_before, wmape_after, wmape_global = run_allocation_process(quantity)
    results.append((quantity, optimization_time, wmape_before, wmape_after, wmape_global))
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print("--------------------")

# Print final results
print("\nResult summary:")
print("Order quantity | Optimization time (s) | WMAPE site (Before SA) | WMAPE site (After SA) | WMAPE global")
print("---------------|-----------------------|------------------------|-----------------------|-------------")
for result in results:
    print(f"{result[0]:14d} | {result[1]:21.2f} | {result[2]:22.3f} | {result[3]:22.3f} | {result[4]:12.3f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(order_quantities, [r[1] for r in results], marker='o')
plt.xlabel('Order quantity')
plt.ylabel('Optimization time (seconds)')
plt.title('Optimization time vs Order quantity')
plt.grid(True)
plt.show()

# Plotting WMAPE values
plt.figure(figsize=(12, 6))
plt.plot(order_quantities, [r[2] for r in results], marker='o', label='WMAPE site (Before SA)')
plt.plot(order_quantities, [r[3] for r in results], marker='s', label='WMAPE site (After SA)')
plt.plot(order_quantities, [r[4] for r in results], marker='^', label='WMAPE global')
plt.xlabel('Order quantity')
plt.ylabel('WMAPE')
plt.title('WMAPE vs Order quantity')
plt.legend()
plt.grid(True)
plt.show()