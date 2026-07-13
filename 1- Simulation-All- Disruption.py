import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

simulation_records = []
production_records = []
shipping_records = []
delivery_records = []

class Manufacturer:
    def __init__(self, env, name, Machine_processing_time, Labor_processing_time,
                 Initial_inventory_M, Max_inventory_M, Min_inventory_M,
                 Labor_capacity=480, Machine_capacity=48, shipping=3, work=3):
        self.env = env
        self.name = name
        self.Inventory_M = Initial_inventory_M.copy()
        self.Machine_processing_time = Machine_processing_time
        self.Labor_processing_time = Labor_processing_time
        self.Max_inventory_M = Max_inventory_M
        self.Min_inventory_M = Min_inventory_M
        self.Labor_capacity = Labor_capacity
        self.Machine_capacity = Machine_capacity
        self.weekly_labor_capacity = Labor_capacity
        self.weekly_machine_capacity = Machine_capacity
        self.shipping = shipping
        self.work = work  
        self.log = []
        self.all_products = list(self.Inventory_M.keys())

    def reset_capacity(self):
        
        self.weekly_machine_capacity = self.Machine_capacity
        self.weekly_labor_capacity = self.Labor_capacity
        self.log.append((self.env.now, self.name, "CapacityReset",
                         self.weekly_machine_capacity, self.weekly_labor_capacity))

    def handle_order(self, product, quantity, dc, week):
        
        self.log.append((self.env.now, self.name, product, "OrderReceived", quantity, week))

        inv_contribution = self.Inventory_M[product] - self.Min_inventory_M[product]

        if inv_contribution >= quantity:
            
            self.Inventory_M[product] -= quantity
            self.log.append((self.env.now, self.name, product, "FulfillFromInventory",
                             self.Inventory_M[product], quantity, week))

            
            yield self.env.timeout(self.shipping)
            dc.receive_shipment(product, quantity)
            shipping_records.append({
                'Week': week,
                'Manufacturer': self.name,
                'Product': product,
                'Shipped_Amount': quantity
            })
            self.log.append((self.env.now, self.name, product, "ShipmentSent", quantity, week))

            
            yield self.env.process(self.produce_to_replenish_all(week))
            
        else:
            
            yield self.env.process(self.produce_and_ship(product, quantity, dc, week, inv_contribution))

    def produce_and_ship(self, product, total_order_qty, dc, week, inv_contribution):
        
        order_to_produce = total_order_qty - inv_contribution
        if order_to_produce < 0:
            order_to_produce = 0
            inv_contribution = total_order_qty

        daily_machine_capacity = self.Machine_capacity / 6
        daily_labor_capacity = self.Labor_capacity / 6

        if order_to_produce > 0:
            product_machine_time = self.Machine_processing_time[product] * order_to_produce
            product_labor_time = self.Labor_processing_time[product] * order_to_produce
            production_time_days = max(product_machine_time / daily_machine_capacity,
                                       product_labor_time / daily_labor_capacity)
        else:
            product_machine_time = 0
            product_labor_time = 0
            production_time_days = 0

        allowed_production_time = self.work

        can_produce_full = (production_time_days <= allowed_production_time and
                            product_machine_time <= self.weekly_machine_capacity/(6/allowed_production_time) and
                            product_labor_time <= self.weekly_labor_capacity/(6/allowed_production_time))

        if can_produce_full:
            self.log.append((self.env.now, self.name, product, "StartFullOrderProduction", order_to_produce, week))
            self.weekly_machine_capacity -= product_machine_time
            self.weekly_labor_capacity -= product_labor_time

            if order_to_produce > 0:
                yield self.env.timeout(production_time_days)
                production_records.append({
                    'Week': week,
                    'Manufacturer': self.name,
                    'Product': product,
                    'Produced_Amount': order_to_produce
                })

            
            if inv_contribution > 0:
                max_safe_use = self.Inventory_M[product] - self.Min_inventory_M[product]
                if inv_contribution > max_safe_use:
                    inv_contribution = max_safe_use
                self.Inventory_M[product] -= inv_contribution
                

            self.env.process(self.produce_to_replenish_all(week))
            yield self.env.timeout(self.shipping)
            dc.receive_shipment(product, total_order_qty)
            shipping_records.append({
                'Week': week,
                'Manufacturer': self.name,
                'Product': product,
                'Shipped_Amount': total_order_qty
            })
            self.log.append((self.env.now, self.name, product, "ShipmentSentAfterProduction", total_order_qty, week))

        else:
            self.log.append((self.env.now, self.name, product, "StartPartialOrderScenario", order_to_produce, week))
            feasible_qty = 0
            if (order_to_produce > 0 and
                self.Machine_processing_time[product] > 0 and
                self.Labor_processing_time[product] > 0):

                max_by_time_machine = (allowed_production_time * daily_machine_capacity) / self.Machine_processing_time[product]
                max_by_time_labor = (allowed_production_time * daily_labor_capacity) / self.Labor_processing_time[product]
                max_by_time = int(min(max_by_time_machine, max_by_time_labor, order_to_produce))

                if max_by_time > 0:
                    test_machine = self.Machine_processing_time[product] * max_by_time
                    test_labor = self.Labor_processing_time[product] * max_by_time
                    if test_machine <= self.weekly_machine_capacity/(6/allowed_production_time) and test_labor <= self.weekly_labor_capacity/(6/allowed_production_time):
                        feasible_qty = max_by_time
                    else:
                        machine_ratio = self.weekly_machine_capacity / (self.Machine_processing_time[product] * order_to_produce) if order_to_produce > 0 else 1
                        labor_ratio = self.weekly_labor_capacity / (self.Labor_processing_time[product] * order_to_produce) if order_to_produce > 0 else 1
                        ratio = min(machine_ratio, labor_ratio)
                        feasible_qty = int(order_to_produce * ratio)
                        if feasible_qty > max_by_time:
                            feasible_qty = max_by_time
                        if feasible_qty < 0:
                            feasible_qty = 0

            used_machine = self.Machine_processing_time[product] * feasible_qty
            used_labor = self.Labor_processing_time[product] * feasible_qty

            self.weekly_machine_capacity -= used_machine
            self.weekly_labor_capacity -= used_labor
            self.log.append((self.env.now, self.name, product, "StartPartialProduction", feasible_qty, week))

            if feasible_qty > 0:
                yield self.env.timeout(allowed_production_time)
                production_records.append({
                    'Week': week,
                    'Manufacturer': self.name,
                    'Product': product,
                    'Produced_Amount': feasible_qty
                })
            else:
                yield self.env.timeout(allowed_production_time)

            remainder = total_order_qty - feasible_qty
            max_safe_use = self.Inventory_M[product] - self.Min_inventory_M[product]
            if inv_contribution > max_safe_use:
                inv_contribution = max_safe_use

            if inv_contribution >= remainder and (self.Inventory_M[product] - remainder) >= self.Min_inventory_M[product]:
                self.Inventory_M[product] -= remainder
                yield self.env.timeout(self.shipping)
                dc.receive_shipment(product, total_order_qty)
                shipping_records.append({
                    'Week': week,
                    'Manufacturer': self.name,
                    'Product': product,
                    'Shipped_Amount': total_order_qty
                })
                self.log.append((self.env.now, self.name, product, "PartialShipmentSentFullOrder", total_order_qty, week))
            else:
                safe_inv_use = min(inv_contribution, max_safe_use)
                if safe_inv_use > 0:
                    feasible_qty += safe_inv_use
                    self.Inventory_M[product] -= safe_inv_use
                    if self.Inventory_M[product] < self.Min_inventory_M[product]:
                        raise ValueError("Inventory fell below minimum after using inventory in partial shipment!")
                        
                self.env.process(self.produce_to_replenish_all(week))
                yield self.env.timeout(self.shipping)
                dc.receive_shipment(product, feasible_qty)
                shipping_records.append({
                    'Week': week,
                    'Manufacturer': self.name,
                    'Product': product,
                    'Shipped_Amount': feasible_qty
                })
                self.log.append((self.env.now, self.name, product, "PartialShipmentSentPartialOrder", feasible_qty, week))

    def produce_to_replenish_all(self, week):

        while True:
            products_below_max = [p for p in self.all_products if self.Inventory_M[p] < self.Max_inventory_M[p]]
            if not products_below_max:
               
                break

            needed_qty = {p: self.Max_inventory_M[p] - self.Inventory_M[p] for p in products_below_max}

            total_machine_time = 0
            total_labor_time = 0
            for p in products_below_max:
                total_machine_time += self.Machine_processing_time[p] * needed_qty[p]
                total_labor_time += self.Labor_processing_time[p] * needed_qty[p]

            if total_machine_time > self.weekly_machine_capacity or total_labor_time > self.weekly_labor_capacity:
                machine_ratio = (self.weekly_machine_capacity / total_machine_time) if total_machine_time > 0 else 1
                labor_ratio = (self.weekly_labor_capacity / total_labor_time) if total_labor_time > 0 else 1
                ratio = min(machine_ratio, labor_ratio)

                partial_production = {p: int(needed_qty[p] * ratio) for p in products_below_max}

                used_machine = sum(self.Machine_processing_time[p] * partial_production[p] for p in products_below_max)
                used_labor = sum(self.Labor_processing_time[p] * partial_production[p] for p in products_below_max)

                self.weekly_machine_capacity -= used_machine
                self.weekly_labor_capacity -= used_labor

                self.log.append((self.env.now, self.name, "AllProducts", "StartPartialReplenishProduction",
                                 partial_production, week))

                daily_machine_capacity = self.Machine_capacity / 6
                daily_labor_capacity = self.Labor_capacity / 6
                time_needed_days = max(used_machine / daily_machine_capacity,
                                       used_labor / daily_labor_capacity) if (used_machine > 0 or used_labor > 0) else 0
                yield self.env.timeout(time_needed_days)

                for p in products_below_max:
                    self.Inventory_M[p] += partial_production[p]
                    if self.Inventory_M[p] > self.Max_inventory_M[p]:
                        self.Inventory_M[p] = self.Max_inventory_M[p]

                    if partial_production[p] > 0:
                        production_records.append({
                            'Week': week,
                            'Manufacturer': self.name,
                            'Product': p,
                            'Produced_Amount': partial_production[p]
                        })
                    self.log.append((self.env.now, self.name, p, "InventoryReplenishedPartial", self.Inventory_M[p], week))

                break
            else:
                self.weekly_machine_capacity -= total_machine_time
                self.weekly_labor_capacity -= total_labor_time

                self.log.append((self.env.now, self.name, "AllProducts", "StartFullReplenishProduction", needed_qty, week))

                daily_machine_capacity = self.Machine_capacity / 6
                daily_labor_capacity = self.Labor_capacity / 6
                time_needed_days = max(total_machine_time / daily_machine_capacity,
                                       total_labor_time / daily_labor_capacity) if (total_machine_time > 0 or total_labor_time > 0) else 0
                yield self.env.timeout(time_needed_days)

                for p in products_below_max:
                    self.Inventory_M[p] += needed_qty[p]
                    if self.Inventory_M[p] > self.Max_inventory_M[p]:
                        self.Inventory_M[p] = self.Max_inventory_M[p]

                    if needed_qty[p] > 0:
                        production_records.append({
                            'Week': week,
                            'Manufacturer': self.name,
                            'Product': p,
                            'Produced_Amount': needed_qty[p]
                        })
                    self.log.append((self.env.now, self.name, p, "InventoryReplenished", self.Inventory_M[p], week))

class DistributionCenter:
    def __init__(self, env, name, 
                 Initial_inventory_DC,
                 Manufacturers,
                 Desired_inventory,
                 Alpha,
                 Beta,
                 Desired_wip,
                 Max_inventory_DC,
                 Min_Inventory_DC):
        
        self.env = env
        self.name = name
        self.Inventory_DC = Initial_inventory_DC.copy()
        self.Manufacturers = Manufacturers
        self.Desired_inventory = Desired_inventory
        self.Alpha = Alpha
        self.Beta = Beta
        self.Desired_wip = Desired_wip
        self.Max_inventory_DC = Max_inventory_DC
        self.Min_Inventory_DC = Min_Inventory_DC
        self.log = []
        self.pending_orders = {p: 0 for p in self.Inventory_DC}
    
    def compute_order_quantity(self, product, current_demand):
        order_qty = max(0, np.mean(current_demand) + (
            self.Alpha[product] * (self.Desired_inventory[product] - self.Inventory_DC[product])
            + self.Beta[product]  * (self.Desired_wip[product])
        ))

        return int(order_qty)

    def receive_shipment(self, product, quantity):
        self.Inventory_DC[product] += quantity
        if self.Inventory_DC[product] > self.Max_inventory_DC[product]:
            overflow = self.Inventory_DC[product] - self.Max_inventory_DC[product]
            self.Inventory_DC[product] = self.Max_inventory_DC[product]
            self.log.append((self.env.now, self.name, product, "ShipmentReceived_Overflow", quantity, overflow))
        else:
            self.log.append((self.env.now, self.name, product, "ShipmentReceived", quantity))

    def calculate_service_level(self, product, fulfilled, demand):
        return fulfilled / demand if demand > 0 else 1.0

    def process_demand(self, product, demand, week):
        order_qty = self.compute_order_quantity(product, demand)
        if order_qty > 0:
            chosen_manufacturer = np.random.choice(self.Manufacturers)
            self.log.append((self.env.now, self.name, product, "OrderPlaced", order_qty, chosen_manufacturer.name, week))
            order_process = self.env.process(chosen_manufacturer.handle_order(product, order_qty, self, week))
        else:
            order_process = None

        on_hand_before = self.Inventory_DC[product]
        fulfilled = 0
        unfulfilled = 0

        available_above_min = self.Inventory_DC[product] - self.Min_Inventory_DC[product]

        if demand <= available_above_min:
            yield self.env.timeout(1)
            self.Inventory_DC[product] -= demand
            if self.Inventory_DC[product] < self.Min_Inventory_DC[product]:
                shortfall = self.Min_Inventory_DC[product] - self.Inventory_DC[product]
                self.Inventory_DC[product] = self.Min_Inventory_DC[product]
                fulfilled = demand - shortfall
                unfulfilled = shortfall
            else:
                fulfilled = demand
                unfulfilled = 0
        else:
            if order_process:
                yield order_process

            after_receipt = self.Inventory_DC[product]
            received_goods = after_receipt - on_hand_before

            if received_goods >= demand:
                yield self.env.timeout(1)
                self.Inventory_DC[product] -= demand
                if self.Inventory_DC[product] < self.Min_Inventory_DC[product]:
                    shortfall = self.Min_Inventory_DC[product] - self.Inventory_DC[product]
                    self.Inventory_DC[product] = self.Min_Inventory_DC[product]
                    fulfilled = demand - shortfall
                    unfulfilled = shortfall
                else:
                    fulfilled = demand
                    unfulfilled = 0
            else:
                remainder = demand - received_goods
                yield self.env.timeout(1)
                new_available_above_min = self.Inventory_DC[product] - self.Min_Inventory_DC[product]

                if remainder <= new_available_above_min:
                    self.Inventory_DC[product] -= remainder
                    if self.Inventory_DC[product] < self.Min_Inventory_DC[product]:
                        shortfall = self.Min_Inventory_DC[product] - self.Inventory_DC[product]
                        self.Inventory_DC[product] = self.Min_Inventory_DC[product]
                        fulfilled = received_goods + remainder - shortfall
                        unfulfilled = shortfall
                    else:
                        fulfilled = demand
                        unfulfilled = 0
                else:
                    can_fulfill_from_inv = max(0, new_available_above_min)
                    self.Inventory_DC[product] -= can_fulfill_from_inv
                    if self.Inventory_DC[product] < self.Min_Inventory_DC[product]:
                        diff = self.Min_Inventory_DC[product] - self.Inventory_DC[product]
                        can_fulfill_from_inv -= diff
                        self.Inventory_DC[product] = self.Min_Inventory_DC[product]

                    fulfilled = received_goods + can_fulfill_from_inv
                    unfulfilled = demand - fulfilled

        if self.Inventory_DC[product] > self.Max_inventory_DC[product]:
            overflow = self.Inventory_DC[product] - self.Max_inventory_DC[product]
            self.Inventory_DC[product] = self.Max_inventory_DC[product]
            self.log.append((self.env.now, self.name, product, "OverflowClamped", overflow))

        if self.Inventory_DC[product] < self.Min_Inventory_DC[product]:
            needed_to_reach_min = self.Min_Inventory_DC[product] - self.Inventory_DC[product]
            self.Inventory_DC[product] = self.Min_Inventory_DC[product]
            fulfilled -= needed_to_reach_min
            if fulfilled < 0:
                unfulfilled += (-fulfilled)
                fulfilled = 0

        service_level = self.calculate_service_level(product, fulfilled, demand)
        if service_level < 0:
            service_level = 0

        simulation_records.append({
            'Week': week,
            'DistributionCenter': self.name,
            'Product': product,
            'Customer_Demand': demand,
            'Manufacturer_Inventory': np.mean([mfg.Inventory_M[product] for mfg in self.Manufacturers]),
            'DC_Inventory': self.Inventory_DC[product],
            'Service_Level': service_level,
            'order_quantity': order_qty
        })

def run_simulation():
    env = simpy.Environment()
    np.random.seed(42)  

    products = ['P1', 'P2', 'P3', 'P4', 'P5']
    manufacturer_names = ['M1', 'M2']

    Machine_processing_time = {
        'M1': {'P1': np.random.uniform(0.209,0.224), 'P2': np.random.uniform(0.088,0.113), 'P3':np.random.uniform(0.162,0.164), 'P4': np.random.uniform(0.194,0.22), 'P5':np.random.uniform(0.126,0.135)},
        'M2': {'P1': np.random.uniform(0.209,0.224), 'P2': np.random.uniform(0.088,0.113), 'P3':np.random.uniform(0.162,0.164), 'P4': np.random.uniform(0.194,0.22), 'P5':np.random.uniform(0.126,0.135)}
    }
    Labor_processing_time = {
        'M1': {'P1': np.random.uniform(0.042,0.057), 'P2':np.random.uniform(0.038,0.058), 'P3': np.random.uniform(0.091,0.106), 'P4': np.random.uniform(0.103,0.12), 'P5': np.random.uniform(0.102,0.122)},
        'M2': {'P1': np.random.uniform(0.042,0.057), 'P2':np.random.uniform(0.038,0.058), 'P3': np.random.uniform(0.091,0.106), 'P4': np.random.uniform(0.103,0.12), 'P5': np.random.uniform(0.102,0.122)}
    }

    Initial_inventory_M = {
        'M1': {'P1': 183, 'P2': 80, 'P3': 200, 'P4': 138, 'P5': 155},
        'M2': {'P1': 183, 'P2': 80, 'P3': 200, 'P4': 138, 'P5': 155}
    }

    # Original parameters
    orig_Labor_capacity = 480
    orig_Machine_capacity = 48
    orig_shipping = 3
    orig_work = 3
    orig_Max_inventory_M = {
        'M1': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170},
        'M2': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170}
    }
    orig_Min_Inventory_M = {
        'M1': {'P1': 100, 'P2': 60, 'P3': 110, 'P4': 90, 'P5': 90},
        'M2': {'P1': 100, 'P2': 60, 'P3': 110, 'P4': 90, 'P5': 90}
    }
    orig_demand_distributions = {
       'P1': (30/1.5, 60/1.5), 'P2': (12/1.5, 26/1.5), 'P3': (32/1.5, 64/1.5), 'P4': (20/1.5, 40/1.5), 'P5': (25/1.5, 50/1.5)
   }

    orig_Max_inventory_DC = {
        'DC1': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC2': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC3': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140}
    }
    orig_Min_Inventory_DC = {
        'DC1': {'P1': 80, 'P2': 40, 'P3': 90, 'P4': 70, 'P5': 70},
        'DC2': {'P1': 80, 'P2': 40, 'P3': 90, 'P4': 70, 'P5': 70},
        'DC3': {'P1': 80, 'P2': 40, 'P3': 90, 'P4': 70, 'P5': 70}
    }
    orig_Desired_inventory = {
        'DC1': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC2': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC3': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135}
    }
    ###########################################################################
    # Modified parameters for weeks 17
    mod1_Labor_capacity = 480
    mod1_Machine_capacity = 23
    mod1_shipping = 6
    mod1_work = 0
    mod1_Max_inventory_M = {
        'M1': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170},
        'M2': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170}
    }
    mod1_Min_Inventory_M = {
        'M1': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1},
        'M2': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1}
    }
    mod1_demand_distributions = {
        'P1': (30/1, 60/1), 'P2': (12/1, 26/1), 'P3': (32/1, 64/1), 'P4': (20/1, 40/1), 'P5': (25/1, 50/1)
    }

    mod1_Max_inventory_DC = {
        'DC1': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC2': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC3': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140}
    }
    mod1_Min_Inventory_DC = {
        'DC1': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC2': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC3': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1}
    }
    mod1_Desired_inventory = {
        'DC1': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC2': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC3': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135}
    }
    
    # Modified parameters for weeks 18
    mod2_Labor_capacity = 480
    mod2_Machine_capacity = 23
    mod2_shipping = 6
    mod2_work = 0
    mod2_Max_inventory_M = {
        'M1': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170},
        'M2': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170}
    }
    mod2_Min_Inventory_M = {
        'M1': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1},
        'M2': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1}
    }
    mod2_demand_distributions = {
        'P1': (30/1, 60/1), 'P2': (12/1, 26/1), 'P3': (32/1, 64/1), 'P4': (20/1, 40/1), 'P5': (25/1, 50/1)
    }

    mod2_Max_inventory_DC = {
        'DC1': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC2': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC3': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140}
    }
    mod2_Min_Inventory_DC = {
        'DC1': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC2': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC3': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1}
    }
    mod2_Desired_inventory = {
        'DC1': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC2': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC3': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135}
    }
    
    # Modified parameters for weeks 19
    mod3_Labor_capacity = 480
    mod3_Machine_capacity = 23
    mod3_shipping = 6
    mod3_work = 0
    mod3_Max_inventory_M = {
        'M1': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170},
        'M2': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170}
    }
    mod3_Min_Inventory_M = {
        'M1': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1},
        'M2': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1}
    }
    mod3_demand_distributions = {
        'P1': (30/1, 60/1), 'P2': (12/1, 26/1), 'P3': (32/1, 64/1), 'P4': (20/1, 40/1), 'P5': (25/1, 50/1)
    }

    mod3_Max_inventory_DC = {
        'DC1': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC2': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC3': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140}
    }
    mod3_Min_Inventory_DC = {
        'DC1': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC2': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC3': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1}
    }
    mod3_Desired_inventory = {
        'DC1': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC2': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC3': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135}
    }
    # Modified parameters for weeks 20,21 
    mod_Labor_capacity = 480
    mod_Machine_capacity = 23
    mod_shipping = 6
    mod_work = 0
    mod_Max_inventory_M = {
        'M1': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170},
        'M2': {'P1': 220, 'P2': 85, 'P3': 220, 'P4': 160, 'P5': 170}
    }
    mod_Min_Inventory_M = {
        'M1': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1},
        'M2': {'P1': 100*1, 'P2': 60*1, 'P3': 110*1, 'P4': 90*1, 'P5': 90*1}
    }
    mod_demand_distributions = {
        'P1': (30/1, 60/1), 'P2': (12/1, 26/1), 'P3': (32/1, 64/1), 'P4': (20/1, 40/1), 'P5': (25/1, 50/1)
    }

    mod_Max_inventory_DC = {
        'DC1': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC2': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140},
        'DC3': {'P1': 162, 'P2': 80, 'P3': 185, 'P4': 140, 'P5': 140}
    }
    mod_Min_Inventory_DC = {
        'DC1': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC2': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1},
        'DC3': {'P1': 80*1, 'P2': 40*1, 'P3': 90*1, 'P4': 70*1, 'P5': 70*1}
    }
    mod_Desired_inventory = {
        'DC1': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC2': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135},
        'DC3': {'P1': 189, 'P2': 60, 'P3': 150, 'P4': 115, 'P5': 135}
    }

    manufacturers = [
        Manufacturer(env, m_name, Machine_processing_time[m_name], Labor_processing_time[m_name],
                     Initial_inventory_M[m_name], orig_Max_inventory_M[m_name], orig_Min_Inventory_M[m_name],
                     Labor_capacity=orig_Labor_capacity, Machine_capacity=orig_Machine_capacity,
                     shipping=orig_shipping, work=orig_work)
        for m_name in manufacturer_names
    ]
    
    distribution_centers = ['DC1', 'DC2', 'DC3']
    customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    current_Max_inventory_DC = orig_Max_inventory_DC
    current_Min_Inventory_DC = orig_Min_Inventory_DC
    current_Desired_inventory = orig_Desired_inventory

    distribution_center_objs = [
        DistributionCenter(env, dc_name,
                           {'P1': 142, 'P2': 65, 'P3': 157, 'P4': 123, 'P5': 121},
                           manufacturers,
                           current_Desired_inventory[dc_name],
                           {product: np.random.uniform(0, 1) for product in products},
                           {product: np.random.uniform(0, 0.4) for product in products},
                           {product:36 if product in ['P1','P3','P4','P5'] else 18 for product in products},
                           current_Max_inventory_DC[dc_name],
                           current_Min_Inventory_DC[dc_name])
        for dc_name in distribution_centers
    ]

    def demand_generator(env, distribution_centers, customers, products, distribution_center_objs):
        week = 0
        while True: # CHeck it manually to work correct
            special_weeks1 = [17,
                             52+17,
                             2*52+17,
                             3*52+17,
                             4*52+17,
                             5*52+17,
                             6*52+17,
                             7*52+17,
                             8*52+17,
                             9*52+17,
                             10*52+17,
                             11*52+17,
                             12*52+17,
                             13*52+17,
                             14*52+17,
                             15*52+17,
                             16*52+17,
                             17*52+17,
                             18*52+17,
                             19*52+17]
            special_weeks2 = [18,
                             52+18,
                             2*52+18,
                             3*52+18,
                             4*52+18,
                             5*52+18,
                             6*52+18,
                             7*52+18,
                             8*52+18,
                             9*52+18,
                             10*52+18,
                             11*52+18,
                             12*52+18,
                             13*52+18,
                             14*52+18,
                             15*52+18,
                             16*52+18,
                             17*52+18,
                             17*52+18,
                             17*52+18]
            special_weeks3 = [19,
                             52+19,
                             2*52+19,
                             3*52+19,
                             4*52+19,
                             5*52+19,
                             6*52+19,
                             7*52+19,
                             8*52+19,
                             9*52+19,
                             10*52+19,
                             11*52+19,
                             12*52+19,
                             13*52+19,
                             14*52+19,
                             15*52+19,
                             16*52+19,
                             17*52+19,
                             18*52+19,
                             19*52+19]
            special_weeks = [20,21,
                             52+20,52+21,
                             2*52+20,2*52+21,
                             3*52+20,3*52+21,
                             4*52+20,4*52+21,
                             5*52+20,5*52+21,
                             6*52+20,6*52+21,
                             7*52+20,7*52+21,
                             8*52+20,8*52+21,
                             9*52+20,9*52+21,
                             10*52+20,10*52+21,
                             11*52+20,11*52+21,
                             12*52+20,12*52+21,
                             13*52+20,13*52+21,
                             14*52+20,14*52+21,
                             15*52+20,15*52+21,
                             16*52+20,16*52+21,
                             17*52+20,17*52+21,
                             18*52+20,18*52+21,
                             19*52+20,19*52+21]
            if week in special_weeks1:
                for mfg in manufacturers:
                    mfg.Labor_capacity = mod1_Labor_capacity
                    mfg.Machine_capacity = mod1_Machine_capacity
                    mfg.shipping = mod1_shipping
                    mfg.work = mod1_work
                    mfg.Max_inventory_M = mod1_Max_inventory_M[mfg.name]
                    mfg.Min_inventory_M = mod1_Min_Inventory_M[mfg.name]
                    mfg.reset_capacity()

                for dc_obj in distribution_center_objs:
                    dc_obj.Max_inventory_DC = mod1_Max_inventory_DC[dc_obj.name]
                    dc_obj.Min_Inventory_DC = mod1_Min_Inventory_DC[dc_obj.name]
                    dc_obj.Desired_inventory = mod1_Desired_inventory[dc_obj.name]

                current_demand_distributions_local = mod1_demand_distributions            
            
            elif week in special_weeks2:
                for mfg in manufacturers:
                    mfg.Labor_capacity = mod2_Labor_capacity
                    mfg.Machine_capacity = mod2_Machine_capacity
                    mfg.shipping = mod2_shipping
                    mfg.work = mod2_work
                    mfg.Max_inventory_M = mod2_Max_inventory_M[mfg.name]
                    mfg.Min_inventory_M = mod2_Min_Inventory_M[mfg.name]
                    mfg.reset_capacity()

                for dc_obj in distribution_center_objs:
                    dc_obj.Max_inventory_DC = mod2_Max_inventory_DC[dc_obj.name]
                    dc_obj.Min_Inventory_DC = mod2_Min_Inventory_DC[dc_obj.name]
                    dc_obj.Desired_inventory = mod2_Desired_inventory[dc_obj.name]

                current_demand_distributions_local = mod2_demand_distributions
            
            elif week in special_weeks3:
                for mfg in manufacturers:
                    mfg.Labor_capacity = mod3_Labor_capacity
                    mfg.Machine_capacity = mod3_Machine_capacity
                    mfg.shipping = mod3_shipping
                    mfg.work = mod3_work
                    mfg.Max_inventory_M = mod3_Max_inventory_M[mfg.name]
                    mfg.Min_inventory_M = mod3_Min_Inventory_M[mfg.name]
                    mfg.reset_capacity()

                for dc_obj in distribution_center_objs:
                    dc_obj.Max_inventory_DC = mod3_Max_inventory_DC[dc_obj.name]
                    dc_obj.Min_Inventory_DC = mod3_Min_Inventory_DC[dc_obj.name]
                    dc_obj.Desired_inventory = mod3_Desired_inventory[dc_obj.name]

                current_demand_distributions_local = mod3_demand_distributions
            
            elif week in special_weeks:
                for mfg in manufacturers:
                    mfg.Labor_capacity = mod_Labor_capacity
                    mfg.Machine_capacity = mod_Machine_capacity
                    mfg.shipping = mod_shipping
                    mfg.work = mod_work
                    mfg.Max_inventory_M = mod_Max_inventory_M[mfg.name]
                    mfg.Min_inventory_M = mod_Min_Inventory_M[mfg.name]
                    mfg.reset_capacity()

                for dc_obj in distribution_center_objs:
                    dc_obj.Max_inventory_DC = mod_Max_inventory_DC[dc_obj.name]
                    dc_obj.Min_Inventory_DC = mod_Min_Inventory_DC[dc_obj.name]
                    dc_obj.Desired_inventory = mod_Desired_inventory[dc_obj.name]

                current_demand_distributions_local = mod_demand_distributions
            else:
                for mfg in manufacturers:
                    mfg.Labor_capacity = orig_Labor_capacity
                    mfg.Machine_capacity = orig_Machine_capacity
                    mfg.shipping = orig_shipping
                    mfg.work = orig_work
                    mfg.Max_inventory_M = orig_Max_inventory_M[mfg.name]
                    mfg.Min_inventory_M = orig_Min_Inventory_M[mfg.name]
                    mfg.reset_capacity()

                for dc_obj in distribution_center_objs:
                    dc_obj.Max_inventory_DC = orig_Max_inventory_DC[dc_obj.name]
                    dc_obj.Min_Inventory_DC = orig_Min_Inventory_DC[dc_obj.name]
                    dc_obj.Desired_inventory = orig_Desired_inventory[dc_obj.name]

                current_demand_distributions_local = orig_demand_distributions
            customer_demand = {
                customer: {
                    product: np.random.randint(current_demand_distributions_local[product][0], current_demand_distributions_local[product][1] + 1)
                    for product in products
                }
                for customer in customers
            }

            product_demand = {
                product: sum(customer_demand[c][product] for c in customers)
                for product in products
            }

            DC_demand = {dc: {} for dc in distribution_centers}
            for product in products:
                total = product_demand[product]
                allocation_per_dc = int(total / len(distribution_centers))
                for dc in distribution_centers:
                    DC_demand[dc][product] = allocation_per_dc

            for dc_obj in distribution_center_objs:
                dc_name = dc_obj.name
                for product in products:
                    dc_demand = DC_demand[dc_name][product]
                    env.process(dc_obj.process_demand(product, dc_demand, week))

            week += 1
            yield env.timeout(7)  

    env.process(demand_generator(env, distribution_centers, customers, products, distribution_center_objs))

    def reset_weekly_capacity(env, manufacturers):
        while True:
            yield env.timeout(7)
            for mfg in manufacturers:
                mfg.reset_capacity()

    env.process(reset_weekly_capacity(env, manufacturers))

    env.run(until=1040*7)

    df_simulation = pd.DataFrame(simulation_records)
    df_simulation = df_simulation[['Week', 'DistributionCenter', 'Product', 'Customer_Demand', 
                                   'Manufacturer_Inventory', 'DC_Inventory', 'Service_Level', 'order_quantity']]
    df_production = pd.DataFrame(production_records)
    df_shipping = pd.DataFrame(shipping_records)
    df_delivery = pd.DataFrame(delivery_records)
    
    with pd.ExcelWriter("1-SupplyChainSimulationResults_All_Disruption.xlsx", engine='xlsxwriter') as writer:
        df_simulation.to_excel(writer, sheet_name='Simulation_Records', index=False)
        df_production.to_excel(writer, sheet_name='Production_Records', index=False)
        df_shipping.to_excel(writer, sheet_name='Shipping_Records', index=False)
        df_delivery.to_excel(writer, sheet_name='Delivery_Records', index=False)
    print("Simulation completed. Results saved to 1-SupplyChainSimulationResults_All_Disruption.xlsx.")

    def generate_decision_rules(data):
        X = data.drop(columns=['Decision'], errors='ignore')
        if 'Decision' in data.columns:
            y = data['Decision']
            clf = DecisionTreeClassifier(max_depth=5)
            clf.fit(X, y)
            return clf

run_simulation()

df = pd.read_excel("1-SupplyChainSimulationResults_All_Disruption.xlsx")
df_relevant = df[['Week', 'Product', 'Customer_Demand', 'Manufacturer_Inventory',
                  'DC_Inventory', 'Service_Level', 'order_quantity']]

grouped_df = df_relevant.groupby(['Week', 'Product'], as_index=False).agg({
    'Customer_Demand': 'sum',
    'Manufacturer_Inventory': 'mean',
    'DC_Inventory': 'mean',
    'Service_Level': 'mean',
    'order_quantity': 'mean'
})

grouped_df.to_excel('2-Output_All_Disruption.xlsx', index=False)
print("\nData has been successfully transformed and saved to '2-Output_All_Disruption.xlsx'.")

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = [
    ('Customer_Demand', 'Customer Demand Over Time'),
    ('DC_Inventory', 'DC Inventory Over Time'),
    ('Manufacturer_Inventory', 'Manufacturer Inventory Over Time'),
    ('Service_Level', 'Service Level Over Time')
]

for ax, (metric, title) in zip(axes.flatten(), metrics):
    sns.lineplot(
        data=grouped_df,
        x='Week',
        y=metric,
        hue='Product',
        style='Product',
        markers=False,
        dashes=False,
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=12)
    ax.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
fig.savefig('All_Disruption.png', dpi=120)



data = pd.read_excel("2-Output_All_Disruption.xlsx")
df_relevant1 = data[['Week', 'Product',  'Manufacturer_Inventory',
                  'DC_Inventory', 'Service_Level']]

def get_weeks(i):
    return [i * 52 + offset for offset in range(17, 22)]

all_data = []

for i in range(20):
    selected_weeks = get_weeks(i)
    filtered_data = data[data['Week'].isin(selected_weeks)]
    filtered_data['Group'] = f'Weeks_i{i}'  
    all_data.append(filtered_data)

combined_data = pd.concat(all_data, ignore_index=True)

output_file = '3-Disruption_Data.xlsx'
combined_data.to_excel(output_file, index=False, sheet_name='All_Weeks')

print(f"All filtered data saved to a single sheet in {output_file}")


input_file = '3-Disruption_Data.xlsx'  
data = pd.read_excel(input_file)

products = data['Product'].unique()

for product in products:
    filtered_data = data[data['Product'] == product]
    output_file = f'Disruption_Data_{product}.xlsx'
    filtered_data.to_excel(output_file, index=False)
    print(f"Data for Product {product} saved to {output_file}")

