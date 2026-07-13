import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
def run_supply_chain_optimization():

    start = time.time()    
    # ------------------------------------------------------
    # 1. Sets 
    # ------------------------------------------------------
    
    I = 5  # products
    J = 2 # manufacturers
    K = 3 # distribution centers
    C = 6 # customers
    T = 3 # periods
    N = 2  # transportation modes
    M = 1 # backup supplier 1
    L  = 2 # lead time 
    # Ranges
    products = range(I)
    manufacturers = range(J)
    distribution_centers = range(K)
    customers = range(C)
    periods = range(T)
    trans_mode = range(N)
    back_sup = range (M)
    
    # ------------------------------------------------------
    # 2. parameters
    # ------------------------------------------------------
    
    np.random.seed(42)
    
    # MIMI: Manufacturers Minimum Inventory
    df_mim = pd.read_excel("Q_M.xlsx")    
    MIM_np = np.zeros((I, T))
    sorted_prods_mim = sorted(df_mim['Product'].unique())
    for i_idx, prod_name in enumerate(sorted_prods_mim):
        rows = df_mim[df_mim['Product'] == prod_name]        
        MIM_np[i_idx,:] = rows.iloc[:,1:].values.flatten()
    MIMI = MIM_np.tolist()      
    
    # MIDI: Distribution Centres Minimum Inventory
    df_mid = pd.read_excel("Q_DC.xlsx")
    MID_np = np.zeros((I, T))
    sorted_prods_mid = sorted(df_mid['Product'].unique())
    for i_idx, prod_name in enumerate(sorted_prods_mid):
        rows = df_mid[df_mid['Product'] == prod_name]
        MID_np[i_idx,:] = rows.iloc[:,1:].values.flatten()
    MIDI = MID_np.tolist()
    
    # CP: Production Cost
    df_pc = pd.read_excel("production_costs.xlsx")
    c_np = np.zeros((I, J, T))
    sorted_prods = sorted(df_pc['Product'].unique())      
    sorted_manu  = sorted(df_pc['Manufacturer'].unique())   
    for i_idx, prod_name in enumerate(sorted_prods):
        for j_idx, man_name in enumerate(sorted_manu):
            rows = df_pc[(df_pc['Product'] == prod_name) &
                         (df_pc['Manufacturer'] == man_name)]
            c_np[i_idx, j_idx, :] = rows.iloc[:, 2:].values.flatten()
    CP = c_np.tolist()
    
    # CT: Transportation Cost M-->DC
    CT_JK = np.random.uniform(0.7, 1.2, size=(I, J, K, N, T)).tolist()

    # CT: Transportation Cost DC-->Customer
    CT_KC = np.random.uniform(0.7, 1.2, size=(I, K, C, N, T)).tolist()
    
    # CT: Transportation Cost BackupSuppliers-->DC
    CT_MK = np.random.uniform(3, 7, size=(I, M, K, N, T)).tolist()
    
    # CMH: Holding cost at Manufacturers
    CMH = np.random.randint(1,5,size=(I,J,T)).tolist() 

    # CDH: Holding cost at DCs   
    CDH = np.random.randint(1,3,size=(I,K,T)).tolist() 
    
    # CB: Backlog Cost         
    CB = np.random.randint(230,250,size=(I,K,C,T)).tolist() 

    # CBS: Backup Supplier Cost
    CBS = np.random.randint(90,110,size=(I,M, K,T)).tolist()
    
    # CL: Labor Cost
    CL = np.random.randint(110,130,size=(I,J,T)).tolist() 
    
    # MPT: Machine Processing Time
    MPT = np.random.uniform(10,12,size=(I,J,T)).tolist()  
    
    # LPT: Labor Processing Time
    LPT = np.random.uniform(0.1,0.3,size=(I,J,T)).tolist()    
    
    # D: Demand Disruption
    df_dem = pd.read_excel("demand_C_to_DC.xlsx") 
    demand_np = np.zeros((I, K, C, T))
    sorted_dcs_dem   = sorted(df_dem['DC'].unique())
    sorted_cust_dem  = sorted(df_dem['Customer'].unique())
    for i_idx, prod_name in enumerate(sorted_prods):
        for dc_idx, dc_name in enumerate(sorted_dcs_dem):
            for cust_idx, cust_name in enumerate(sorted_cust_dem):
                rows = df_dem[(df_dem['Product'] == prod_name) &
                              (df_dem['DC'] == dc_name) &
                              (df_dem['Customer'] == cust_name)]
                demand_np[i_idx, dc_idx, cust_idx, :] = rows.iloc[:,3:].values.flatten()
    D = demand_np.tolist()
    
    # MPTA: Total Processing Time Available on Machines - Disruption
    MPTA = [
        [600]*T,  
        [600]*T   
    ]
    
    # TA: Working Time Available Per Worker 
    TA = np.random.randint(45,48,size=(J,T)).tolist()
    
    # MLA: Maximum Number of Workers Available 
    MLA = np.random.randint(8,10,size=(J,T)).tolist()   

    # MAMI: Max Manufacturers Inventory Capacity
    MAMI = np.random.randint(2000,3000,size=J).tolist()
    
    # MADI: Max DCs Inventory Capacity
    MADI = np.random.randint(1000,1500,size=K).tolist()
    
    # MSL: Minimum Service Level
    MSL=0.9 
    
    # MACBS: Maximum Backup Suppliers Capacity
    MACBS= np.random.randint(250,300,size=(M,T)).tolist()

    # W: Weight of Each Product
    W = np.random.randint(1, 3, size=(I,))

    # CAPTM: Capacity of Transport Modes
    CAPTM = [200, 400]
    
    # Dis_JK: Distance M-->DC
    Dis_JK = np.random.randint(10, 15, size=(J, K)).tolist()
    
    # Dis_KC: Distance DC-->Customer
    Dis_KC = np.random.randint(3, 7, size=(K, C)).tolist()
    
    # Dis_MK: Distance BackupSuppliers-->DC
    Dis_MK = np.random.randint(20, 30, size=(M, K)).tolist()
    
    # Initial Inventories 
    manuf_init_total = [240,70,400,300,180]
    dc_init_total    = [120,70,120,110,110]
    MH_initM = [[0.0]*J for _ in products]
    DH_initD = [[0.0]*K for _ in products]
    for i in products:
        half_val = manuf_init_total[i] / float(J)
        for j in manufacturers:
            MH_initM[i][j] = half_val
        third_val = dc_init_total[i] / float(K)
        for dc in distribution_centers:
            DH_initD[i][dc] = third_val
            
    Y_1 = [0,0,0,0,0]
    DH_1 =[630,290,430,490,370] 
    B_1    = [1,7,12,10,10]
    initY_1 = [[0.0]*K for _ in products]
    initDH_1 = [[0.0]*K for _ in products]
    initB_1 = [[0.0]*K for _ in products]
    for i in products: 
        third_val0 = Y_1[i] / float(K)
        for k in distribution_centers:
            initY_1[i][k] = third_val0
        third_val1 = DH_1[i] / float(K)
        for k in distribution_centers:
            initDH_1[i][k] = third_val1
        third_val2 = B_1[i] / float(K)
        for k in distribution_centers:
            initB_1[i][k] = third_val2
            
    Y_2 = [0,0,0,0,0] 
    initY_2 = [[0.0]*K for _ in products]
    for i in products: 
        third_val0 = Y_2[i] / float(K)
        for k in distribution_centers:
            initY_2[i][k] = third_val0        
  
    model = gp.Model("Supply_Chain_Optimization")
    
    # ------------------------------------------------------
    # 4. Decision Variables
    # ------------------------------------------------------
    
    # X: Quantity of production
    X = model.addVars(I, J, T, vtype=GRB.CONTINUOUS, name="X")
    
    # Y: Quantity of Shipping M-->DC    
    Y = model.addVars(I, J, K, N, T, vtype=GRB.CONTINUOUS, name="Y") 
    
    # Z: Quantity of Delivering DC-->Customer
    Z = model.addVars(I, K, C, N, T, vtype=GRB.CONTINUOUS, name="Z")  
    
    # MH: Manufacture Inventory
    MH = model.addVars(I, J, T, vtype=GRB.CONTINUOUS, name="MH")
    
    # DH: DCs Inventory
    DH = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, name="DH")
    
    # B: Backlog
    B = model.addVars(I, K, C, T, vtype=GRB.CONTINUOUS, name="B")
    
    # O: Operators
    O  = model.addVars(I, J, T, vtype=GRB.INTEGER,    name="O")
    
    # TM_JK: M-->DC
    TM_JK = model.addVars(J, K, N, T, vtype=GRB.BINARY, name="TM_JK")  
    
    # TM_KC: DC-->Customer
    TM_KC = model.addVars(K, C, N, T, vtype=GRB.BINARY, name="TM_KC")  
    
    #TM_MK: BackupSuppliers-->DC
    TM_MK = model.addVars(M,K, N, T, vtype=GRB.BINARY, name="TM_MK")
    
    # BS: Puchased from Backup Suppliers
    BS = model.addVars(I, M, K, N, T, vtype=GRB.CONTINUOUS, name="BS")  
    
    # ------------------------------------------------------
    # 5. Objective 
    # ------------------------------------------------------
    
    obj = (
        gp.quicksum(X[i,j,t] * CP[i][j][t] 
                    for i in products for j in manufacturers for t in periods)
        
      + gp.quicksum(Y[i,j,k,n,t]* CT_JK[i][j][k][n][t]*Dis_JK[j][k]
                    for i in products for j in manufacturers for k in distribution_centers for n in trans_mode for t in periods)
      + gp.quicksum(Z[i,k,c,n,t]* CT_KC[i][k][c][n][t]*Dis_KC[k][c]
                    for i in products for k in distribution_centers for c in customers for n in trans_mode for t in periods)
      
      + gp.quicksum(MH[i,j,t] * CMH[i][j][t]
                    for i in products for j in manufacturers for t in periods)
      + gp.quicksum(DH[i,k,t] * CDH[i][k][t]
                    for i in products for k in distribution_centers for t in periods)
      + gp.quicksum(B[i,k,c,t] * CB[i][k][c][t]
                    for i in products for k in distribution_centers for c in customers for t in periods)
      + gp.quicksum(O[i,j,t] * CL[i][j][t]
                    for i in products for j in manufacturers for t in periods)
    
      + gp.quicksum(BS[i,m,k,n,t] * CBS[i][m][k][t]
                      for i in products for m in back_sup for k in distribution_centers for n in trans_mode for t in periods)       
      + gp.quicksum(BS[i,m,k,n,t] * CT_MK[i][m][k][n][t]*Dis_MK[m][k]
                      for i in products for m in back_sup for k in distribution_centers for n in trans_mode for t in periods)
      )
    model.setObjective(obj, GRB.MINIMIZE)
    
    # ------------------------------------------------------
    # 6. Constraints
    # ------------------------------------------------------
    
    for j in manufacturers:
        for t in periods:
            model.addConstr(
                gp.quicksum(MPT[i][j][t]* X[i,j,t] for i in products) <= MPTA[j][t],
                name=f"machine_capacity_j{j}_t{t}"
            )
    
    for i in products:
        for j in manufacturers:
            for t in periods:
                if t==0 :           
                    model.addConstr(
                        MH[i,j,t] == MH_initM[i][j]
                                   + X[i,j,t]
                                   - gp.quicksum(Y[i,j,k,n,t] for k in distribution_centers for n in trans_mode),
                        name=f"ManufInv_i{i}_j{j}_t0"
                    )                  
                else: 
                    model.addConstr(
                        MH[i,j,t] == MH[i,j,t-1] + X[i,j,t]
                                       - gp.quicksum(Y[i,j,k,n,t] for k in distribution_centers for n in trans_mode),
                            name=f"ManufInv_i{i}_j{j}_t{t}"
                        )
    
    for i in products:
        for k in distribution_centers:
            for t in periods:
                if t == 0 :
                    if L==0:                        
                         inbound_time = t - L if t>=L else t
                         inbound = gp.quicksum(Y[i,j,k,n,inbound_time] for j in manufacturers for n in trans_mode)
                         model.addConstr(
                             DH[i,k,t]  == initDH_1[i][k] + inbound
                                          - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)                                         
                                          +gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode) ,
                             name=f"InvDC_i{i}_dc{dc}_t{t}"
                         )   
                    else :                        
                         model.addConstr(
                             DH[i,k,t] == initY_2[i][j]
                             - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)
                             +initDH_1[i][k] 
                             + gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode),
                             name=f"InvDC_i{i}_dc{dc}_t{t}"
                         )
                    
                if t == 1 :
                      if L==2:                        
                           model.addConstr(
                               DH[i,k,t] == initY_1[i][j]
                               - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)
                               +DH[i,dc,t-1]  
                               + gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode),
                               name=f"InvDC_i{i}_dc{dc}_t{t}"
                           )   
                      elif L==3:                        
                           model.addConstr(
                               DH[i,k,t] == initY_2[i][j]
                               - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)
                               +DH[i,dc,t-1]  
                               + gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode),
                               name=f"InvDC_i{i}_dc{dc}_t{t}"
                           )                      
                      else:
                          inbound_time = t - L if t>=L else t
                          inbound = gp.quicksum(Y[i,j,k,n,inbound_time] for j in manufacturers for n in trans_mode)
                          model.addConstr(
                              DH[i,k,t]  == DH[i,dc,t-1] + inbound
                                           - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)                                           
                                           +gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode) ,
                              name=f"InvDC_i{i}_dc{dc}_t{t}"
                          )                              
                if t == 2 :
                      if L==3:                        
                           model.addConstr(
                               DH[i,k,t] == initY_1[i][j]
                               - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)
                               +DH[i,dc,t-1]  
                               + gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode),
                               name=f"InvDC_i{i}_dc{dc}_t{t}"
                           )                             
                      else:
                          inbound_time = t - L if t>=L else t
                          inbound = gp.quicksum(Y[i,j,k,n,inbound_time] for j in manufacturers for n in trans_mode)
                          model.addConstr(
                              DH[i,k,t]  == DH[i,dc,t-1] + inbound
                                           - gp.quicksum(Z[i,k,c,n,t] for c in customers for n in trans_mode)                                           
                                           +gp.quicksum(BS[i,m,k,n,t]for m in back_sup for n in trans_mode) ,
                              name=f"InvDC_i{i}_dc{dc}_t{t}"
                          )                
    

    for i in products:
        for k in distribution_centers:
            for c in customers:                
                    for t in periods:
                            if t==0:                                
                                model.addConstr(
                                   B[i,k,c,t]  == D[i][k][c][t] + initB_1[i][k]
                                     -gp.quicksum(Z[i,k,c,n,t]for n in trans_mode),
                                    name=f"z_le_dplusv_i{i}_k{k}_cust{c}_t{t}"
                                )
                            else:
                                model.addConstr(
                                     B[i,k,c,t]== D[i][k][c][t] + B[i,k,c,t-1]
                                     -gp.quicksum(Z[i,k,c,n,t]for n in trans_mode),
                                    name=f"z_le_dplusv_i{i}_k{k}_c{c}_t{t}"                           
                                )
    
    for i in products:
        for k in distribution_centers:
            for c in customers:
                for t in periods:                    
                    model.addConstr(
                        gp.quicksum(Z[i, k, c, n, t] for n in trans_mode)                        
                        >= D[i][k][c][t] * MSL,
                        name=f"service_level{i}_k{k}_c{c}_t{t}"
                    )
    
    for j in manufacturers:
        for t in periods:
            model.addConstr(
                gp.quicksum(LPT[i][j][t] * X[i,j,t] for i in products)
                <= TA[j][t] * O[i,j,t],
                name=f"labor_j{j}_t{t}"
            )
    
    for j in manufacturers:
        for t in periods:
            model.addConstr(
                gp.quicksum(O[i,j,t] for i in products) <= MLA[j][t],
                name=f"max_workforce_j{j}_t{t}"
            )
    
    for i in products:
        for j in manufacturers:
            for t in periods:
                model.addConstr(
                    MH[i,j,t] >= MIMI[i][t],
                    name=f"MinInvM_i{i}_j{j}_t{t}"
                )
    
    for i in products:
        for k in distribution_centers:
            for t in periods:
                model.addConstr(
                    DH[i,k,t] >= MIDI[i][t],
                    name=f"MinInvDC_i{i}_dc{dc}_t{t}"
                )
    
    for j in manufacturers:
        for t in periods:
            model.addConstr(
                gp.quicksum(MH[i,j,t] for i in products) <= MAMI[j],
                name=f"MaxInvManuf_j{j}_t{t}"
            )
          
    for k in distribution_centers:
        for t in periods:
            model.addConstr(
                gp.quicksum(DH[i,k,t] for i in products) <= MADI[k],
                name=f"MaxInvDC_dc{dc}_t{t}"
            )
        
    for j in manufacturers:
        for k in distribution_centers:
            for n in trans_mode:
                for t in periods:
                    model.addConstr(
                        gp.quicksum(Y[i, j, dc, n, t] * W[i] for i in products)
                        <= TM_JK[j, k, n, t] * CAPTM[n],
                        name=f"TransportCapM2DC_j{j}_k{k}_n{n}_t{t}"
                    )
                    
    for k in distribution_centers:
        for c in customers:
            for n in trans_mode:
                for t in periods:
                    model.addConstr(
                        gp.quicksum(Z[i, k, c, n, t] * W[i] for i in products)
                        <= TM_KC[k, c, n, t] * CAPTM[n],
                        name=f"TransportCapDC2C_k{k}_c{c}_n{n}_t{t}"
                    )
                  
    for m in back_sup:
        for k in distribution_centers:
            for n in trans_mode:
                for t in periods:
                    model.addConstr(
                        gp.quicksum(BS[i, m, k, n, t] * W[i] for i in products)
                        <= TM_MK[m, k, n, t] * CAPTM[n],
                        name=f"TransportCapS2DC_m{m}_k{k}_n{n}_t{t}"
                    )
                    
    for m in back_sup:
        for t in periods:
                    model.addConstr(
                        gp.quicksum(BS[i, m, k, n, t] for i in products for k in distribution_centers for n in trans_mode)
                        <= MACBS[m][t],
                        name=f"backupsupliersCap{m}_t{t}"
                    )
    for m in back_sup:
        for t in periods:
            model.addConstr(
                gp.quicksum(BS[i, m, k, n, t] for i in products for k in distribution_centers for n in trans_mode) <= MACBS[m][t],
                name=f"backupsupliersCap{m}_t{t}"
            )
   
    model.optimize()
    model.update()
    print("Vars:", model.NumVars)
    print("Constrs:", model.NumConstrs)
    print("Binary:", model.NumBinVars)
    print("Integer:", model.NumIntVars)

   
    if model.Status == GRB.OPTIMAL:
        print(f"Optimal objective value = {model.ObjVal:.2f}")

        # partial costs
        cost_prod = sum(X[i,j,t].X * CP[i][j][t]
                        for i in products for j in manufacturers for t in periods)
        cost_m2dc = sum(Y[i,j,k,n,t].X* CT_JK[i][j][k][n][t]*Dis_JK[j][k]
                      for i in products for j in manufacturers for k in distribution_centers for n in trans_mode for t in periods)
        cost_dc2c = sum(Z[i,k,c,n,t].X* CT_KC[i][k][c][n][t]*Dis_KC[k][c]
                      for i in products for k in distribution_centers for c in customers for n in trans_mode for t in periods)
        cost_su2dc = sum(BS[i,m,k,n,t].X * CT_MK[i][m][k][n][t]*Dis_MK[m][k]
                      for i in products for m in back_sup for k in distribution_centers for n in trans_mode for t in periods)
        cost_holdm= sum(MH[i,j,t].X * CMH[i][j][t]
                        for i in products for j in manufacturers for t in periods)
        cost_holdd= sum(DH[i,k,t].X * CDH[i][k][t]
                        for i in products for k in distribution_centers for t in periods)
        cost_back = sum(B[i,k,c,t].X * CB[i][k][c][t]
                        for i in products for k in distribution_centers for c in customers for t in periods)
        cost_O    = sum(O[i,j,t].X * CL[i][j][t]
                        for i in products for j in manufacturers for t in periods)
        cost_bsup = sum(BS[i,m,k,n,t].X * CBS[i][m][k][t]
                      for i in products for m in back_sup for k in distribution_centers for n in trans_mode for t in periods)
        
        print("----- Partial Costs -----")
        print(f"Production:       {cost_prod:.2f}")
        print(f"Transp(M->DC):    {cost_m2dc:.2f}")
        print(f"Transp(DC->Cust): {cost_dc2c:.2f}")
        print(f"Transp(SU->DC):   {cost_su2dc:.2f}")
        print(f"Holding(Manuf):   {cost_holdm:.2f}")
        print(f"Holding(DC):      {cost_holdd:.2f}")
        print(f"Backlog:          {cost_back:.2f}")
        print(f"Operator:         {cost_O:.2f}")
        print(f"Backup Supplier:  {cost_bsup:.2f}")
        print("--------------------------------")
        print(f"Total:            {model.ObjVal:.2f}")
        
        sol_data = []
        for var in model.getVars():
            sol_data.append([var.VarName, var.X])
        df_sol = pd.DataFrame(sol_data, columns=["Variable","Value"])
        df_sol.to_excel("solution_gurobi.xlsx", index=False)
        print("Solution exported to solution_gurobi.xlsx")
    else:
        print(f"No optimal solution found. Status={model.Status}")
        if model.Status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            for c in model.getConstrs():
                if c.IISConstr:
                    print("Infeasible constraint:", c.constrName)

    end = time.time() 
    print("Execution time:", end - start, "seconds")
if __name__ == "__main__":
    run_supply_chain_optimization()