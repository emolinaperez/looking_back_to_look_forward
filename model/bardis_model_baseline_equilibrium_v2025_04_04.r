library(deSolve)

# Define the system of differential equations
dynamics_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    
    # Flows

    #Resources 
    resource_inflow <- k_resources* Resources
    extractive_pollution <- k_pollution * Economy * Resources 

    #Economy 
    production <- ef_economy_resources_on_prod * Resources * Economy * Bureaucracy  
    depreciation <- k_deprec * Economy * Pollution 
    
    #Bureaucracy 
    bureaucracy_creation <- ef_economy_on_bureaucracy * Economy * Bureaucracy
    bureaucracy_decay <- k_decay_bureaucracy  * Bureaucracy * Pollution 

    #Pollution 
    pollution_abtement <- k_pollution_decay * Pollution

    # Rate of change
    dResources <- resource_inflow - production - extractive_pollution
    dEconomy <- production - depreciation - bureaucracy_creation
    dBureaucracy <- bureaucracy_creation - bureaucracy_decay
    dPollution <- depreciation + bureaucracy_decay + extractive_pollution - pollution_abtement
    
    list(c(dResources, dEconomy, dBureaucracy, dPollution),
            inflow = bureaucracy_creation, 
            outflow1 = bureaucracy_decay
            )
  })
}


# Initial conditions
state <- c(Resources = 1.0, Economy = 0.1, Bureaucracy = 0.01, Pollution = 0.001)

# Model parameters
parameters <- c(
  k_resources =  0.15 * 0.1, # Autoregeneration rate of resources  
  ef_economy_resources_on_prod = 17.0,          # Production rate
  ef_bureaucracy_on_prod = 0.02, # Effect of bureaucracy on production 
  k_deprec = 0.001 * 0.1,       # Depreciation rate
  ef_pollution_on_depreciation = 0.05, # Effect of pollution on economy depreciation 
  k_bureaucracy = 0.01,    # Bureaucracy formation rate
  ef_economy_on_bureaucracy = 3.5 * 0.8, # Effect of the Economy of bureaucracy formation 
  k_decay_bureaucracy = 0.5 * 5,          # Bureaucracy decay rate
  ef_pollution_on_bureaucracy = 0.02, # Effect of pollution on bureaucracy decay  
  k_pollution = 0.12, #Pollution generation rate 
  k_pollution_decay = 0.0 # Pollution decay rate
) * 0.08

# Time sequence (note the temporal granuality is required to avoid integration errors with rk4)
time <- seq(0, 200, by = 0.01)

# Solve the system
out <- ode(y = state, times = time, func = dynamics_model, parms = parameters, method="rk4")
# Convert output to data frame
out_df <- as.data.frame(out)
dir.output <- "/home/tony-ubuntu/decision_sciences/looking_back_to_look_forward/tableau/"
write.csv(out_df,paste0(dir.output,"baseline.csv"),row.names=FALSE)
 
# head(out)