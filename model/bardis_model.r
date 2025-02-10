library(deSolve)

# Define the system of differential equations
dynamics_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    
    # Flows
    production <- k_prod * Resources * Economy
    depreciation <- k_deprec * Economy
    bureaucracy_creation <- k_bureaucracy * Economy
    bureaucracy_decay <- k_decay * Bureaucracy 
    pollution_generation <- k_pollution * (Economy + Bureaucracy)
    pollution_abtement <- k_pollution_decay * Pollution
    
    # Rate of change
    dResources <- -production
    dEconomy <- production - depreciation - bureaucracy_creation
    dBureaucracy <- bureaucracy_creation - bureaucracy_decay
    dPollution <- pollution_generation - pollution_abtement
    
    list(c(dResources, dEconomy, dBureaucracy, dPollution))
  })
}


# Initial conditions
state <- c(Resources = 1.0, Economy = 0.1, Bureaucracy = 0.01, Pollution = 0.001)

# Model parameters
parameters <- c(
  k_prod = 0.38*2.0,          # Production rate
  k_deprec = 0.15,        # Depreciation rate
  k_bureaucracy = 0.3,    # Bureaucracy formation rate
  k_decay = 0.5,          # Bureaucracy decay rate
  k_pollution = 0.2*1.5,      # Pollution generation rate
  k_pollution_decay = 0.1 # Pollution decay rate
)

# Time sequence
time <- seq(0, 200, by = 1)

# Solve the system
out <- ode(y = state, times = time, func = dynamics_model, parms = parameters)

# Convert output to data frame
out_df <- as.data.frame(out)
dir.output <- "/Users/edmun/Library/CloudStorage/OneDrive-Personal/Edmundo-ITESM/3.Proyectos/63. Looking back/Tableau/"
write.csv(out_df,paste0(dir.output,"baseline.csv"),row.names=FALSE)
